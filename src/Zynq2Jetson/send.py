import time
import socket
import json
import cv2
import numpy as np
from dnndk import n2cube

# --- 核心参数 ---
JETSON_IP = '10.42.0.1'
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

'''resize image with unchanged aspect ratio using padding'''
def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    #print("image scale:",scale)

    nw = int(iw*scale)
    nh = int(ih*scale)

    #print("image wide:",nw)
    #print("image height:",nh)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image
'''image preprocessing'''
def pre_process(image, model_image_size):
    image = image[...,::-1]
    image_h, image_w, _ = image.shape

    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    return image_data

def main():
    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel('tf_yolov3_light')
    task = n2cube.dpuCreateTask(kernel, 0)

    cap = cv2.VideoCapture(0)
    print("[INFO] Zynq Inference Started...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 预处理耗时记录
        t_pre_start = time.perf_counter()
        img_data = pre_process(frame, (416, 416)) # 假设你已定义该函数
        t_pre = (time.perf_counter() - t_pre_start) * 1000

        # 2. DPU 推理耗时记录
        t_dpu_start = time.perf_counter()
        n2cube.dpuSetInputTensorInHWCFP32(task, "conv2d_1_convolution", img_data, 416*416*3)
        n2cube.dpuRunTask(task)
        # 获取输出张量的逻辑 (省略具体 getTensor 代码)
        t_dpu = (time.perf_counter() - t_dpu_start) * 1000

        # 3. 后处理耗时记录 (NMS等)
        t_post_start = time.perf_counter()
        # objects = eval_fast(outs, ...)
        objects = [] # 模拟结果
        t_post = (time.perf_counter() - t_post_start) * 1000

        # 4. 打包数据并附带发送时间戳 (使用墙上时钟用于同步)
        send_ts = time.time()
        payload = {
            "send_ts": send_ts,
            "pre_ms": round(t_pre, 2),
            "dpu_ms": round(t_dpu, 2),
            "post_ms": round(t_post, 2),
            "count": len(objects)
        }

        sock.sendto(json.dumps(payload).encode(), (JETSON_IP, UDP_PORT))

    n2cube.dpuDestroyTask(task)
    n2cube.dpuClose()

if __name__ == "__main__":
    main()