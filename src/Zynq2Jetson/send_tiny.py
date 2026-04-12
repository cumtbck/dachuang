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

# ===================== 用户配置 =====================
OBJECT_TYPE = 'mix'
INPUT_SIZE = (512, 512)  # YOLOv3-Tiny训练/量化输入尺寸
KERNEL_CONV = 'tf_yolov3_' + OBJECT_TYPE + '_tiny'
CONV_INPUT_NODE = "conv2d_1_convolution"
CONV_OUTPUT_NODES = [
    "conv2d_10_convolution",  # head1
    "conv2d_13_convolution",  # head2
]

# ===================== 工具函数 =====================
def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    image_resized = cv2.resize(image, (nw, nh))
    new_image = np.ones((h, w, 3), dtype=np.uint8) * 128
    top, left = (h-nh)//2, (w-nw)//2
    new_image[top:top+nh, left:left+nw, :] = image_resized
    return new_image

def pre_process(image, input_size):
    image = image[..., ::-1]  # BGR->RGB
    image_data = letterbox_image(image, tuple(reversed(input_size)))
    image_data = np.array(image_data, dtype=np.float32) / 255.0
    return np.expand_dims(image_data, 0)

# 简单 NMS/后处理占位函数（可替换为真实 eval 函数）
def post_process(yolo_outputs):
    # YOLOv3-Tiny两个 head的输出可以在此解析
    # 这里模拟返回对象列表
    objects = []  # 每个对象格式: [x1, y1, x2, y2, score, class_id]
    return objects

# ===================== 主函数 =====================
def main():
    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
    task = n2cube.dpuCreateTask(kernel, 0)

    cap = cv2.VideoCapture(0)
    print("[INFO] Zynq YOLOv3-Tiny Inference Started...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 1. 前处理 ---
        t_pre_start = time.perf_counter()
        img_data = pre_process(frame, INPUT_SIZE)
        t_pre = (time.perf_counter() - t_pre_start) * 1000

        # --- 2. DPU 推理 ---
        t_dpu_start = time.perf_counter()
        input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
        n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, img_data, input_len)
        n2cube.dpuRunTask(task)

        # 获取 Tiny 两个 head 输出（可 reshape，具体 grid 根据量化确定）
        yolo_outputs = []
        for i, node in enumerate(CONV_OUTPUT_NODES):
            out_size = n2cube.dpuGetOutputTensorSize(task, node)
            out_data = n2cube.dpuGetOutputTensorInHWCFP32(task, node, out_size)
            if i == 0:
                grid_h, grid_w = 16,16
            else:
                grid_h, grid_w = 32,32
            dim = 3*(3+5)  # 假设3类，anchors=3
            yolo_outputs.append(out_data.reshape(1, grid_h, grid_w, dim))
        t_dpu = (time.perf_counter() - t_dpu_start) * 1000

        # --- 3. 后处理 ---
        t_post_start = time.perf_counter()
        objects = post_process(yolo_outputs)  # 可以替换为真实 eval_yolo
        t_post = (time.perf_counter() - t_post_start) * 1000

        # --- 4. 打包并发送 UDP ---
        send_ts = time.time()
        payload = {
            "send_ts": send_ts,
            "pre_ms": round(t_pre,2),
            "dpu_ms": round(t_dpu,2),
            "post_ms": round(t_post,2),
            "count": len(objects)
        }
        sock.sendto(json.dumps(payload).encode(), (JETSON_IP, UDP_PORT))

    n2cube.dpuDestroyTask(task)
    n2cube.dpuClose()
    cap.release()

if __name__ == "__main__":
    main()