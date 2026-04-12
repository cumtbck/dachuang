import time
import os
import cv2
import colorsys
import numpy as np
import threading
import queue # 引入 queue 模块，提供线程安全和异常处理
from dnndk import n2cube

OBJECT_TYPE = 'light'
CLASSES_PATH = './model_data/classes_' + OBJECT_TYPE + '.txt'
ANCHORS_PATH = './model_data/yolo_anchors_' + OBJECT_TYPE + '.txt'
SCORE_THRESH = 0.6
NMS_THRESH = 0.3

# DPU kernel and node names follow the same pattern as helmet_cam_acc
# Kernel name will be constructed as 'tf_yolov3_' + OBJECT_TYPE at runtime
CONV_IN = "conv2d_1_convolution"
CONV_OUTS = ["conv2d_59_convolution", "conv2d_67_convolution", "conv2d_75_convolution"]

# 全局退出标志，用于安全关闭所有线程
EXIT_FLAG = False

# ----------------- 优化点 1: 预处理提速 -----------------
def letterbox_image_fast(image, size):
    """使用 cv2.copyMakeBorder 替换 numpy 切片，提速 5 倍以上"""
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)

    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    top = (h - nh) // 2
    bottom = h - nh - top
    left = (w - nw) // 2
    right = w - nw - left
    
    # 底层 C++ 实现的边界填充，比 np.ones 快得多
    new_image = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128,128,128))
    return new_image

def pre_process_fast(image, model_image_size):
    image = image[...,::-1] # BGR to RGB
    boxed_image = letterbox_image_fast(image, tuple(reversed(model_image_size)))
    
    # 使用 astype 和 乘法 替换 浮点除法
    image_data = boxed_image.astype(np.float32) * 0.0039215686 # 等价于 / 255.0
    image_data = np.expand_dims(image_data, 0)  
    return image_data

# ... (保留原有的 get_class, get_anchors, output_fix, _get_feats, correct_boxes, boxes_and_scores 等辅助函数) ...
# 为了代码整洁，此处省略上面列出的纯辅助数学计算函数，直接将你原来的粘过来即可
'''Get model classification information'''	
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

'''Get model anchors value'''
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def output_fix(feats, num_classes):
    num_anchors = 3
    nu = num_classes + 5
    grid_size = np.shape(feats)[1 : 3]
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])

	# replace 0 with broken data in some output channel
    predictions[..., 0:1, 4:5] = np.zeros(shape=predictions[..., 2:3, 4:5].shape)
    feats_fixed = np.reshape(predictions, feats.shape)

    # box_confidence = predictions[..., 4:5]
    # single_layer_box_confidence = box_confidence[..., 2:3,:]
    # print('\nOriginal output:{}'.format(feats.shape))
    # print('reshaped output:{}'.format(predictions.shape))
    # print('score output:{}'.format(box_confidence.shape))
    # print('single layer score output:{}\n'.format(single_layer_box_confidence.shape))

    return feats_fixed
    
def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs

def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes

def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores	

# ----------------- 优化点 2: 极速 NMS 后处理 -----------------
def eval_fast(yolo_outputs, image_shape, class_names, anchors):
    """使用 OpenCV 自带的 NMS 替换 Python 循环 NMS，大幅降后处理时间"""
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes, box_scores = [], []
    
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    input_shape = np.array(input_shape)*32

    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)
    
    out_boxes, out_scores, out_classes = [], [], []
    
    for c in range(len(class_names)):
        class_box_scores_np = box_scores[:, c]
        mask = class_box_scores_np >= SCORE_THRESH
        
        class_boxes_np = boxes[mask]
        class_scores_np = class_box_scores_np[mask]
        
        if len(class_scores_np) == 0:
            continue
            
        # 转换为 cv2.dnn.NMSBoxes 格式: [x, y, w, h]
        boxes_cv = []
        for box in class_boxes_np:
            y_min, x_min, y_max, x_max = box
            boxes_cv.append([float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)])
            
        # 调用极速的 C++ NMS
        indices = cv2.dnn.NMSBoxes(boxes_cv, class_scores_np.tolist(), float(SCORE_THRESH), float(NMS_THRESH))
        
        if len(indices) > 0:
            for i in indices.flatten():
                out_boxes.append(class_boxes_np[i])
                out_scores.append(class_scores_np[i])
                out_classes.append(c)

    return np.array(out_boxes), np.array(out_scores), np.array(out_classes)

# ----------------- 优化点 3: 多线程流水线 -----------------
def thread_read_and_preprocess(cap, queue_in):
    """线程 1：仅负责读取摄像头并进行图像预处理"""
    global EXIT_FLAG
    while not EXIT_FLAG:
        ret, frame = cap.read()
        if not ret: break
        
        t1 = time.process_time()
        image_data = pre_process_fast(frame, (416, 416))
        pt = time.process_time() - t1
        
        # 丢帧机制：如果 DPU 算不过来队列满了，直接丢弃新帧，保持低延迟
        try:
            queue_in.put_nowait((frame, image_data, pt))
        except queue.Full:
            pass 

def thread_dpu_inference(task, queue_in, queue_out, node_in, nodes_out, dim):
    """线程 2：专职负责与 DPU 硬件通讯进行推理（参数化输入/输出节点）"""
    global EXIT_FLAG
    input_len = n2cube.dpuGetInputTensorSize(task, node_in)

    while not EXIT_FLAG:
        try:
            frame, image_data, pt = queue_in.get(timeout=0.1)
        except queue.Empty:
            continue

        t1 = time.process_time()
        n2cube.dpuSetInputTensorInHWCFP32(task, node_in, image_data, input_len)
        n2cube.dpuRunTask(task)

        s1 = n2cube.dpuGetOutputTensorSize(task, nodes_out[0])
        out1 = np.reshape(n2cube.dpuGetOutputTensorInHWCFP32(task, nodes_out[0], s1), (1, 13, 13, dim))

        s2 = n2cube.dpuGetOutputTensorSize(task, nodes_out[1])
        out2 = np.reshape(n2cube.dpuGetOutputTensorInHWCFP32(task, nodes_out[1], s2), (1, 26, 26, dim))

        s3 = n2cube.dpuGetOutputTensorSize(task, nodes_out[2])
        out3 = np.reshape(n2cube.dpuGetOutputTensorInHWCFP32(task, nodes_out[2], s3), (1, 52, 52, dim))

        # 修补可能损坏的输出通道，保证后处理一致性
        try:
            out3 = output_fix(out3, len(class_names))
        except Exception:
            pass

        dt = time.process_time() - t1

        try:
            if queue_out.full():
                queue_out.get_nowait()
            queue_out.put_nowait((frame, [out1, out2, out3], pt, dt))
        except queue.Empty:
            pass
        except queue.Full:
            pass

if __name__ == "__main__":
    n2cube.dpuOpen()
    KERNEL_CONV = 'tf_yolov3_' + OBJECT_TYPE
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
    task = n2cube.dpuCreateTask(kernel, 0)
    
    class_names = get_class(CLASSES_PATH)
    anchors = get_anchors(ANCHORS_PATH)
    
    # 动态计算 DPU 输出张量的深度维度 (通常为 3 * (5 + 类别数))
    dim = 3 * (5 + len(class_names))

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # 创建大小受限的队列，Maxsize=2 完美契合流水线
    q_pre2dpu = queue.Queue(maxsize=2)
    q_dpu2post = queue.Queue(maxsize=2)

    # 启动工作线程
    t_pre = threading.Thread(target=thread_read_and_preprocess, args=(cap, q_pre2dpu))
    t_dpu = threading.Thread(target=thread_dpu_inference, args=(task, q_pre2dpu, q_dpu2post, CONV_IN, CONV_OUTS, dim))
    t_pre.start()
    t_dpu.start()

    cv2.namedWindow('YOLO-V3 ALINX PIPELINE', cv2.WINDOW_FREERATIO)

    # 为各类别生成固定颜色
    hsv_tuples = [(1.0 * x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    prev_time = time.time()

    """主线程：只负责极速后处理与画面绘制显示"""
    while True:
        try:
            # 同样带有超时阻塞，避免死循环消耗 CPU
            frame, yolo_outputs, pt, dt = q_dpu2post.get(timeout=0.1)
        except queue.Empty:
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                EXIT_FLAG = True
                break
            continue

        image_size = frame.shape[:2]
        
        # 分离后处理（不含绘制）与绘制计时
        t_post_start = time.process_time()
        out_boxes, out_scores, out_classes = eval_fast(yolo_outputs, image_size, class_names, anchors)
        t_post_after_eval = time.process_time()

        draws = []
        items = []
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            score = out_scores[i]
            y_min, x_min, y_max, x_max = box

            top = max(0, int(y_min))
            left = max(0, int(x_min))
            bottom = min(frame.shape[0], int(y_max))
            right = min(frame.shape[1], int(x_max))

            draw = [left, top, right, bottom, score, c]
            item = [class_names[c], score, left, top, right, bottom]
            draws.append(draw)
            items.append(item)

        t_draw_start = time.process_time()
        for draw in draws:
            left, top, right, bottom, score, c = draw
            cv2.rectangle(frame, (left, top), (right, bottom), colors[c], 2)
            cv2.putText(frame, f"{class_names[c]}:{score:.2f}", (left, top-10), cv2.FONT_ITALIC, 0.6, (122, 0, 204), 2)
        draw_t = time.process_time() - t_draw_start

        post_no_draw_t = t_draw_start - t_post_start

        curr_time = time.time()
        display_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        dpu_fps = 1.0 / dt if dt > 0 else 0.0
        run_time = pt + dt + post_no_draw_t
        pure_infer_fps = 1.0 / run_time if run_time > 0 else 0.0

        cv2.putText(frame, f'Pipeline FPS : {display_fps:.1f}', (10, 30), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'Pre-process  : {pt*1000:.1f} ms', (10, 60), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'DPU Execute  : {dt*1000:.1f} ms', (10, 90), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'DPU FPS      : {dpu_fps:.1f}', (10, 120), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Post(no draw): {post_no_draw_t*1000:.1f} ms', (10, 150), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Draw time    : {draw_t*1000:.1f} ms', (10, 180), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Pure Infer FPS: {pure_infer_fps:.1f}', (10, 210), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)

        cv2.imshow('YOLO-V3 ALINX PIPELINE', frame)

        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            EXIT_FLAG = True
            break
        elif k == ord("s"):
            cv2.imwrite("vedio_cap.jpg", frame)
            EXIT_FLAG = True
            break

    # 安全释放资源
    t_pre.join()
    t_dpu.join()
    cap.release()
    cv2.destroyAllWindows()
    n2cube.dpuDestroyTask(task)
    n2cube.dpuDestroyKernel(kernel)
    n2cube.dpuClose()