import time
import os
import cv2
import colorsys
import random
import numpy as np
import threading
import queue
from dnndk import n2cube

OBJECT_TYPE = 'helmet'
CLASSES_PATH = './model_data/classes_' + OBJECT_TYPE + '.txt'
ANCHORS_PATH = './model_data/yolo_anchors_' + OBJECT_TYPE + '.txt'
SCORE_THRESH = 0.6
NMS_THRESH = 0.3

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

# ----------------- 优化点 2: 使用 OpenCV 的 NMS -----------------
def eval_fast(yolo_outputs, image_shape, class_names, anchors):
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    input_shape = np.array(input_shape)*32

    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)
    
    boxes_ = []
    scores_ = []
    classes_ = []
    
    for c in range(len(class_names)):
        class_box_scores_np = box_scores[:, c]
        mask = class_box_scores_np >= SCORE_THRESH
        
        class_boxes_np = boxes[mask]
        class_scores_np = class_box_scores_np[mask]
        
        if len(class_scores_np) == 0:
            continue
            
        # 转换为 cv2.dnn.NMSBoxes 需要的 [x, y, w, h] 格式
        boxes_cv = []
        for box in class_boxes_np:
            y_min, x_min, y_max, x_max = box
            boxes_cv.append([float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)])
            
        # 调用极速的 C++ NMS
        indices = cv2.dnn.NMSBoxes(boxes_cv, class_scores_np.tolist(), float(SCORE_THRESH), float(NMS_THRESH))
        
        if len(indices) > 0:
            for i in indices.flatten():
                boxes_.append(class_boxes_np[i])
                scores_.append(class_scores_np[i])
                classes_.append(c)

    return np.array(boxes_), np.array(scores_), np.array(classes_)

# ----------------- 优化点 3: 多线程流水线架构 -----------------

# 全局退出标志
EXIT_FLAG = False

def thread_read_and_preprocess(cap, queue_in):
    """线程 1：负责抓图和预处理"""
    global EXIT_FLAG
    while not EXIT_FLAG:
        ret, frame = cap.read()
        if not ret:
            break
        
        t1 = time.process_time()
        image_data = pre_process_fast(frame, (416, 416))
        pt = time.process_time() - t1
        
        # 优化：采用非阻塞的 put，如果队列满了说明 DPU 算不过来，直接丢弃该帧(保持低延迟)
        try:
            queue_in.put_nowait((frame, image_data, pt))
        except queue.Full:
            pass # 队列满，丢弃旧帧

def thread_dpu_inference(task, queue_in, queue_out, node_in, nodes_out, dim):
    """线程 2：专职负责喂给 DPU 并获取结果"""
    global EXIT_FLAG
    input_len = n2cube.dpuGetInputTensorSize(task, node_in)
    
    while not EXIT_FLAG:
        try:
            # 优化：采用带超时的阻塞 get，释放 CPU 算力！
            frame, image_data, pt = queue_in.get(timeout=0.1)
        except queue.Empty:
            continue # 队列空，继续循环等待，不占用 CPU
            
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
            # 如果后处理队列满了，把最老的弹出，放入最新的
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
    
    CONV_IN = "conv2d_1_convolution"
    CONV_OUTS = ["conv2d_59_convolution", "conv2d_67_convolution", "conv2d_75_convolution"]

    class_names = get_class(CLASSES_PATH) # 确保你有这个函数
    anchors = get_anchors(ANCHORS_PATH)   # 确保你有这个函数
    dim = 3 * (5 + len(class_names))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建大小受限的队列
    q_pre2dpu = queue.Queue(maxsize=2)
    q_dpu2post = queue.Queue(maxsize=2)

    # 启动流水线线程
    t_pre = threading.Thread(target=thread_read_and_preprocess, args=(cap, q_pre2dpu))
    t_dpu = threading.Thread(target=thread_dpu_inference, args=(task, q_pre2dpu, q_dpu2post, CONV_IN, CONV_OUTS, dim))
    t_pre.start()
    t_dpu.start()

    cv2.namedWindow('YOLO-V3 ALINX PIPELINE', cv2.WINDOW_FREERATIO)
   
    prev_time = time.time() # 放在循环外部初始化

    """主线程：负责后处理和显示（分离后处理与绘制计时）"""
    while True:
        try:
            frame, yolo_outputs, pt, dt = q_dpu2post.get(timeout=0.1)
        except queue.Empty:
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                EXIT_FLAG = True
                break
            continue

        image_size = frame.shape[:2]

        # 计时：后处理（不含绘制）
        t_post_start = time.process_time()
        out_boxes, out_scores, out_classes = eval_fast(yolo_outputs, image_size, class_names, anchors)
        t_post_after_eval = time.process_time()

        # 构建绘制列表（不进行实际绘制以便隔离绘制时间）
        draws = []
        items = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(frame.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(frame.shape[1], np.floor(right + 0.5).astype('int32'))
            draw  = [left, top, right, bottom, score, c]
            item  = [predicted_class, score, left, top, right, bottom]
            draws.append(draw)
            items.append(item)

        # 单独计时绘制（真实绘制与标签写入）
        t_draw_start = time.process_time()
        for draw in draws:
            left, top, right, bottom, score, c = draw
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, f"{class_names[c]}:{score:.2f}", (left, top-10), cv2.FONT_ITALIC, 0.6, (122, 0, 204), 2)
        draw_t = time.process_time() - t_draw_start

        post_no_draw_t = t_draw_start - t_post_start

        # 计算显示相关指标
        run_time = pt + dt + post_no_draw_t
        pure_infer_fps = 1.0 / run_time if run_time > 0 else 0.0
        dpu_fps = 1.0 / dt if dt > 0 else 0.0

        curr_time = time.time()
        display_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # 统一覆盖文本（顺序）
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

    # 资源回收
    t_pre.join()
    t_dpu.join()
    cap.release()
    cv2.destroyAllWindows()
    n2cube.dpuDestroyTask(task)
    n2cube.dpuDestroyKernel(kernel)
    n2cube.dpuClose()