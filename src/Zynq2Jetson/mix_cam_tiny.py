import time
import cv2
import numpy as np
from dnndk import n2cube
import colorsys
import random

# ===================== 用户配置区 =====================
OBJECT_TYPE = 'mix'
CLASSES_PATH = './model_data/classes_' + OBJECT_TYPE + '.txt'
ANCHORS_PATH = './model_data/tiny_yolo_anchors_' + OBJECT_TYPE + '.txt'

KERNEL_CONV = 'tf_yolov3_' + OBJECT_TYPE + '_tiny'
CONV_INPUT_NODE = "conv2d_1_convolution"
CONV_OUTPUT_NODES = [
    "conv2d_10_convolution",  # head1
    "conv2d_13_convolution",  # head2
]

INPUT_SIZE = (512, 512)  # 训练/量化输入尺寸
SCORE_THRESH = 0.1
NMS_THRESH = 0.3
DEBUG = False

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
    image = image[..., ::-1]
    image_data = letterbox_image(image, tuple(reversed(input_size)))
    image_data = np.array(image_data, dtype=np.float32) / 255.0
    return np.expand_dims(image_data, 0)

def draw_bbox(image, bboxes, classes):
    num_classes = len(classes)
    hsv_tuples = [(1.0*x/num_classes,1.,1.) for x in range(num_classes)]
    colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255))
              for c in map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)]
    random.seed(0)
    random.shuffle(colors)
    for bbox in bboxes:
        coor = np.array(bbox[:4], dtype=np.int32)
        class_id = int(bbox[5])
        color = colors[class_id]
        cv2.rectangle(image, (coor[0], coor[1]), (coor[2], coor[3]), color, 2)
        cv2.putText(image, f"{classes[class_id]}:{bbox[4]:.2f}",
                    (coor[0], coor[1]-5), cv2.FONT_ITALIC, 0.5, color, 1)
    return image

def nms_boxes(boxes, scores, nms_thresh):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        ovr = inter/(areas[i]+areas[order[1:]]-inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds+1]
    return keep

def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1 / (1 + np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_conf = 1 / (1 + np.exp(-predictions[..., 4:5]))
    box_class_probs = 1 / (1 + np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_conf, box_class_probs

def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype=np.float32)
    image_shape = np.array(image_shape, dtype=np.float32)
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
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_conf, box_class_probs = _get_feats(feats, anchors, num_classes, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = boxes.reshape(-1,4)
    scores = (box_conf*box_class_probs).reshape(-1,num_classes)
    return boxes, scores

def eval_yolo(yolo_outputs, image_shape, class_names, anchors):
    boxes_list, scores_list, classes_list = [], [], []
    num_classes = len(class_names)
    input_shape = np.shape(yolo_outputs[0])[1:3]
    input_shape = np.array(input_shape) * 32

    for i, feats in enumerate(yolo_outputs):
        boxes, scores = boxes_and_scores(feats, anchors[i], num_classes, input_shape, image_shape)
        for c in range(num_classes):
            class_boxes = boxes[scores[:,c] >= SCORE_THRESH]
            class_scores = scores[:,c][scores[:,c] >= SCORE_THRESH]
            if len(class_scores)==0: continue
            keep = nms_boxes(class_boxes, class_scores, NMS_THRESH)
            boxes_list.extend(class_boxes[keep])
            scores_list.extend(class_scores[keep])
            classes_list.extend([c]*len(keep))
    if DEBUG:
        print(f"[DEBUG] SCORE_THRESH={SCORE_THRESH:.4f}")
        print(f"[DEBUG] Total objects detected: {len(boxes_list)}")
    return np.array([[x[0], x[1], x[2], x[3], s, c]
                     for x,s,c in zip(boxes_list,scores_list,classes_list)])

def eval(yolo_outputs, image_shape, class_names, anchors):
    if isinstance(anchors, np.ndarray):
        if anchors.shape[0] < 6:
            raise ValueError("mix_cam_tiny requires at least 6 anchors")
        anchors = [anchors[[0, 1, 2]], anchors[[3, 4, 5]]]

    bboxes = eval_yolo(yolo_outputs, image_shape, class_names, anchors)
    bboxes = np.asarray(bboxes)
    if bboxes.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)

    return bboxes[:, :4], bboxes[:, 4], bboxes[:, 5].astype(np.int32)

# ===================== 主函数 =====================
if __name__ == "__main__":
    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
    task = n2cube.dpuCreateTask(kernel, 0)

    class_names = [c.strip() for c in open(CLASSES_PATH)]
    anchors_all = np.array([float(x) for x in open(ANCHORS_PATH).readline().split(',')]).reshape(-1,2)
    anchors_per_head = [anchors_all[[0,1,2]], anchors_all[[3,4,5]]]  # Tiny 两个 head

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        image_h, image_w = frame.shape[:2]

        # 前处理
        t_pre_start = time.perf_counter()
        img_data = pre_process(frame, INPUT_SIZE)
        t_pre = (time.perf_counter()-t_pre_start)*1000

        # DPU 推理
        t_dpu_start = time.perf_counter()
        input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)
        n2cube.dpuSetInputTensorInHWCFP32(task, CONV_INPUT_NODE, img_data, input_len)
        n2cube.dpuRunTask(task)

        yolo_outputs = []
        for i, node in enumerate(CONV_OUTPUT_NODES):
            out_size = n2cube.dpuGetOutputTensorSize(task, node)
            out_data = n2cube.dpuGetOutputTensorInHWCFP32(task, node, out_size)
            grid_h, grid_w = (16,16) if i==0 else (32,32)
            dim = 3*(len(class_names)+5)
            yolo_outputs.append(out_data.reshape(1, grid_h, grid_w, dim))
            print(f"[DEBUG] DPU head {i}: node={node}, reshaped={yolo_outputs[-1].shape}")

        t_dpu = (time.perf_counter()-t_dpu_start)*1000

        # 后处理
        t_post_start = time.perf_counter()
        bboxes = eval_yolo(yolo_outputs, (image_h,image_w), class_names, anchors_per_head)
        t_post = (time.perf_counter()-t_post_start)*1000

        print(f"[INFO] Frame pre={t_pre:.2f}ms, DPU={t_dpu:.2f}ms, post={t_post:.2f}ms, objs={len(bboxes)}")

        # 绘制
        result_img = draw_bbox(frame, bboxes, class_names)
        cv2.imshow("YOLOv3-Tiny Adaptive", result_img)

        if cv2.waitKey(1) & 0xFF==27: break

    cap.release()
    cv2.destroyAllWindows()
    n2cube.dpuDestroyTask(task)
    n2cube.dpuClose()