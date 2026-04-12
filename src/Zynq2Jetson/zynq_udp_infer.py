#!/usr/bin/env python3
import argparse
import json
import socket
import struct
import time

import cv2
import numpy as np

try:
    from dnndk import n2cube
except ImportError:  # pragma: no cover - only on Zynq
    n2cube = None

HEADER_FORMAT = "!2sQHH"
HEADER_MAGIC = b"MC"


class FrameAssembler(object):
    def __init__(self, max_age_sec=0.5):
        self.max_age_sec = max_age_sec
        self.frames = {}

    def add_packet(self, stamp_ns, chunk_index, total_chunks, payload):
        now = time.time()
        frame = self.frames.get(stamp_ns)
        if frame is None:
            frame = {"total": total_chunks, "chunks": {}, "last_seen": now}
            self.frames[stamp_ns] = frame
        frame["chunks"][chunk_index] = payload
        frame["last_seen"] = now

        if len(frame["chunks"]) == frame["total"]:
            data = b"".join(frame["chunks"][index] for index in range(frame["total"]))
            self.frames.pop(stamp_ns, None)
            return stamp_ns, data

        self._purge_old(now)
        return None

    def _purge_old(self, now):
        expired = [key for key, frame in self.frames.items() if now - frame["last_seen"] > self.max_age_sec]
        for key in expired:
            self.frames.pop(key, None)


class YoloDpuRunner(object):
    def __init__(self, module, score_thresh=None):
        self.module = module
        if score_thresh is not None:
            self.module.SCORE_THRESH = score_thresh
        self.class_names = self.module.get_class(self.module.CLASSES_PATH)
        self.anchors = self.module.get_anchors(self.module.ANCHORS_PATH)
        self.kernel = n2cube.dpuLoadKernel(self.module.KERNEL_CONV)
        self.task = n2cube.dpuCreateTask(self.kernel, 0)
        self.nn_obj_info_dim = 3 * (5 + len(self.class_names))

    def infer(self, image):
        image_size = image.shape[:2]
        image_data = self.module.pre_process(image, (416, 416))
        image_data = np.array(image_data, dtype=np.float32)
        input_len = n2cube.dpuGetInputTensorSize(self.task, self.module.CONV_INPUT_NODE)
        n2cube.dpuSetInputTensorInHWCFP32(self.task, self.module.CONV_INPUT_NODE, image_data, input_len)
        n2cube.dpuRunTask(self.task)

        outputs = []
        for output_node, shape in (
            (self.module.CONV_OUTPUT_NODE1, (1, 13, 13, self.nn_obj_info_dim)),
            (self.module.CONV_OUTPUT_NODE2, (1, 26, 26, self.nn_obj_info_dim)),
            (self.module.CONV_OUTPUT_NODE3, (1, 52, 52, self.nn_obj_info_dim)),
        ):
            output_size = n2cube.dpuGetOutputTensorSize(self.task, output_node)
            output_raw = n2cube.dpuGetOutputTensorInHWCFP32(self.task, output_node, output_size)
            outputs.append(np.reshape(output_raw, shape))

        out_boxes, out_scores, out_classes = self.module.eval(outputs, image_size, self.class_names, self.anchors)
        detections = []

        for index, class_id in enumerate(out_classes):
            class_name = self.class_names[class_id]
            score = float(out_scores[index])
            top, left, bottom, right = out_boxes[index]
            detection = {
                "class": class_name,
                "score": score,
                "bbox": [int(left), int(top), int(right), int(bottom)],
            }
            detections.append(detection)
        return detections

    def close(self):
        n2cube.dpuDestroyTask(self.task)
        n2cube.dpuDestroyKernel(self.kernel)


def parse_args():
    parser = argparse.ArgumentParser(description="Zynq UDP inference server")
    parser.add_argument("--listen-port", type=int, default=5000)
    parser.add_argument("--send-host", default="192.168.1.2")
    parser.add_argument("--send-port", type=int, default=5001)
    parser.add_argument("--score-thresh", type=float, default=0.6)
    parser.add_argument("--max-age", type=float, default=0.5)
    parser.add_argument("--disable-helmet", action="store_true")
    parser.add_argument("--disable-light", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    assembler = FrameAssembler(max_age_sec=args.max_age)
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_socket.bind(("", args.listen_port))

    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    detectors = []
    if n2cube is not None:
        n2cube.dpuOpen()
        import helmet_cam
        import light_cam

        if not args.disable_helmet:
            detectors.append(YoloDpuRunner(helmet_cam, score_thresh=args.score_thresh))
        if not args.disable_light:
            detectors.append(YoloDpuRunner(light_cam, score_thresh=args.score_thresh))
    else:
        print("[WARN] dnndk not found; running in passthrough mode")

    try:
        while True:
            packet, _address = listen_socket.recvfrom(65535)
            if len(packet) < struct.calcsize(HEADER_FORMAT):
                continue
            magic, stamp_ns, chunk_index, total_chunks = struct.unpack(
                HEADER_FORMAT, packet[: struct.calcsize(HEADER_FORMAT)]
            )
            if magic != HEADER_MAGIC:
                continue

            payload = packet[struct.calcsize(HEADER_FORMAT) :]
            result = assembler.add_packet(stamp_ns, chunk_index, total_chunks, payload)
            if result is None:
                continue

            stamp_ns, jpeg_data = result
            image_array = np.frombuffer(jpeg_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                continue

            detections = []
            for detector in detectors:
                detections.extend(detector.infer(image))

            message = json.dumps({"stamp_ns": stamp_ns, "detections": detections})
            send_socket.sendto(message.encode("utf-8"), (args.send_host, args.send_port))
    finally:
        for detector in detectors:
            detector.close()
        if n2cube is not None:
            n2cube.dpuClose()


if __name__ == "__main__":
    main()
