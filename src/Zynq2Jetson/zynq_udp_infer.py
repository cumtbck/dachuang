#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import socket
import struct
import sys
import time
from typing import Any

import cv2
import numpy as np

try:
    from dnndk import n2cube as _n2cube
except ImportError:  # pragma: no cover - only on Zynq
    _n2cube = None

n2cube: Any = _n2cube

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

HEADER_FORMAT = "!2sQQHH"
HEADER_MAGIC = b"MC"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_REASONABLE_CHUNKS = 1024
DEFAULT_LISTEN_PORT = 5000
DEFAULT_SEND_HOST = "10.42.0.1"
DEFAULT_SEND_PORT = 5001
DEFAULT_MAX_AGE = 0.5
DEFAULT_PROFILES = "mix,mix_tiny"
DEFAULT_DEBUG_PRINT_INTERVAL = 1.0
DEFAULT_LATENCY_OFFSET_MS = 0.2
DEFAULT_LATENCY_MIN_SAMPLE_MS = 0.1

PROFILE_CONFIGS = {
    "mix": {
        "module_name": "mix_cam",
        "kernel_name": "tf_yolov3_mix",
        "input_size": (416, 416),
        "input_node": "conv2d_1_convolution",
        "output_nodes": [
            "conv2d_59_convolution",
            "conv2d_67_convolution",
            "conv2d_75_convolution",
        ],
        "output_strides": [32, 16, 8],
        "classes_path": "./model_data/classes_mix.txt",
        "anchors_path": "./model_data/yolo_anchors_mix.txt",
        "swap_xy": True,
    },
    "mix_tiny": {
        "module_name": "mix_cam_tiny",
        "kernel_name": "tf_yolov3_mix_tiny",
        "input_size": (512, 512),
        "input_node": "conv2d_1_convolution",
        "output_nodes": ["conv2d_10_convolution", "conv2d_13_convolution"],
        "output_strides": [32, 16],
        "classes_path": "./model_data/classes_mix.txt",
        "anchors_path": "./model_data/tiny_yolo_anchors_mix.txt",
        "swap_xy": False,
    },
}


def _time_ns():
    try:
        return time.time_ns()
    except AttributeError:
        return int(time.time() * 1.0e9)


def _perf_counter_ns():
    try:
        return time.perf_counter_ns()
    except AttributeError:
        return int(time.perf_counter() * 1.0e9)


def ns_to_ms(delta_ns):
    return float(delta_ns) / 1.0e6


def parse_profiles(value):
    profiles = [profile.strip() for profile in value.split(",") if profile.strip()]
    if not profiles:
        profiles = ["mix", "mix_tiny"]
    unknown_profiles = [profile for profile in profiles if profile not in PROFILE_CONFIGS]
    if unknown_profiles:
        raise ValueError("Unsupported profile(s): {}".format(", ".join(unknown_profiles)))
    return profiles


def load_module(module_name):
    return importlib.import_module(module_name)


def load_classes(classes_path):
    with open(classes_path) as file_handle:
        return [line.strip() for line in file_handle if line.strip()]


def load_anchors(anchors_path):
    with open(anchors_path) as file_handle:
        values = file_handle.readline()
    anchors = [float(value) for value in values.split(",") if value.strip()]
    return np.array(anchors, dtype=np.float32).reshape(-1, 2)


def letterbox_image(image, size):
    image_height, image_width, _ = image.shape
    target_width, target_height = size
    scale = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 128
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    new_image[top : top + new_height, left : left + new_width, :] = image_resized
    return new_image


def pre_process(image, model_image_size):
    image = image[..., ::-1]
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, "Multiples of 32 required"
        assert model_image_size[1] % 32 == 0, "Multiples of 32 required"
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        image_height, image_width, _ = image.shape
        new_image_size = (image_width - (image_width % 32), image_height - (image_height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype=np.float32)
    image_data /= 255.0
    return np.expand_dims(image_data, 0)


class FrameAssembler(object):
    def __init__(self, max_age_sec=0.5):
        self.max_age_ns = int(max_age_sec * 1.0e9)
        self.frames = {}

    def add_packet(self, packet_info, payload):
        stamp_ns = packet_info["stamp_ns"]
        recv_ns = packet_info["recv_ns"]
        frame = self.frames.get(stamp_ns)
        if frame is None:
            frame = {
                "stamp_ns": stamp_ns,
                "send_ts_ns": packet_info.get("send_ts_ns", 0),
                "total_chunks": packet_info["total_chunks"],
                "chunks": {},
                "first_seen_ns": recv_ns,
                "last_seen_ns": recv_ns,
            }
            self.frames[stamp_ns] = frame
        else:
            if not frame["send_ts_ns"] and packet_info.get("send_ts_ns", 0):
                frame["send_ts_ns"] = packet_info["send_ts_ns"]
            frame["total_chunks"] = max(frame["total_chunks"], packet_info["total_chunks"])
            frame["last_seen_ns"] = recv_ns

        frame["chunks"][packet_info["chunk_index"]] = payload
        frame["last_seen_ns"] = recv_ns

        if len(frame["chunks"]) == frame["total_chunks"]:
            jpeg_data = b"".join(frame["chunks"][index] for index in range(frame["total_chunks"]))
            self.frames.pop(stamp_ns, None)
            return {
                "stamp_ns": stamp_ns,
                "send_ts_ns": frame["send_ts_ns"],
                "server_recv_ns": recv_ns,
                "frame_first_seen_ns": frame["first_seen_ns"],
                "frame_last_seen_ns": frame["last_seen_ns"],
                "frame_age_ns": recv_ns - frame["first_seen_ns"],
                "total_chunks": frame["total_chunks"],
                "jpeg_data": jpeg_data,
            }

        self._purge_old(recv_ns)
        return None

    def _purge_old(self, now_ns):
        expired = [key for key, frame in self.frames.items() if now_ns - frame["last_seen_ns"] > self.max_age_ns]
        for key in expired:
            self.frames.pop(key, None)


class LatencyTracker(object):
    def __init__(self, min_sample_ms=DEFAULT_LATENCY_MIN_SAMPLE_MS, offset_ms=DEFAULT_LATENCY_OFFSET_MS):
        self.min_sample_ms = float(min_sample_ms)
        self.offset_ms = float(offset_ms)
        self.min_latency_ms = None

    def observe(self, raw_latency_ms):
        if raw_latency_ms <= self.min_sample_ms:
            return
        if self.min_latency_ms is None or raw_latency_ms < self.min_latency_ms:
            self.min_latency_ms = raw_latency_ms

    def calibrated(self, raw_latency_ms):
        baseline = self.min_latency_ms if self.min_latency_ms is not None else raw_latency_ms
        calibrated_latency = raw_latency_ms - baseline + self.offset_ms
        if calibrated_latency < self.min_sample_ms:
            calibrated_latency = self.min_sample_ms
        return calibrated_latency


class DetectorRunner(object):
    def __init__(self, profile_name, module, config, score_thresh=None, nms_thresh=None):
        self.profile_name = profile_name
        self.module = module
        self.config = config
        self.input_size = tuple(config["input_size"])
        self.kernel_name = config["kernel_name"]
        self.input_node = config["input_node"]
        self.output_nodes = list(config["output_nodes"])
        self.output_strides = list(config["output_strides"])
        self.swap_xy = bool(config.get("swap_xy", False))
        self.class_names = load_classes(config["classes_path"])
        self.anchors = load_anchors(config["anchors_path"])
        self.num_classes = len(self.class_names)
        self.output_dim = 3 * (self.num_classes + 5)

        if score_thresh is not None:
            if hasattr(self.module, "SCORE_THRESH"):
                self.module.SCORE_THRESH = float(score_thresh)
            if hasattr(self.module, "MIN_SCORE_THRESH"):
                self.module.MIN_SCORE_THRESH = float(score_thresh)
        if nms_thresh is not None and hasattr(self.module, "NMS_THRESH"):
            self.module.NMS_THRESH = float(nms_thresh)

        self.kernel = n2cube.dpuLoadKernel(self.kernel_name)
        self.task = n2cube.dpuCreateTask(self.kernel, 0)

    def _pre_process(self, image):
        if hasattr(self.module, "pre_process"):
            return self.module.pre_process(image, self.input_size)
        return pre_process(image, self.input_size)

    def _format_bbox(self, box):
        if self.swap_xy:
            return [int(box[1]), int(box[0]), int(box[3]), int(box[2])]
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

    def infer(self, image):
        pre_start_ns = _perf_counter_ns()
        image_data = self._pre_process(image)
        pre_ms = ns_to_ms(_perf_counter_ns() - pre_start_ns)

        dpu_start_ns = _perf_counter_ns()
        input_length = n2cube.dpuGetInputTensorSize(self.task, self.input_node)
        image_data = np.asarray(image_data, dtype=np.float32)
        n2cube.dpuSetInputTensorInHWCFP32(self.task, self.input_node, image_data, input_length)
        n2cube.dpuRunTask(self.task)

        outputs = []
        for output_node, stride in zip(self.output_nodes, self.output_strides):
            output_size = n2cube.dpuGetOutputTensorSize(self.task, output_node)
            output_raw = n2cube.dpuGetOutputTensorInHWCFP32(self.task, output_node, output_size)
            output_shape = (
                1,
                int(self.input_size[0] // stride),
                int(self.input_size[1] // stride),
                self.output_dim,
            )
            outputs.append(np.reshape(output_raw, output_shape))
        dpu_ms = ns_to_ms(_perf_counter_ns() - dpu_start_ns)

        post_start_ns = _perf_counter_ns()
        image_shape = image.shape[:2]
        out_boxes, out_scores, out_classes = self.module.eval(outputs, image_shape, self.class_names, self.anchors)
        detections = []

        box_array = np.asarray(out_boxes)
        score_array = np.asarray(out_scores)
        class_array = np.asarray(out_classes)
        if box_array.size != 0:
            if box_array.ndim == 1:
                box_array = box_array.reshape(1, -1)
            if score_array.ndim == 0:
                score_array = score_array.reshape(1)
            if class_array.ndim == 0:
                class_array = class_array.reshape(1)

            count = min(len(box_array), len(score_array), len(class_array))
            for index in range(count):
                class_index = int(class_array[index])
                if class_index < 0 or class_index >= len(self.class_names):
                    continue
                detections.append(
                    {
                        "class": self.class_names[class_index],
                        "score": float(score_array[index]),
                        "bbox": self._format_bbox(box_array[index]),
                    }
                )

        post_ms = ns_to_ms(_perf_counter_ns() - post_start_ns)
        return detections, {
            "pre_ms": pre_ms,
            "dpu_ms": dpu_ms,
            "post_ms": post_ms,
            "total_ms": pre_ms + dpu_ms + post_ms,
        }

    def close(self):
        n2cube.dpuDestroyTask(self.task)
        n2cube.dpuDestroyKernel(self.kernel)


def parse_args():
    parser = argparse.ArgumentParser(description="Zynq mix RGB inference server")
    parser.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--send-host", default=DEFAULT_SEND_HOST)
    parser.add_argument("--send-port", type=int, default=DEFAULT_SEND_PORT)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES, help="Comma-separated profiles: mix,mix_tiny")
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=None)
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE)
    parser.add_argument("--debug-latency", action="store_true", help="Enable latency reporting and debug fields")
    parser.add_argument(
        "--latency-mode",
        choices=("one-way", "rtt", "both"),
        default="both",
        help="Latency view to emit when debug is enabled",
    )
    parser.add_argument("--debug-print-interval", type=float, default=DEFAULT_DEBUG_PRINT_INTERVAL)
    parser.add_argument("--latency-offset-ms", type=float, default=DEFAULT_LATENCY_OFFSET_MS)
    parser.add_argument("--latency-min-sample-ms", type=float, default=DEFAULT_LATENCY_MIN_SAMPLE_MS)
    return parser.parse_args()


def build_detectors(profiles, args):
    detectors = []
    if n2cube is None:
        print("[WARN] dnndk not found; running in passthrough mode")
        return detectors

    n2cube.dpuOpen()
    for profile_name in profiles:
        config = PROFILE_CONFIGS[profile_name]
        module = load_module(config["module_name"])
        detectors.append(
            DetectorRunner(
                profile_name=profile_name,
                module=module,
                config=config,
                score_thresh=args.score_thresh,
                nms_thresh=args.nms_thresh,
            )
        )
    return detectors


def parse_packet_header(packet):
    if len(packet) < HEADER_SIZE:
        return None

    magic, stamp_ns, send_ts_ns, chunk_index, total_chunks = struct.unpack(HEADER_FORMAT, packet[:HEADER_SIZE])
    if magic != HEADER_MAGIC:
        return None
    if not (0 <= chunk_index < total_chunks <= MAX_REASONABLE_CHUNKS):
        return None

    return {
        "stamp_ns": stamp_ns,
        "send_ts_ns": send_ts_ns,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "payload_offset": HEADER_SIZE,
    }


def format_detector_summary(detector_stats):
    if not detector_stats:
        return "none"

    parts = []
    for stat in detector_stats:
        parts.append(
            "{profile}:{count} pre={pre_ms:.1f} dpu={dpu_ms:.1f} post={post_ms:.1f}".format(
                profile=stat.get("profile", "unknown"),
                count=stat.get("count", 0),
                pre_ms=stat.get("pre_ms", 0.0),
                dpu_ms=stat.get("dpu_ms", 0.0),
                post_ms=stat.get("post_ms", 0.0),
            )
        )
    return " | ".join(parts)


def maybe_print_debug_dashboard(args, profiles_label, frame_meta, response, detector_stats, latency_tracker, last_print_ns):
    if not args.debug_latency:
        return last_print_ns

    now_ns = _time_ns()
    interval_ns = int(max(args.debug_print_interval, 0.05) * 1.0e9)
    if last_print_ns and now_ns - last_print_ns < interval_ns:
        return last_print_ns

    lines = ["=== Zynq Mix Debug ==="]
    lines.append("profiles : {}".format(profiles_label))
    lines.append("count    : {}".format(response.get("count", 0)))
    lines.append("frame age: {:.2f} ms".format(response.get("frame_age_ms", 0.0)))
    if args.latency_mode in ("one-way", "both") and response.get("one_way_latency_ms") is not None:
        lines.append(
            "one-way  : raw {:.2f} ms | fix {:.2f} ms | cal {:.2f} ms".format(
                response.get("one_way_latency_ms", 0.0),
                response.get("latency_fix_ms", 0.0),
                response.get("calibrated_latency_ms", 0.0),
            )
        )
    if args.latency_mode in ("rtt", "both"):
        lines.append("send ts  : {}".format(response.get("send_ts_ns", 0)))
    lines.append("stages   : {}".format(format_detector_summary(detector_stats)))
    if latency_tracker.min_latency_ms is not None:
        lines.append("sync fix : {:.4f} ms".format(latency_tracker.min_latency_ms))

    print("\n".join(lines))
    return now_ns


def main():
    args = parse_args()
    profiles = parse_profiles(args.profiles)
    profiles_label = ",".join(profiles)
    detectors = build_detectors(profiles, args)
    assembler = FrameAssembler(max_age_sec=args.max_age)
    latency_tracker = LatencyTracker(
        min_sample_ms=args.latency_min_sample_ms,
        offset_ms=args.latency_offset_ms,
    )

    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_socket.bind(("", args.listen_port))

    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_debug_print_ns = 0

    try:
        while True:
            packet, _address = listen_socket.recvfrom(65535)
            header = parse_packet_header(packet)
            if header is None:
                continue

            payload = packet[header["payload_offset"] :]
            frame_meta = assembler.add_packet(
                {
                    "stamp_ns": header["stamp_ns"],
                    "send_ts_ns": header["send_ts_ns"],
                    "chunk_index": header["chunk_index"],
                    "total_chunks": header["total_chunks"],
                    "recv_ns": _time_ns(),
                },
                payload,
            )
            if frame_meta is None:
                continue

            decode_start_ns = _perf_counter_ns()
            image_array = np.frombuffer(frame_meta["jpeg_data"], dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            decode_ms = ns_to_ms(_perf_counter_ns() - decode_start_ns)
            if image is None:
                continue

            detections = []
            detector_stats = []
            for detector in detectors:
                detector_detections, timings = detector.infer(image)
                detections.extend(detector_detections)
                detector_stats.append(
                    {
                        "profile": detector.profile_name,
                        "count": len(detector_detections),
                        "pre_ms": round(timings["pre_ms"], 3),
                        "dpu_ms": round(timings["dpu_ms"], 3),
                        "post_ms": round(timings["post_ms"], 3),
                        "total_ms": round(timings["total_ms"], 3),
                    }
                )

            total_pre_ms = sum(stat["pre_ms"] for stat in detector_stats)
            total_dpu_ms = sum(stat["dpu_ms"] for stat in detector_stats)
            total_post_ms = sum(stat["post_ms"] for stat in detector_stats)
            response = {
                "stamp_ns": frame_meta["stamp_ns"],
                "send_ts_ns": frame_meta["send_ts_ns"],
                "send_ts": frame_meta["send_ts_ns"] / 1.0e9 if frame_meta["send_ts_ns"] else 0.0,
                "count": len(detections),
                "detections": detections,
                "pre_ms": round(total_pre_ms, 3),
                "dpu_ms": round(total_dpu_ms, 3),
                "post_ms": round(total_post_ms, 3),
                "total_ms": round(total_pre_ms + total_dpu_ms + total_post_ms, 3),
            }

            if args.debug_latency:
                server_send_ns = _time_ns()
                response["server_recv_ns"] = frame_meta["server_recv_ns"]
                response["server_send_ns"] = server_send_ns
                response["frame_age_ms"] = round(ns_to_ms(frame_meta["frame_age_ns"]), 3)
                response["decode_ms"] = round(decode_ms, 3)
                response["detector_stats"] = detector_stats
                response["profiles"] = profiles
                response["latency_mode"] = args.latency_mode
                if frame_meta["send_ts_ns"]:
                    raw_latency_ms = ns_to_ms(frame_meta["server_recv_ns"] - frame_meta["send_ts_ns"])
                    latency_tracker.observe(raw_latency_ms)
                    response["one_way_latency_ms"] = round(raw_latency_ms, 3)
                    response["calibrated_latency_ms"] = round(latency_tracker.calibrated(raw_latency_ms), 3)
                    response["latency_fix_ms"] = round(latency_tracker.min_latency_ms or 0.0, 3)

            message = json.dumps(response, separators=(",", ":"))
            send_socket.sendto(message.encode("utf-8"), (args.send_host, args.send_port))

            last_debug_print_ns = maybe_print_debug_dashboard(
                args,
                profiles_label,
                frame_meta,
                response,
                detector_stats,
                latency_tracker,
                last_debug_print_ns,
            )
    finally:
        for detector in detectors:
            detector.close()
        if n2cube is not None:
            n2cube.dpuClose()


if __name__ == "__main__":
    main()
