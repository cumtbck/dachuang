#!/usr/bin/env python
import json
import socket
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool, Header


def _time_ns():
    try:
        return time.time_ns()
    except AttributeError:
        return int(time.time() * 1.0e9)


def _time_from_ns(stamp_ns, use_now_if_zero=False):
    stamp_ns = int(stamp_ns or 0)
    if stamp_ns <= 0:
        return rospy.Time.now() if use_now_if_zero else rospy.Time(0, 0)
    secs = stamp_ns // 1000000000
    nsecs = stamp_ns % 1000000000
    return rospy.Time(secs, nsecs)


class DepthBuffer(object):
    def __init__(self, max_length):
        self._buffer = deque(maxlen=max_length)
        self._lock = threading.Lock()

    def push(self, stamp_ns, depth_image, frame_id):
        with self._lock:
            self._buffer.append((stamp_ns, depth_image, frame_id))

    def closest(self, stamp_ns):
        with self._lock:
            if not self._buffer:
                return None
            return min(self._buffer, key=lambda item: abs(item[0] - stamp_ns))


class SemanticSolver(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.listen_port = int(rospy.get_param("~listen_port", 5001))
        self.target_frame = rospy.get_param("~target_frame", "base_link")
        self.buffer_size = int(rospy.get_param("~buffer_size", 10))
        self.depth_scale = float(rospy.get_param("~depth_scale", 1000.0))
        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 0.2))
        self.traffic_stop_distance = float(rospy.get_param("~traffic_stop_distance", 2.0))
        self.helmet_stop_distance = float(rospy.get_param("~helmet_stop_distance", 1.0))
        self.helmet_classes = set(rospy.get_param("~helmet_classes", ["helmet"]))
        self.red_light_classes = set(rospy.get_param("~red_light_classes", ["red_light"]))
        self.green_light_classes = set(rospy.get_param("~green_light_classes", ["green_light"]))
        self.enable_result_view = bool(rospy.get_param("~enable_result_view", False))
        self.result_view_topic = rospy.get_param("~result_view_topic", "/camera/color/image_raw")
        self.result_view_window = rospy.get_param("~result_view_window", "mix_detections")
        self.result_view_fps = max(float(rospy.get_param("~result_view_fps", 10.0)), 0.1)
        self.result_view_buffer_size = int(rospy.get_param("~result_view_buffer_size", self.buffer_size))
        self.result_view_match_tolerance_ms = max(
            float(rospy.get_param("~result_view_match_tolerance_ms", 80.0)),
            0.0,
        )
        self.debug_latency = bool(rospy.get_param("~debug_latency", False))
        self.latency_mode = str(rospy.get_param("~latency_mode", "both")).lower()
        self.latency_log_interval = max(float(rospy.get_param("~latency_log_interval", 1.0)), 0.0)

        self.depth_buffer = DepthBuffer(self.buffer_size)
        self.rgb_buffer = DepthBuffer(self.result_view_buffer_size)
        self.camera_matrix = None
        self.camera_frame = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.semantic_pub = rospy.Publisher("/semantic_obstacles", PointCloud2, queue_size=1)
        self.traffic_pub = rospy.Publisher("/traffic_light_status", Bool, queue_size=1)
        self.safety_pub = rospy.Publisher("/safety_stop", Bool, queue_size=1)

        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=5)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        if self.enable_result_view:
            rospy.Subscriber(self.result_view_topic, Image, self.rgb_callback, queue_size=1, buff_size=2 ** 24)

        self._traffic_state = False
        self._safety_state = False
        self._last_latency_log_time = 0.0
        self._last_result_view_time = 0.0
        self._result_window_ready = False

        if self.enable_result_view:
            rospy.on_shutdown(self._shutdown_result_view)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", self.listen_port))
        self.socket.settimeout(0.2)

        rospy.loginfo(
            "semantic_solver: depth_topic=%s camera_info_topic=%s result_view_topic=%s"
            % (self.depth_topic, self.camera_info_topic, self.result_view_topic)
        )

        self.receiver_thread = threading.Thread(target=self._receive_loop)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def camera_info_callback(self, info_msg):
        self.camera_matrix = info_msg.K
        self.camera_frame = info_msg.header.frame_id

    def depth_callback(self, depth_msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            rospy.logwarn("semantic_solver: depth conversion failed: %s", exc)
            return
        self.depth_buffer.push(depth_msg.header.stamp.to_nsec(), depth_image, depth_msg.header.frame_id)

    def rgb_callback(self, rgb_msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn("semantic_solver: rgb conversion failed: %s", exc)
            return
        self.rgb_buffer.push(rgb_msg.header.stamp.to_nsec(), rgb_image, rgb_msg.header.frame_id)

    def _receive_loop(self):
        while not rospy.is_shutdown():
            try:
                payload, _address = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except socket.error as exc:
                rospy.logwarn("semantic_solver: UDP recv failed: %s", exc)
                continue

            try:
                message = json.loads(payload.decode("utf-8"))
            except ValueError:
                rospy.logwarn("semantic_solver: invalid JSON payload")
                continue

            stamp_ns = int(message.get("stamp_ns", 0))
            self._maybe_log_latency_debug(message, stamp_ns)
            detections = message.get("detections", [])
            if not detections:
                self._publish_empty_cloud(stamp_ns)
                if self._safety_state:
                    self._safety_state = False
                    self.safety_pub.publish(Bool(data=False))
                self._maybe_show_result_image(stamp_ns, detections)
                continue

            self._handle_detections(stamp_ns, detections)
            self._maybe_show_result_image(stamp_ns, detections)

    def _maybe_log_latency_debug(self, message, stamp_ns):
        if not self.debug_latency:
            return

        send_ts_ns = int(message.get("send_ts_ns", 0))
        if send_ts_ns <= 0:
            return

        now_sec = time.time()
        if now_sec - self._last_latency_log_time < self.latency_log_interval:
            return
        self._last_latency_log_time = now_sec

        parts = []
        recv_ns = _time_ns()

        if self.latency_mode in ("rtt", "both"):
            rtt_ms = (recv_ns - send_ts_ns) / 1.0e6
            parts.append("RTT %.2f ms" % rtt_ms)

        if self.latency_mode in ("one-way", "both"):
            server_recv_ns = int(message.get("server_recv_ns", 0))
            server_send_ns = int(message.get("server_send_ns", 0))
            if server_recv_ns > 0:
                one_way_ms = (server_recv_ns - send_ts_ns) / 1.0e6
                parts.append("one-way %.2f ms" % one_way_ms)
            if server_recv_ns > 0 and server_send_ns > 0:
                server_ms = (server_send_ns - server_recv_ns) / 1.0e6
                parts.append("zynq %.2f ms" % server_ms)

        frame_age_ms = message.get("frame_age_ms")
        if frame_age_ms is not None:
            parts.append("frame-age %.2f ms" % float(frame_age_ms))

        detector_stats = message.get("detector_stats", [])
        if detector_stats:
            summary = []
            for detector_stat in detector_stats:
                summary.append(
                    "%s:%s pre=%.1f dpu=%.1f post=%.1f"
                    % (
                        detector_stat.get("profile", "unknown"),
                        detector_stat.get("count", 0),
                        float(detector_stat.get("pre_ms", 0.0)),
                        float(detector_stat.get("dpu_ms", 0.0)),
                        float(detector_stat.get("post_ms", 0.0)),
                    )
                )
            parts.append("stages " + " | ".join(summary))

        if parts:
            rospy.loginfo("semantic_solver latency (stamp_ns=%s): %s" % (stamp_ns, "; ".join(parts)))

    def _maybe_show_result_image(self, stamp_ns, detections):
        if not self.enable_result_view:
            return

        now_sec = time.time()
        min_interval_sec = 1.0 / self.result_view_fps
        if self._last_result_view_time and now_sec - self._last_result_view_time < min_interval_sec:
            return

        rgb_entry = self.rgb_buffer.closest(stamp_ns)
        if rgb_entry is None:
            rospy.logwarn_throttle(5.0, "semantic_solver: no RGB frame available for result view")
            return

        rgb_stamp_ns, rgb_image, _frame_id = rgb_entry
        if self.result_view_match_tolerance_ms > 0.0:
            mismatch_ms = abs(rgb_stamp_ns - stamp_ns) / 1.0e6
            if mismatch_ms > self.result_view_match_tolerance_ms:
                rospy.logwarn_throttle(
                    5.0,
                    "semantic_solver: RGB frame mismatch %.2f ms exceeds tolerance" % mismatch_ms,
                )
                return

        overlay = self._draw_detections(rgb_image, detections)
        try:
            if not self._result_window_ready:
                cv2.namedWindow(self.result_view_window, cv2.WINDOW_NORMAL)
                self._result_window_ready = True
            cv2.imshow(self.result_view_window, overlay)
            cv2.waitKey(1)
            self._last_result_view_time = now_sec
        except cv2.error as exc:
            rospy.logwarn("semantic_solver: result view failed: %s", exc)
            self.enable_result_view = False

    def _draw_detections(self, image, detections):
        overlay = image.copy()
        image_height, image_width = overlay.shape[:2]

        for detection in detections:
            bbox = detection.get("bbox")
            class_name = detection.get("class", "object")
            if not bbox or len(bbox) < 4:
                continue

            x_min, y_min, x_max, y_max = [int(value) for value in bbox[:4]]
            x_min = max(0, min(x_min, image_width - 1))
            x_max = max(0, min(x_max, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            y_max = max(0, min(y_max, image_height - 1))
            if x_max <= x_min or y_max <= y_min:
                continue

            color = self._color_for_label(class_name)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)

            label = class_name
            score = detection.get("score")
            if score is not None:
                label = "%s:%.2f" % (class_name, float(score))

            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_width = text_size[0] + 6
            text_height = text_size[1] + baseline + 4
            text_top = max(0, y_min - text_height)
            cv2.rectangle(overlay, (x_min, text_top), (x_min + text_width, text_top + text_height), color, -1)
            cv2.putText(
                overlay,
                label,
                (x_min + 3, text_top + text_size[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return overlay

    @staticmethod
    def _color_for_label(label):
        palette = [
            (71, 99, 255),
            (255, 191, 0),
            (50, 205, 50),
            (0, 215, 255),
            (180, 105, 255),
            (230, 216, 173),
        ]
        index = sum(ord(char) for char in str(label)) % len(palette)
        return palette[index]

    def _shutdown_result_view(self):
        if not self._result_window_ready:
            return
        try:
            cv2.destroyWindow(self.result_view_window)
        except cv2.error:
            pass

    def _handle_detections(self, stamp_ns, detections):
        depth_entry = self.depth_buffer.closest(stamp_ns)
        if depth_entry is None or self.camera_matrix is None:
            rospy.logwarn_throttle(5.0, "semantic_solver: missing depth or camera info")
            return

        _depth_stamp, depth_image, depth_frame = depth_entry
        fx, fy, cx, cy = self.camera_matrix[0], self.camera_matrix[4], self.camera_matrix[2], self.camera_matrix[5]
        image_height, image_width = depth_image.shape[:2]

        semantic_points = []
        traffic_detected = None
        safety_detected = False

        for detection in detections:
            class_name = detection.get("class")
            bbox = detection.get("bbox")
            if not bbox or class_name is None:
                continue

            x_min, y_min, x_max, y_max = [int(value) for value in bbox]
            x_min = max(0, min(x_min, image_width - 1))
            x_max = max(0, min(x_max, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            y_max = max(0, min(y_max, image_height - 1))
            if x_max <= x_min or y_max <= y_min:
                continue

            depth_value = self._depth_from_bbox(depth_image, x_min, y_min, x_max, y_max)
            point_base = None
            distance_xy = None
            if depth_value is not None:
                center_u = (x_min + x_max) / 2.0
                center_v = (y_min + y_max) / 2.0
                point_camera = self._project_to_camera(center_u, center_v, depth_value, fx, fy, cx, cy)
                if point_camera is not None:
                    point_base = self._transform_point(point_camera, depth_frame, stamp_ns)
                    if point_base is not None:
                        distance_xy = (point_base[0] ** 2 + point_base[1] ** 2) ** 0.5

            if class_name in self.helmet_classes and point_base is not None:
                semantic_points.extend(self._expand_obstacle(point_base))
                if distance_xy is not None and distance_xy <= self.helmet_stop_distance:
                    safety_detected = True

            if class_name in self.red_light_classes:
                if distance_xy is None or distance_xy <= self.traffic_stop_distance:
                    traffic_detected = True

            if class_name in self.green_light_classes and traffic_detected is None:
                traffic_detected = False

        if semantic_points:
            cloud = self._create_cloud(semantic_points, stamp_ns)
            self.semantic_pub.publish(cloud)
        else:
            self._publish_empty_cloud(stamp_ns)

        if traffic_detected is not None:
            self._traffic_state = traffic_detected
            self.traffic_pub.publish(Bool(data=self._traffic_state))

        if safety_detected != self._safety_state:
            self._safety_state = safety_detected
            self.safety_pub.publish(Bool(data=self._safety_state))

    def _depth_from_bbox(self, depth_image, x_min, y_min, x_max, y_max):
        box_width = x_max - x_min
        box_height = y_max - y_min
        offset_x = int(box_width * 0.35)
        offset_y = int(box_height * 0.35)
        roi_x_min = x_min + offset_x
        roi_y_min = y_min + offset_y
        roi_x_max = x_max - offset_x
        roi_y_max = y_max - offset_y
        if roi_x_max <= roi_x_min or roi_y_max <= roi_y_min:
            roi_x_min, roi_y_min, roi_x_max, roi_y_max = x_min, y_min, x_max, y_max

        region = depth_image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        if region.size == 0:
            return None

        region = region.astype(np.float32)
        if depth_image.dtype == np.uint16:
            region /= self.depth_scale
        region = region[np.isfinite(region)]
        region = region[region > 0.05]
        if region.size == 0:
            return None
        return float(np.median(region))

    @staticmethod
    def _project_to_camera(u_value, v_value, depth_value, fx, fy, cx, cy):
        if depth_value <= 0.0:
            return None
        x_value = (u_value - cx) * depth_value / fx
        y_value = (v_value - cy) * depth_value / fy
        return x_value, y_value, depth_value

    def _transform_point(self, point_camera, source_frame, stamp_ns):
        point_msg = PointStamped()
        point_msg.header.stamp = _time_from_ns(stamp_ns)
        point_msg.header.frame_id = source_frame or self.camera_frame
        point_msg.point.x = point_camera[0]
        point_msg.point.y = point_camera[1]
        point_msg.point.z = point_camera[2]
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                point_msg.header.frame_id,
                point_msg.header.stamp,
                rospy.Duration(0.1),
            )
            point_out = tf2_geometry_msgs.do_transform_point(point_msg, transform)
            return (point_out.point.x, point_out.point.y, point_out.point.z)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            rospy.logwarn_throttle(5.0, "semantic_solver: TF unavailable from %s to %s", point_msg.header.frame_id, self.target_frame)
            return None

    def _expand_obstacle(self, point_base):
        radius = max(self.obstacle_radius, 0.05)
        points = []
        points.append(point_base)
        for angle_index in range(12):
            angle = (2.0 * np.pi / 12.0) * angle_index
            offset_x = radius * np.cos(angle)
            offset_y = radius * np.sin(angle)
            points.append((point_base[0] + offset_x, point_base[1] + offset_y, point_base[2]))
        return points

    def _create_cloud(self, points, stamp_ns):
        header = Header()
        header.stamp = _time_from_ns(stamp_ns)
        header.frame_id = self.target_frame
        return point_cloud2.create_cloud_xyz32(header, points)

    def _publish_empty_cloud(self, stamp_ns):
        header = Header()
        header.stamp = _time_from_ns(stamp_ns, use_now_if_zero=True)
        header.frame_id = self.target_frame
        empty_cloud = point_cloud2.create_cloud_xyz32(header, [])
        self.semantic_pub.publish(empty_cloud)


def main():
    rospy.init_node("semantic_solver")
    SemanticSolver()
    rospy.spin()


if __name__ == "__main__":
    main()
