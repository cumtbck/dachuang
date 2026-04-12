#!/usr/bin/env python
import math
import socket
import struct

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


HEADER_FORMAT = "!2sQHH"
HEADER_MAGIC = b"MC"


class UdpImageSender(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.target_ip = rospy.get_param("~target_ip", "192.168.1.10")
        self.target_port = int(rospy.get_param("~target_port", 5000))
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 80))
        self.max_payload = int(rospy.get_param("~max_payload", 60000))
        self.send_rate = float(rospy.get_param("~fps", 10.0))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.min_period = rospy.Duration(1.0 / max(self.send_rate, 0.1))
        self.last_send_time = rospy.Time(0)

        self.subscriber = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

    def image_callback(self, image_msg):
        now_time = rospy.Time.now()
        if now_time - self.last_send_time < self.min_period:
            return

        self.last_send_time = now_time
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn("rgb_sender: cv_bridge failed: %s", exc)
            return

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, encoded = cv2.imencode(".jpg", cv_image, encode_params)
        if not success:
            rospy.logwarn("rgb_sender: JPEG encoding failed")
            return

        payload = encoded.tobytes()
        total_chunks = int(math.ceil(len(payload) / float(self.max_payload)))
        stamp_ns = image_msg.header.stamp.to_nsec()

        for chunk_index in range(total_chunks):
            start_index = chunk_index * self.max_payload
            end_index = start_index + self.max_payload
            chunk = payload[start_index:end_index]
            header = struct.pack(HEADER_FORMAT, HEADER_MAGIC, stamp_ns, chunk_index, total_chunks)
            packet = header + chunk
            try:
                self.socket.sendto(packet, (self.target_ip, self.target_port))
            except socket.error as exc:
                rospy.logwarn("rgb_sender: UDP send failed: %s", exc)
                break


def main():
    rospy.init_node("rgb_sender")
    UdpImageSender()
    rospy.spin()


if __name__ == "__main__":
    main()
