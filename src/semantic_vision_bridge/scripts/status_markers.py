#!/usr/bin/env python
import rospy
import tf2_ros
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray


class StatusMarkers(object):
    def __init__(self):
        self.marker_topic = rospy.get_param("~marker_topic", "/semantic_status_markers")
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.text_z = float(rospy.get_param("~text_z", 1.2))
        self.text_scale = float(rospy.get_param("~text_scale", 0.35))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.use_tf = bool(rospy.get_param("~use_tf", False))
        self.map_frame = rospy.get_param("~map_frame", "map")

        self.traffic_stop = None
        self.safety_stop = None

        rospy.Subscriber("/traffic_light_status", Bool, self._traffic_cb, queue_size=5)
        rospy.Subscriber("/safety_stop", Bool, self._safety_cb, queue_size=5)

        self.pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        period = 1.0 / max(self.publish_rate, 0.1)
        self.timer = rospy.Timer(rospy.Duration(period), self._publish_timer)

    def _traffic_cb(self, msg):
        self.traffic_stop = bool(msg.data)

    def _safety_cb(self, msg):
        self.safety_stop = bool(msg.data)

    def _resolve_frame(self):
        if not self.use_tf:
            return self.frame_id, (0.0, 0.0, self.text_z)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.frame_id,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
            t = transform.transform.translation
            return self.map_frame, (t.x, t.y, t.z + self.text_z)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            return self.frame_id, (0.0, 0.0, self.text_z)

    def _make_text_marker(self, marker_id, text, rgba, frame_id, position):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        marker.ns = "semantic_status"
        marker.id = int(marker_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.z = self.text_scale
        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])
        marker.text = text
        marker.lifetime = rospy.Duration(0.3)
        return marker

    def _publish_timer(self, _event):
        frame_id, position = self._resolve_frame()

        lines = []
        # traffic_stop: True means "red light stop", False means "green allowed"
        if self.traffic_stop is True:
            lines.append(("TL: RED (stop)", (1.0, 0.1, 0.1, 1.0)))
        elif self.traffic_stop is False:
            lines.append(("TL: GREEN", (0.2, 1.0, 0.2, 1.0)))
        else:
            lines.append(("TL: n/a", (0.8, 0.8, 0.8, 0.9)))

        if self.safety_stop is True:
            lines.append(("SAFETY: STOP", (1.0, 0.6, 0.0, 1.0)))
        elif self.safety_stop is False:
            lines.append(("SAFETY: OK", (0.8, 0.8, 0.8, 0.9)))
        else:
            lines.append(("SAFETY: n/a", (0.8, 0.8, 0.8, 0.9)))

        markers = MarkerArray()
        for index, (text, rgba) in enumerate(lines):
            markers.markers.append(self._make_text_marker(index, text, rgba, frame_id, position))

        self.pub.publish(markers)


def main():
    rospy.init_node("semantic_status_markers")
    StatusMarkers()
    rospy.spin()


if __name__ == "__main__":
    main()
