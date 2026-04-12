#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class TrafficManager(object):
    def __init__(self):
        self.cmd_vel_in = rospy.get_param("~cmd_vel_in", "/cmd_vel_nav")
        self.cmd_vel_out = rospy.get_param("~cmd_vel_out", "/cmd_vel")
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))

        self.last_cmd = Twist()
        self.traffic_stop = False
        self.safety_stop = False

        rospy.Subscriber(self.cmd_vel_in, Twist, self.cmd_vel_callback, queue_size=10)
        rospy.Subscriber("/traffic_light_status", Bool, self.traffic_callback, queue_size=10)
        rospy.Subscriber("/safety_stop", Bool, self.safety_callback, queue_size=10)

        self.publisher = rospy.Publisher(self.cmd_vel_out, Twist, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_timer)

    def cmd_vel_callback(self, cmd_msg):
        self.last_cmd = cmd_msg

    def traffic_callback(self, status_msg):
        self.traffic_stop = bool(status_msg.data)

    def safety_callback(self, status_msg):
        self.safety_stop = bool(status_msg.data)

    def publish_timer(self, _event):
        if self.traffic_stop or self.safety_stop:
            self.publisher.publish(Twist())
        else:
            self.publisher.publish(self.last_cmd)


def main():
    rospy.init_node("traffic_manager")
    TrafficManager()
    rospy.spin()


if __name__ == "__main__":
    main()
