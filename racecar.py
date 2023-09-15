#!/usr/bin/python3

import rospy
import yaml
import numpy as np 

from argparse import Namespace
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from planner.fgm_ref_way import FGM_REF_WAY

class RosProc:
    def __init__(self):

        self.planner = FGM_REF_WAY()

        self.pub = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom/pf', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.ackermann = AckermannDriveStamped()
        self.ackermann.drive.speed = 0.0
        self.ackermann.drive.steering_angle = 0.0

        self.scan_data = {'ranges': [0] * 1080}
        self.odom_data = {
            'pose_x': 0.0,
            'pose_y': 0.0,
            'pose_theta': 0.0,
            'linear_vels_x': 0.0,
            'lookahead_distance': 0.82461887897713965,
            'vgain': 0.90338203837889
        }

    def odom_callback(self, odom_msg):
        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)

        current_position_theta = np.arctan2(siny_cosp, cosy_cosp)
        current_position_x = odom_msg.pose.pose.position.x
        current_position_y = odom_msg.pose.pose.position.y

        self.odom_data['pose_x'] = current_position_x
        self.odom_data['pose_y'] = current_position_y
        self.odom_data['pose_theta'] = current_position_theta
        self.odom_data['linear_vels_x'] = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        self.scan_data['ranges'] = scan_msg.ranges

        observation = {
            'poses_x': self.odom_data['pose_x'],
            'poses_y': self.odom_data['pose_y'],
            'poses_theta': self.odom_data['pose_theta'],
            'linear_vels_x': self.odom_data['linear_vels_x'],
            'scans': self.scan_data['ranges']
        }
        speed, steering_angle = self.planner.driving(observation)

        self.ackermann.drive.speed = speed
        self.ackermann.drive.steering_angle = steering_angle
        self.pub.publish(self.ackermann)


if __name__ == "__main__":
    rospy.init_node('racecar')
    rate = rospy.Rate(100)
    app = RosProc()
    while not rospy.is_shutdown():
        rate.sleep()
