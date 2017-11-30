#! /usr/bin/env python

import rospy
import message_filters # ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import tf
import math
import numpy as np
import cv2

from mapper import Occ_Map


class ScanToRangeBearing(object):

    def __init__(self):

        # initialize class variables
        self.min_range = 0.0
        self.max_range = 0.0
        self.min_angle = 0.0
        self.max_angle = 0.0
        self.angle_incr = 0.0

        self.count = 0

        self.laser_offset = 1.03*np.pi

        self.pose = np.zeros(3)
        self.pose_valid = False

        # initialize bearings array
        self.bearings = np.zeros(360)

        self.first = True

        # initialize LaserScan subscriber
        # self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # self.pose_sub = rospy.Subscriber("/nwu_turtlebot_pose", PoseStamped, self.pose_callback)

        self.pose_sub = message_filters.Subscriber("/nwu_turtlebot_pose", PoseStamped)
        self.scan_sub = message_filters.Subscriber("/scan", LaserScan)
        ats = message_filters.ApproximateTimeSynchronizer([self.pose_sub, self.scan_sub], queue_size=15, slop=0.01)
        ats.registerCallback(self.update)

        # self.body_offset = (0.08, -0.075)
        self.body_offset = (0.12, -0.07)
        self.map_offset = (5., 3.)
        self.resolution = 0.1
        self.width = 10.
        self.height = 6.
        self.map_params = {'width':self.width,
                           'height':self.height, 
                           'offset':self.map_offset, 
                           'body_offset':self.body_offset,
                           'resolution':self.resolution, 
                           'z_max':150,
                           'alpha':2*self.resolution,
                           'beta':1.5*np.pi/180, 
                           'p_free':0.3,
                           'p_occ':0.75}
        self.mapper = Occ_Map(**self.map_params)
        # self.mapper = Occ_Map(width=self.width, height=self.height, offset=self.map_offset, 
        #                       body_offset=self.body_offset, resolution=self.resolution, 
        #                       z_max = 150, alpha=2*self.resolution, beta=1.5*np.pi/180, 
        #                       p_free = 0.3, p_occ = 0.75)

        # cv2.namedWindow('map',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('map', 600,600)
        # initialize publisher

    def update(self, pose_msg, scan_msg):
        self.pose_callback(pose_msg)
        self.scan_callback(scan_msg)
        # print("Callback")


    def pose_callback(self, msg):
        self.pose[0] = msg.pose.position.x
        self.pose[1] = msg.pose.position.y

        q = np.zeros(4)
        q[0] = msg.pose.orientation.x
        q[1] = msg.pose.orientation.y
        q[2] = msg.pose.orientation.z
        q[3] = msg.pose.orientation.w


        euler = tf.transformations.euler_from_quaternion(q)

        self.pose[2] = euler[2]
        self.pose_valid = True
        # print(self.pose)

    def scan_callback(self, msg):

        self.count = self.count + 1

        # if self.count == 10:   # 

            # self.count = 0

        # first time around, get important LaserScan info
        if self.first:
            self.min_range = msg.range_min
            self.max_range = msg.range_max
            self.min_angle = msg.angle_min
            self.max_angle = msg.angle_max
            self.angle_incr = msg.angle_increment
            self.num_measurements = len(msg.ranges)

            # create bearings array
            self.bearings = np.linspace(self.min_angle, self.max_angle, num=self.num_measurements) + self.laser_offset
            # self.bearings = np.linspace(0, 2*math.pi, num=num_measurements)
            # self.bearings = np.linspace(math.pi, -math.pi, num=num_measurements)
            # done
            self.first = False

        # create numpy array of the range measurements
        raw_ranges = np.array(msg.ranges)
        # print(self.num_measurements)
        # get a mask of locations where we have valid range measurements (mostly ignoring the 'inf' measurements here)
        mask = np.where(np.logical_and(np.greater_equal(raw_ranges,self.min_range), np.less_equal(raw_ranges,self.max_range)))

        # store valid range and bearing measurements in arrays
        range_meas = raw_ranges[mask]
        bearing_meas = self.bearings[mask]

        # print(max(range_meas))
        # skip = 10
        scans = 15
        idx = np.random.randint(0, len(range_meas), scans)
        # self.mapper.update(self.pose, range_meas[::skip], bearing_meas[::skip])
        self.mapper.update(self.pose, range_meas[idx], bearing_meas[idx])

        # m = self.mapper.get_map()
        m = np.array(1. - self.mapper.get_map(), dtype=np.float32)
        m = cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
        x = np.array(self.pose[:2], dtype=np.int32)
        x[0] = int((self.pose[0] + self.map_offset[0])/self.resolution)
        x[1] = int((self.pose[1] + self.map_offset[1])/self.resolution)
        arrow = np.copy(x)
        arrow[0] += int(5*np.cos(self.pose[2]))
        arrow[1] += int(5*np.sin(self.pose[2]))
        cv2.line(m, (x[1], x[0]), (arrow[1], arrow[0]), (0., 0., 1.))
        m = cv2.resize(m, (0,0), fx=10, fy=10)
        cv2.imshow("map", m)
        cv2.waitKey(1)
        # publish range and bearings?


def main():
    # initialize a node
    rospy.init_node('range_bearing_pub')
    print("Initializing mapping node.")
    # create instance of ScanToRangeBearing class
    translator = ScanToRangeBearing()

    # spin
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
