#! /usr/bin/env python

import rospy
import message_filters # ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf
import math
import numpy as np
import cv2
from slam import FastSLAM

from mapper import Occ_Map


def f(x, u, dt):
    v = u[0]
    w = u[1]
    if np.abs(w) < 10*np.finfo(np.float32).eps:
        w = 10*np.finfo(np.float32).eps

    theta = x[2]
    dx = np.array([-v/w*np.sin(theta) + v/w*np.sin(theta + w*dt),
                         v/w*np.cos(theta) - v/w*np.cos(theta + w*dt),
                         w*dt])
    x_next = x + dx
    #print(x_next)
    return x_next

def del_f_u(x, u, dt):
    v = u.flatten()[0]
    w = u.flatten()[1]
    theta = x.flatten()[2]

    if np.abs(w) < 10*np.finfo(np.float32).eps:
        w = 10*np.finfo(np.float32).eps

    B = np.array([[1/w*(-np.sin(theta) + np.sin(theta + w*dt)), 
                       v/w*(1/w*(np.sin(theta) - np.sin(theta + w*dt)) + np.cos(theta + w*dt)*dt)],
                  [1/w*(np.cos(theta) - np.cos(theta + w*dt)), 
                       v/w*(-1/w*(np.cos(theta) - np.cos(theta + w*dt)) + np.sin(theta + w*dt)*dt)],
                  [0, dt]])
    return B

def Qu(u):
    alpha = np.array([0.1, 0.01, 0.01, 0.1])
    v = u[0]
    w = u[1]
    return np.array([[alpha[0]*v**2 + alpha[1]*w**2, 0],
                     [0, alpha[2]*v**2 + alpha[3]*w**2]])


class Slammer(object):

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

        # velocity inputs [v, w]
        self.u = np.zeros(2)

        # initialize bearings array
        self.bearings = np.zeros(360)

        self.first = True

        # initialize LaserScan subscriber
        # self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # self.pose_sub = rospy.Subscriber("/nwu_turtlebot_pose", PoseStamped, self.pose_callback)

        # self.pose_sub = message_filters.Subscriber("/nwu_turtlebot_pose", PoseStamped)
        self.scan_sub = message_filters.Subscriber("/scan", LaserScan)
        self.odom_sub = message_filters.Subscriber("/odom", Odometry)
        ats = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], queue_size=15, slop=0.06)
        ats.registerCallback(self.update)

        # self.body_offset = (0.08, -0.075)
        self.body_offset = (0.12, -0.07)
        self.map_offset = (6., 3.)
        self.resolution = 0.1
        self.width = 12.
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

        self.num_particles = 10
        self.Ts = 0.0796
        x0 = np.zeros(3)

        self.slammer = FastSLAM(x0, self.num_particles, self.map_params, f, del_f_u, Qu, self.Ts)
        # self.mapper = Occ_Map(width=self.width, height=self.height, offset=self.map_offset, 
        #                       body_offset=self.body_offset, resolution=self.resolution, 
        #                       z_max = 150, alpha=2*self.resolution, beta=1.5*np.pi/180, 
        #                       p_free = 0.3, p_occ = 0.75)

        # cv2.namedWindow('map',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('map', 600,600)
        # initialize publisher

    def update(self, odom_msg, scan_msg):
        self.odom_callback(odom_msg)
        self.scan_callback(scan_msg)
        # print("Callback")

    def odom_callback(self, msg):
        self.u[0] = msg.twist.twist.linear.x
        self.u[1] = msg.twist.twist.angular.z



    # def pose_callback(self, msg):
    #     self.pose[0] = msg.pose.position.x
    #     self.pose[1] = msg.pose.position.y

    #     q = np.zeros(4)
    #     q[0] = msg.pose.orientation.x
    #     q[1] = msg.pose.orientation.y
    #     q[2] = msg.pose.orientation.z
    #     q[3] = msg.pose.orientation.w


    #     euler = tf.transformations.euler_from_quaternion(q)

    #     self.pose[2] = euler[2]
    #     self.pose_valid = True
    #     # print(self.pose)

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
        # scans = 15
        # idx = np.random.randint(0, len(range_meas), scans)
        # # self.mapper.update(self.pose, range_meas[::skip], bearing_meas[::skip])
        # self.mapper.update(self.pose, range_meas[idx], bearing_meas[idx])
        z = np.vstack((range_meas[None, :], bearing_meas[None, :]))
        self.slammer.update(self.u, z)

        m = self.slammer.get_map()
        m = np.array(1. - m, dtype=np.float32)
        m = cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
        x = np.array(self.pose[:2], dtype=np.int32)
        x[0] = int((self.pose[0] + self.map_offset[0])/self.resolution)
        x[1] = int((self.pose[1] + self.map_offset[1])/self.resolution)
        arrow = np.copy(x)
        arrow[0] += int(5*np.cos(self.pose[2]))
        arrow[1] += int(5*np.sin(self.pose[2]))
        cv2.line(m, (x[1], x[0]), (arrow[1], arrow[0]), (0., 0., 1.))
        m = cv2.resize(m, (0,0), fx=4, fy=4)
        cv2.imshow("map", m)
        cv2.waitKey(1)
        # publish range and bearings?


def main():
    # initialize a node
    rospy.init_node('slammer')
    print("Initializing slamming node.")
    # create instance of ScanToRangeBearing class
    slammer = Slammer()

    # spin
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
