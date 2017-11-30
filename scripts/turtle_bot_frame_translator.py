#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
import tf
import math
import numpy as np


class BotFrameTranslator(object):

    def __init__(self):

        self.pose_nwu = PoseStamped()

        # define rotation from Motion capture frame (m) to NWU frame (nwu)
        self.rot_m_t = np.array([[0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]])

        # initialize pose subscriber
        self.pose_sub = rospy.Subscriber("/vrpn_client_node/Turtlebot/pose", PoseStamped, self.pose_callback)

        # initialize pose publisher
        self.pose_pub = rospy.Publisher("/nwu_turtlebot_pose", PoseStamped, queue_size=1)

    def pose_callback(self, msg):

        # x y and z data from motion capture
        mo_cap_pos = np.array([[msg.pose.position.x], [msg.pose.position.y], [msg.pose.position.z]])

        # rotate mocap x y z data into the nwu frame
        nwu_pos = np.dot(self.rot_m_t, mo_cap_pos)

        x_nwu = nwu_pos[0][0]
        y_nwu = nwu_pos[1][0]
        z_nwu = nwu_pos[2][0]

        # this is a hack just switching the vector components of the quaternion
        quaternion_nwu = (
            msg.pose.orientation.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.w)

        # fill out the message
        self.pose_nwu.header.stamp = msg.header.stamp
        self.pose_nwu.pose.position.x = x_nwu
        self.pose_nwu.pose.position.y = y_nwu
        self.pose_nwu.pose.position.z = z_nwu
        self.pose_nwu.pose.orientation.x = quaternion_nwu[0]
        self.pose_nwu.pose.orientation.y = quaternion_nwu[1]
        self.pose_nwu.pose.orientation.z = quaternion_nwu[2]
        self.pose_nwu.pose.orientation.w = quaternion_nwu[3]

        # publish the message
        self.pose_pub.publish(self.pose_nwu)


        # euler = tf.transformations.euler_from_quaternion(quaternion_nwu)

        # print "Yaw: " + str(math.degrees(euler[2]))


def main():
    # initialize a node
    rospy.init_node('bot_frame_translator')

    # create instance of ExposureController class
    translator = BotFrameTranslator()

    # spin
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
