#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

"""
This file contains the code of the benchmark service client.
Its features are:

1. connect to the ros camera topics to read rgb, depth, point cloud, camera parameters
2. send a request to a grasp planning algorithm of type GraspPlanner.srv / GraspPlannerCloud.srv
3. connect with robot to send cartesian grasp pose commands
(4. assess if the grasp was successful or not)

"""

import rospy
import warnings
import message_filters
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Transform
from std_msgs.msg import Bool
import tf2_ros

from grasping_benchmarks.base.transformations import quaternion_to_matrix, matrix_to_quaternion

from grasping_benchmarks_ros.srv import *
from grasping_benchmarks_ros.msg import BenchmarkGrasp

from panda_grasp_srv.srv import PandaGrasp, PandaGraspRequest, PandaGraspResponse

import numpy as np


NEW_MSG = {
"new_data": False,
"data": {},
}

# TODO: find a way to define the grasp service depending on the type of grasp planner
GRASP_PLANNER_SRV = {
    'GraspPlanner': GraspPlanner,
    'GraspPlannerCloud': GraspPlannerCloud,
    }


class GraspingBenchmarksManager(object):
    def __init__(self, grasp_planner_service_name, grasp_planner_service, user_cmd_service_name, panda_service_name, verbose=False):

        self._verbose = verbose

        # --- new grasp command service --- #
        self._new_grasp_srv = rospy.Service(user_cmd_service_name, UserCmd, self.user_cmd)

        # --- grasp planner service --- #
        self._grasp_planner_srv = GRASP_PLANNER_SRV[grasp_planner_service]

        rospy.loginfo("GraspingBenchmarksManager: Waiting for grasp planner service...")
        rospy.wait_for_service(grasp_planner_service_name, timeout=10.0)
        self._grasp_planner = rospy.ServiceProxy(grasp_planner_service_name, self._grasp_planner_srv)
        rospy.loginfo("...Connected with service {}".format(grasp_planner_service_name))

        # --- panda service --- #
        panda_service_name =  "/panda_grasp"
        rospy.loginfo("GraspingBenchmarksManager: Waiting for panda control service...")
        rospy.wait_for_service(panda_service_name, timeout=60.0)
        self._panda = rospy.ServiceProxy(panda_service_name, PandaGrasp)
        rospy.loginfo("...Connected with service {}".format(panda_service_name))

        # --- subscribers to camera topics --- #
        self._cam_info_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self._rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self._depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self._pc_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
        self._seg_sub = rospy.Subscriber('rgb/image_seg', Image, self.seg_img_callback, queue_size=10)

        # --- camera data synchronizer --- #
        self._tss = message_filters.ApproximateTimeSynchronizer([self._cam_info_sub, self._rgb_sub, self._depth_sub, self._pc_sub],
                                                                queue_size=1, slop=0.5)
        self._tss.registerCallback(self._camera_data_callback)

        # --- robot/camera transform listener --- #
        self._tfBuffer = tf2_ros.buffer.Buffer()
        listener = tf2_ros.transform_listener.TransformListener(self._tfBuffer)

        # --- camera messages --- #
        self._cam_info_msg = None
        self._rgb_msg = None
        self._depth_msg = None
        self._pc_msg = None
        self._camera_pose = geometry_msgs.msg.Transform()

        self._seg_msg = NEW_MSG

        self._new_camera_data = False
        self._abort = False

    # ---------------------- #
    # Grasp planning handler #
    # ---------------------- #
    def user_cmd(self, req):
        """New grasp request handler

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        if self._verbose:
            print("Received new command from user...")

        available_commands = "help : display available commands\n \
                              grasp: compute a new grasp and send it to the robot for execution\n \
                              abort: interrupt grasp computation / do not send computed pose to the robot"

        self._abort = False

        if req.cmd.data == "help":
            print("Available commands are:\n", available_commands)
            return Bool(True)

        elif req.cmd.data == "grasp":

            # --- get images --- #
            if self._verbose:
                print("... waiting for images ...")

            count = 0
            while not self._new_camera_data and count < 1000:
                count += 1

            if count >= 1000:
                print("...no images received")
                return Bool(False)

            self._new_camera_data = False

            # --- create srv request --- #
            # GraspPlanner
            if self._grasp_planner_srv is GraspPlanner:

                planner_req = GraspPlannerRequest()

                planner_req.color_image = self._rgb_msg
                planner_req.depth_image = self._depth_msg
                planner_req.camera_info = self._cam_info_msg

            # or GraspPlannerCloud
            elif self._grasp_planner_srv is GraspPlannerCloud:

                planner_req = GraspPlannerCloudRequest()

                # define cloud
                planner_req.cloud = self._pc_msg

                camera_pose_msg = geometry_msgs.msg.Pose()

                camera_pose_msg.position.x = self._camera_pose.translation.x
                camera_pose_msg.position.y = self._camera_pose.translation.y
                camera_pose_msg.position.z = self._camera_pose.translation.z
                camera_pose_msg.orientation = self._camera_pose.rotation

                planner_req.view_point = camera_pose_msg

            if self._verbose:
                print("... send request to server ...")

            if self._abort:
                rospy.loginfo("grasp computation was aborted by the user")
                return Bool(False)

            try:
                reply = self._grasp_planner(planner_req)

                print("Service {} reply is: \n{}" .format(self._grasp_planner.resolved_name, reply))

            except rospy.ServiceException as e:
                print("Service {} call failed: {}" .format(self._grasp_planner.resolved_name, e))

                return Bool(False)

            if self._abort:
                rospy.loginfo("grasp execution was aborted by the user")
                return Bool(False)

            return self.execute_grasp(reply.grasp, self._camera_pose)

        elif req.cmd.data == "abort":
            self._abort = True
            return Bool(True)

    def execute_grasp(self, grasp, cam_pose):
        """
        Parameters:
            - grasp (obj: BenchmarkGrasp msg)
            - cam_pose (obj: geometry_msgs.msg.Transform)
        """
        # Need to tranform the grasp pose from camera frame to world frame
        # w_T_grasp = w_T_cam * cam_T_grasp
        gp_quat = grasp.pose.pose.orientation
        gp_pose = grasp.pose.pose.position
        cam_T_grasp = np.eye(4)
        cam_T_grasp[:3,:3] = quaternion_to_matrix([gp_quat.x, gp_quat.y, gp_quat.z, gp_quat.w])
        cam_T_grasp[:3,3] = np.array([gp_pose.x, gp_pose.y, gp_pose.z])

        cam_quat = cam_pose.rotation
        cam_pose = cam_pose.translation
        w_T_cam = np.eye(4)
        w_T_cam[:3, :3] = quaternion_to_matrix([cam_quat.x, cam_quat.y, cam_quat.z, cam_quat.w])
        w_T_cam[:3, 3] = np.array([cam_pose.x, cam_pose.y, cam_pose.z])

        w_T_grasp = np.matmul(w_T_cam, cam_T_grasp)
        print("w_T_cam\n ", w_T_cam)
        print("cam_T_grasp\n ", cam_T_grasp)
        print("w_T_grasp\n ", w_T_grasp)

        # Create the ROS pose message to send to robot
        grasp_pose_msg = geometry_msgs.msg.PoseStamped()

        grasp_pose_msg.header.frame_id = 'panda_link0'

        grasp_pose_msg.pose.position.x = w_T_grasp[0,3]
        grasp_pose_msg.pose.position.y = w_T_grasp[1,3]
        grasp_pose_msg.pose.position.z = w_T_grasp[2,3]

        q = matrix_to_quaternion(w_T_grasp[:3,:3])
        grasp_pose_msg.pose.orientation.x = q[0]
        grasp_pose_msg.pose.orientation.y = q[1]
        grasp_pose_msg.pose.orientation.z = q[2]
        grasp_pose_msg.pose.orientation.w = q[3]

        panda_req = PandaGraspRequest()
        panda_req.grasp = grasp_pose_msg
        panda_req.width = grasp.width

        print("request to panda is: \n{}" .format(panda_req))

        try:
            reply = self._panda(panda_req)

            print("Service {} reply is: \n{}" .format(self._panda.resolved_name, reply))

            return Bool(True)

        except rospy.ServiceException as e:
            print("Service {} call failed: {}" .format(self._panda.resolved_name, e))

            return Bool(False)

    # ------------------- #
    # Camera data handler #
    # ------------------- #
    def _camera_data_callback(self, cam_info, rgb, depth, pc):
        # rospy.loginfo("New data from camera!")
        self._cam_info_msg = cam_info
        self._rgb_msg = rgb
        self._depth_msg = depth
        self._pc_msg = pc

        self._new_camera_data = True

        # define camera view point
        try:
            camera_pose_tf = self._tfBuffer.lookup_transform( 'panda_link0', self._cam_info_msg.header.frame_id, rospy.Time())
            self._camera_pose = camera_pose_tf.transform

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            warnings.warn("tf listener could not get camera pose. Are you publishing camera poses on tf?")

            self._camera_pose = geometry_msgs.msg.Transform()
            self._camera_pose.rotation.w = 1.0

    def seg_img_callback(self, data):
        if self._verbose:
            print("Got segmentation image...")

        self._seg_msg['data'] = data
        self._seg_msg['new_data'] = True


if __name__ == "__main__":
    # Init node
    rospy.init_node('grasping_benchmarks_manager')

    # Get rosparam config
    grasp_planner_service_name = rospy.get_param("~grasp_planner_service_name")
    grasp_planner_service = rospy.get_param("~grasp_planner_service")
    new_grasp_service_name = rospy.get_param("~user_cmd_service_name")
    panda_service_name = "panda_grasp" # rospy.get_param("/panda_service_name")

    # Instantiate benchmark client class
    bench_manager = GraspingBenchmarksManager(grasp_planner_service_name, grasp_planner_service, new_grasp_service_name, panda_service_name, verbose=True)

    # Spin forever.
    rospy.spin()
