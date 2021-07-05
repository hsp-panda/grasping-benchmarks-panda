#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

"""
This file contains the code of the benchmark service client.
Its features are:

1. connect to the ros camera topics to read rgb, depth, point cloud, camera parameters
2. send a request to a grasp planning algorithm of type GraspPlanner.srv
3. connect with robot to send cartesian grasp pose commands
(4. assess if the grasp was successful or not)
"""

import rospy
import warnings
import message_filters
import tf2_ros
import sensor_msgs.point_cloud2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs.point_cloud2 import read_points
from geometry_msgs.msg import Transform, Pose, PoseStamped, TransformStamped
from std_msgs.msg import Bool
import copy

from grasping_benchmarks.base.transformations import quaternion_to_matrix, matrix_to_quaternion

from grasping_benchmarks_ros.srv import UserCmd, UserCmdRequest, UserCmdResponse
from grasping_benchmarks_ros.srv import GraspPlanner, GraspPlannerRequest, GraspPlannerResponse
from grasping_benchmarks_ros.msg import BenchmarkGrasp

from panda_grasp_srv.srv import PandaGrasp, PandaGraspRequest, PandaGraspResponse

import numpy as np

NUMBER_OF_CANDIDATES = 10

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
        # TODO: I guess we can get rid of this one
        self._grasp_planner_srv = GRASP_PLANNER_SRV[grasp_planner_service]

        rospy.loginfo("GraspingBenchmarksManager: Waiting for grasp planner service...")
        rospy.wait_for_service(grasp_planner_service_name, timeout=30.0)
        self._grasp_planner = rospy.ServiceProxy(grasp_planner_service_name, GraspPlanner)
        rospy.loginfo("...Connected with service {}".format(grasp_planner_service_name))

        # --- panda service --- #
        panda_service_name =  "/panda_grasp_server/panda_grasp"
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
        self._camera_pose = TransformStamped()
        self._root_reference_frame = 'panda_link0'

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
                              get_candidates: computes a number of candidates and saves them to file\n \
                              abort: interrupt grasp computation / do not send computed pose to the robot"

        self._abort = False

        if req.cmd.data == "help":
            print("Available commands are:\n", available_commands)
            return Bool(True)

        elif req.cmd.data == "grasp" or req.cmd.data == "get_candidates":

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

            # Create the service request
            planner_req = GraspPlannerRequest()

            # Fill in the 2D camera data
            planner_req.color_image = self._rgb_msg
            planner_req.depth_image = self._depth_msg
            planner_req.camera_info = self._cam_info_msg
            planner_req.n_of_candidates = NUMBER_OF_CANDIDATES

            # Fill in the camera viewpoint field
            planner_req.view_point.header = self._camera_pose.header
            planner_req.view_point.pose.orientation = self._camera_pose.transform.rotation
            planner_req.view_point.pose.position = self._camera_pose.transform.translation

            # Transform the point cloud in the root reference frame and include it
            transform = self._camera_pose.transform
            pc_in = self._pc_msg
            tr_matrix = np.identity(4)
            tr_matrix[:3, :3] = quaternion_to_matrix([transform.rotation.x,
                                                    transform.rotation.y,
                                                    transform.rotation.z,
                                                    transform.rotation.w])
            tr_matrix[:3, 3] = [transform.translation.x,
                                transform.translation.y,
                                transform.translation.z]
            points=[]
            for p_in in read_points(pc_in, skip_nans=False, field_names=("x", "y", "z", "rgb")):
                p_transformed = np.dot(tr_matrix, np.array([p_in[0], p_in[1], p_in[2], 1.0]))
                p_out=[]
                p_out.append(p_transformed[0])
                p_out.append(p_transformed[1])
                p_out.append(p_transformed[2])
                p_out.append(p_in[3])
                points.append(p_out)

            header = pc_in.header
            header.frame_id = self._camera_pose.header.frame_id
            pc_out = sensor_msgs.point_cloud2.create_cloud(header=header, fields=pc_in.fields, points=points)

            planner_req.cloud = pc_out

            # Set number of candidates
            planner_req.n_of_candidates = NUMBER_OF_CANDIDATES

            if self._verbose:
                print("... send request to server ...")

            # Plan for grasps
            try:
                reply = self._grasp_planner(planner_req)
                if self._verbose:
                    print("Service {} reply is: \n{}" .format(self._grasp_planner.resolved_name, reply))
            except rospy.ServiceException as e:
                print("Service {} call failed: {}" .format(self._grasp_planner.resolved_name, e))
                return Bool(False)

            # Handle abort command
            if self._abort:
                rospy.loginfo("grasp execution was aborted by the user")
                return Bool(False)

            # --- If the request is to execute a grasp, execute the first candidate
            # TODO: not the first candidate, the best
            if req.cmd.data == "grasp":
                return self.execute_grasp(self.get_best_grasp(reply.grasp_candidates), self._camera_pose)
            else:
                return self.dump_grasps(reply.grasp_candidates, self._camera_pose)

        elif req.cmd.data == "abort":
            self._abort = True
            return Bool(True)

    def get_best_grasp(self, grasps:list):

        best_candidate = grasps[0]

        for grasp in grasps[1:]:
            if grasp.score > best_candidate.score:
                best_candidate = grasp

        return best_candidate

    def dump_grasps(self, grasps:list):

        # TODO: not implemented yet

        raise NotImplementedError

    def execute_grasp(self, grasp):
        """Assumes the grasp pose is already in the root reference frame

        Parameters:
            - grasp (obj: BenchmarkGrasp msg)
        """

        # gp_quat = grasp.pose.pose.orientation
        # gp_pose = grasp.pose.pose.position

        # # TODO: THIS SHOULD NOT BE THE CASE. THIS CHECK AND FIX WILL BE REMOVED SHORTLY, SINCE
        # # THE VIEWPOINT IS NOW EMBEDDED INTO THE SERVICE REQUEST AND THE PLANNER SERVICE SHOULD
        # # ANSWER WITH A POSE IN THE ROOT REFERENCE FRAME AND TRANSFORM IT IF NEEDED.

        # if self._grasp_planner_srv is GraspPlanner:
        #     # Need to tranform the grasp pose from camera frame to world frame
        #     # w_T_grasp = w_T_cam * cam_T_grasp

        #     cam_T_grasp = np.eye(4)
        #     cam_T_grasp[:3,:3] = quaternion_to_matrix([gp_quat.x, gp_quat.y, gp_quat.z, gp_quat.w])
        #     cam_T_grasp[:3,3] = np.array([gp_pose.x, gp_pose.y, gp_pose.z])

        #     cam_quat = cam_pose.rotation
        #     cam_pose = cam_pose.translation
        #     w_T_cam = np.eye(4)
        #     w_T_cam[:3, :3] = quaternion_to_matrix([cam_quat.x, cam_quat.y, cam_quat.z, cam_quat.w])
        #     w_T_cam[:3, 3] = np.array([cam_pose.x, cam_pose.y, cam_pose.z])

        #     w_T_grasp = np.matmul(w_T_cam, cam_T_grasp)
        #     print("w_T_cam\n ", w_T_cam)
        #     print("cam_T_grasp\n ", cam_T_grasp)
        #     print("w_T_grasp\n ", w_T_grasp)

        # elif self._grasp_planner_srv is GraspPlannerCloud:
        #     # In this case, there is no need to change the grasp pose reference frame

        #     w_T_grasp = np.eye(4)
        #     w_T_grasp[:3,:3] = quaternion_to_matrix([gp_quat.x, gp_quat.y, gp_quat.z, gp_quat.w])
        #     w_T_grasp[:3,3] = np.array([gp_pose.x, gp_pose.y, gp_pose.z])

        #     # However, we need to make sure the X axis points upwards and not downwards
        #     # We don' t want the wrist to twist too much

        #     z_axis = np.array([0.0, 0.0, 1.0])
        #     # if np.dot(w_T_grasp[:3, 0], z_axis) < 0:
        #     #     rotation_around_z = quaternion_to_matrix([0,0,1,0])
        #     #     w_T_grasp[:3,:3] = np.dot(w_T_grasp[:3,:3], rotation_around_z)

        #     print("w_T_grasp\n ", w_T_grasp)

        # # Create the ROS pose message to send to robot
        # grasp_pose_msg = geometry_msgs.msg.PoseStamped()

        # grasp_pose_msg.header.frame_id = 'panda_link0'

        # grasp_pose_msg.pose.position.x = w_T_grasp[0,3]
        # grasp_pose_msg.pose.position.y = w_T_grasp[1,3]
        # grasp_pose_msg.pose.position.z = w_T_grasp[2,3]

        # q = matrix_to_quaternion(w_T_grasp[:3,:3])
        # grasp_pose_msg.pose.orientation.x = q[0]
        # grasp_pose_msg.pose.orientation.y = q[1]
        # grasp_pose_msg.pose.orientation.z = q[2]
        # grasp_pose_msg.pose.orientation.w = q[3]

        panda_req = PandaGraspRequest()
        panda_req.grasp = grasp.pose
        panda_req.width = grasp.width

        if self._verbose:
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

        self._cam_info_msg = cam_info
        self._rgb_msg = rgb
        self._depth_msg = depth
        self._pc_msg = pc

        self._new_camera_data = True

        # Get the camera transform wrt the root reference frame of this class
        try:
            self._camera_pose = self._tfBuffer.lookup_transform(self._root_reference_frame, self._cam_info_msg.header.frame_id, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            warnings.warn("tf listener could not get camera pose. Are you publishing camera poses on tf?")
            self._camera_pose = TransformStamped()
            self._camera_pose.transform.rotation.w = 1.0

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
