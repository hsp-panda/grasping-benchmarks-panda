#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import json
import math
import os
import time

from grasping_benchmarks.base import grasp
from grasping_benchmarks.base.transformations import matrix_to_quaternion, quaternion_to_matrix
from numpy.core.multiarray import result_type

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy

from autolab_core import YamlConfig
from perception import (CameraIntrinsics, ColorImage, DepthImage, BinaryImage,
                        RgbdImage)
from visualization import Visualizer2D as vis

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from grasping_benchmarks_ros.srv import GraspPlanner, GraspPlannerRequest, GraspPlannerResponse
from grasping_benchmarks_ros.msg import BenchmarkGrasp

from grasping_benchmarks.base.base_grasp_planner import CameraData

from grasping_benchmarks.dexnet.dexnet_grasp_planner import DexnetGraspPlanner

DEBUG = True


class DexnetGraspPlannerService(DexnetGraspPlanner):
    def __init__(self, model_file, fully_conv, grasp_offset, cv_bridge, grasp_service_name, grasp_publisher_name):
        """
        Parameters
        ----------
        model_config_file (str): path to model configuration file of type config.json
        fully_conv (bool): flag to use fully-convolutional network
        grasp_offset (list): static offset transformation to apply to the grasp
        cv_bridge: (obj:`CvBridge`): ROS `CvBridge`

        grasp_pose_publisher: (obj:`Publisher`): ROS publisher to publish pose of planned grasp for visualization.
        """

        super(DexnetGraspPlannerService, self).__init__(model_file, fully_conv, grasp_offset)

        self.camera_viewpoint = PoseStamped()

        self.cv_bridge = cv_bridge

        # Create publisher to publish pose of final grasp.
        self.grasp_pose_publisher = None
        if grasp_publisher_name is not None:
            self.grasp_pose_publisher = rospy.Publisher(grasp_publisher_name, PoseStamped, queue_size=10)

        # Initialize the ROS service.
        self._grasp_planning_service = rospy.Service(grasp_service_name, GraspPlanner,
                                            self.plan_grasp_handler)



    def read_images(self, req):
        """Reads images from a ROS service request.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service.
        """
        # Get the raw depth and color images as ROS `Image` objects.
        raw_color = req.color_image
        raw_depth = req.depth_image

        # Get the raw camera info as ROS `CameraInfo`.
        raw_camera_info = req.camera_info

        # Wrap the camera info in a BerkeleyAutomation/perception
        # `CameraIntrinsics` object.
        camera_intr = CameraIntrinsics(
                        frame=raw_camera_info.header.frame_id,
                        fx = raw_camera_info.K[0],
                        fy = raw_camera_info.K[4],
                        cx= raw_camera_info.K[2],
                        cy = raw_camera_info.K[5],
                        width = raw_camera_info.width,
                        height= raw_camera_info.height,
                        )

        # Create wrapped BerkeleyAutomation/perception RGB and depth images by
        # unpacking the ROS images using ROS `CvBridge`
        try:
            color_im = ColorImage(self.cv_bridge.imgmsg_to_cv2(raw_color, "rgb8"),
                                  frame=camera_intr.frame)

            cv2_depth = self.cv_bridge.imgmsg_to_cv2(raw_depth, desired_encoding="passthrough")
            cv2_depth = np.array(cv2_depth, dtype=np.float32)

            cv2_depth *= 0.001

            nan_idxs = np.isnan(cv2_depth)
            cv2_depth[nan_idxs] = 1000.0

            depth_im = DepthImage(cv2_depth, frame=camera_intr.frame)

        except CvBridgeError as cv_bridge_exception:
            print("except CvBridgeError")
            rospy.logerr(cv_bridge_exception)

        # Check image sizes.
        if color_im.height != depth_im.height or color_im.width != depth_im.width:
            msg = ("Color image and depth image must be the same shape! Color"
                   " is %d x %d but depth is %d x %d") % (
                       color_im.height, color_im.width,
                       depth_im.height, depth_im.width)

            rospy.logerr(msg)
            raise rospy.ServiceException(msg)

        if (color_im.height < self.min_height or color_im.width < self.min_width):
            msg = ("Color image is too small! Must be at least %d x %d"
                   " resolution but the requested image is only %d x %d") % (
                       self.min_height, self.min_width, color_im.height,
                       color_im.width)

            rospy.logerr(msg)
            raise rospy.ServiceException(msg)

        # --- create CameraData struct --- #
        camera_data = CameraData()
        camera_data.rgb_img = color_im
        camera_data.depth_img = depth_im
        camera_data.intrinsic_params = camera_intr

        return camera_data

    def plan_grasp_handler(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        camera_data = self.read_images(req)

        n_of_candidates = req.n_of_candidates if req.n_of_candidates else 1

        self.grasp_poses = []
        ok = self.plan_grasp(camera_data, n_candidates=n_of_candidates)

        if ok:
            self.camera_viewpoint = req.view_point
            return self._create_grasp_planner_srv_msg()
        else:
            return None

    def plan_grasp_bb_handler(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            `ROS ServiceRequest` for grasp planner service.
        """
        camera_data = self.read_images(req)

        n_of_candidates = req.n_of_candidates if req.n_of_candidates else 1

        camera_data.bounding_box = {'min_x': req.bounding_box.minX, 'min_y': req.bounding_box.minY,
                                    'max_x': req.bounding_box.maxX, 'max_y': req.bounding_box.maxY}

        self.grasp_poses = []
        ok = self.plan_grasp(camera_data, n_candidates=n_of_candidates)

        if ok:
            return self._create_grasp_planner_srv_msg()
        else:
            return None

    def plan_grasp_segmask_handler(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """

        camera_data = self.read_images(req)
        raw_segmask = req.segmask

        n_of_candidates = req.n_of_candidates if req.n_of_candidates else 1

        # create segmentation mask
        try:
            camera_data.seg_img = BinaryImage(self.cv_bridge.imgmsg_to_cv2(raw_segmask, desired_encoding="passthrough"),
                                              frame=camera_data.intrinsic_params.frame)

        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

        if camera_data.rgb_img.height != camera_data.seg_img.height or \
           camera_data.rgb_img.width != camera_data.seg_img.width:

            msg = ("Images and segmask must be the same shape! Color image is"
                   " %d x %d but segmask is %d x %d") % (
                       camera_data.rgb_img.height, camera_data.rgb_img.width,
                       camera_data.seg_img.height, camera_data.seg_img.width)

            rospy.logerr(msg)
            raise rospy.ServiceException(msg)

        self.grasp_poses = []
        ok = self.plan_grasp(camera_data, n_candidates=n_of_candidates)

        if ok:
            return self._create_grasp_planner_srv_msg()
        else:
            return None

    def transform_grasp_to_world(self, grasp_pose, camera_viewpoint):
        """Refer the grasp pose to 6D, if the camera viewpoint is given

        Parameters
        ---------
        grasp_pose: geometry_msgs/PoseStamped
            The candidate to transform
        camera_viewpoint: geometry_msgs/PoseStamped
            The camera viewpoint wrt world reference frame

        Returns
        -------
            geometry_msgs/PoseStamped
            The candidate in the world reference frame
        """
        # Need to tranform the grasp pose from camera frame to world frame
        # w_T_cam : camera pose in world ref frame
        # cam_T_grasp : grasp pose in camera ref frame
        # w_T_grasp = w_T_cam * cam_T_grasp

        # Construct the 4x4 affine transf matrices from ROS poses
        grasp_quat = grasp_pose.pose.orientation
        grasp_pos = grasp_pose.pose.position
        cam_T_grasp = np.eye(4)
        cam_T_grasp[:3,:3] = quaternion_to_matrix([grasp_quat.x,
                                                   grasp_quat.y,
                                                   grasp_quat.z,
                                                   grasp_quat.w])
        cam_T_grasp[:3,3] = np.array([grasp_pos.x, grasp_pos.y, grasp_pos.z])

        cam_quat = camera_viewpoint.pose.orientation
        cam_pos = camera_viewpoint.pose.position
        w_T_cam = np.eye(4)
        w_T_cam[:3,:3] = quaternion_to_matrix([cam_quat.x,
                                               cam_quat.y,
                                               cam_quat.z,
                                               cam_quat.w])
        w_T_cam[:3,3] = np.array([cam_pos.x, cam_pos.y, cam_pos.z])

        # Obtain the w_T_grasp affine transformation
        w_T_grasp = np.matmul(w_T_cam, cam_T_grasp)

        if DEBUG:
            print("[DEBUG] Grasp pose reference system transform")
            print("w_T_cam\n ", w_T_cam)
            print("cam_T_grasp\n ", cam_T_grasp)
            print("w_T_grasp\n ", w_T_grasp)


        # Create and return a StampedPose object
        w_T_grasp_quat = matrix_to_quaternion(w_T_grasp[:3,:3])
        result_pose = PoseStamped()
        result_pose.pose.orientation.x = w_T_grasp_quat[0]
        result_pose.pose.orientation.y = w_T_grasp_quat[1]
        result_pose.pose.orientation.z = w_T_grasp_quat[2]
        result_pose.pose.orientation.w = w_T_grasp_quat[3]
        result_pose.pose.position.x = w_T_grasp[0,3]
        result_pose.pose.position.y = w_T_grasp[1,3]
        result_pose.pose.position.z = w_T_grasp[2,3]
        result_pose.header = camera_viewpoint.header

        return result_pose

    def _create_grasp_planner_srv_msg(self):

        response = GraspPlannerResponse()

        if len(self.grasp_poses) == 0:
            return False

        # --- Create `BenchmarkGrasp` list return message --- #
        for grasp_candidate in self.grasp_poses:
            # --- Set the pose in PoseStamped format --- #
            # Grasp poses are grasp.Grasp6D
            grasp_msg = BenchmarkGrasp()
            p = PoseStamped()
            p.header.frame_id = grasp_candidate.ref_frame
            p.header.stamp = rospy.Time.now()
            p.pose.position.x = grasp_candidate.position[0]
            p.pose.position.y = grasp_candidate.position[1]
            p.pose.position.z = grasp_candidate.position[2]
            p.pose.orientation.w = grasp_candidate.quaternion[3]
            p.pose.orientation.x = grasp_candidate.quaternion[0]
            p.pose.orientation.y = grasp_candidate.quaternion[1]
            p.pose.orientation.z = grasp_candidate.quaternion[2]
            grasp_msg.pose = self.transform_grasp_to_world(p, self.camera_viewpoint)

            # --- Set the candidate quality score and width --- #
            grasp_msg.score.data = grasp_candidate.score
            grasp_msg.width.data = grasp_candidate.width

            response.grasp_candidates.append(grasp_msg)

        if self.grasp_pose_publisher is not None:
            # --- Publish poses for visualization on Rviz ---#
            # TODO: properly publish all the poses
            print("Publishing grasps on topic")
            # self.grasp_pose_publisher.publish(response.grasp_candidates[0].pose)

        return response

if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node("Dexnet_Grasp_Planner")

    # Initialize `CvBridge`.
    cv_bridge = CvBridge()

    # Get configs.
    model_name = rospy.get_param("~model_name")
    model_dir = rospy.get_param("~model_dir")
    fully_conv = rospy.get_param("~fully_conv")
    grasp_service_name = rospy.get_param("~grasp_planner_service_name")
    grasp_publisher_name = rospy.get_param("~grasp_publisher_name")
    grasp_offset = rospy.get_param("~grasp_pose_offset", [0.0, 0.0, 0.0])

    grasp_offset = np.array(grasp_offset[:3])

    if model_dir.lower() == "default":
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "models")

    model_dir = os.path.join(model_dir, model_name)

    # Instantiate the grasp planner.
    grasp_planner = DexnetGraspPlannerService(model_dir, fully_conv, grasp_offset, cv_bridge,
                                              grasp_service_name, grasp_publisher_name)

    rospy.loginfo("Grasping Policy Initialized")

    # Spin forever.
    rospy.spin()
