#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import json
import math
import os
import time

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy

from autolab_core import YamlConfig
from perception import (CameraIntrinsics, ColorImage, DepthImage, BinaryImage,
                        RgbdImage)
from visualization import Visualizer2D as vis

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from grasping_benchmarks_ros.srv import GraspPlanner
from grasping_benchmarks_ros.msg import BenchmarkGrasp

from grasping_benchmarks.base.base_grasp_planner import CameraData

from grasping_benchmarks.dexnet.dexnet_grasp_planner import DexnetGraspPlanner


class DexnetGraspPlannerService(DexnetGraspPlanner):
    def __init__(self, model_file, fully_conv, cv_bridge, grasp_service_name, grasp_publisher_name):
        """
        Parameters
        ----------
        model_config_file (str): path to model configuration file of type config.json
        fully_conv (bool): flag to use fully-convolutional network
        cv_bridge: (obj:`CvBridge`): ROS `CvBridge`

        grasp_pose_publisher: (obj:`Publisher`): ROS publisher to publish pose of planned grasp for visualization.
        """

        super(DexnetGraspPlannerService, self).__init__(model_file, fully_conv)

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

        self.grasp_poses = []
        ok = self.plan_grasp(camera_data, n_candidates=1)

        if ok:
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

        camera_data.bounding_box = {'min_x': req.bounding_box.minX, 'min_y': req.bounding_box.minY,
                                    'max_x': req.bounding_box.maxX, 'max_y': req.bounding_box.maxY}

        self.grasp_poses = []
        ok = self.plan_grasp(camera_data, n_candidates=1)

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
        ok = self.plan_grasp(camera_data, n_candidates=1)

        if ok:
            return self._create_grasp_planner_srv_msg()
        else:
            return None

    def _create_grasp_planner_srv_msg(self):


        if len(self.grasp_poses) == 0:
            return False

        # --- Create `BenchmarkGrasp` return message --- #
        grasp_msg = BenchmarkGrasp()

        # set pose...
        p = PoseStamped()
        p.header.frame_id = self.best_grasp.ref_frame
        p.header.stamp = rospy.Time.now()

        p.pose.position.x = self.best_grasp.position[0]
        p.pose.position.y = self.best_grasp.position[1]
        p.pose.position.z = self.best_grasp.position[2]

        p.pose.orientation.w = self.best_grasp.quaternion[3]
        p.pose.orientation.x = self.best_grasp.quaternion[0]
        p.pose.orientation.y = self.best_grasp.quaternion[1]
        p.pose.orientation.z = self.best_grasp.quaternion[2]

        grasp_msg.pose = p

        # ...score
        grasp_msg.score.data = self.best_grasp.score

        # ... and width
        grasp_msg.width.data = self.best_grasp.width

        if self.grasp_pose_publisher is not None:
            # Publish the pose alone for easy visualization of grasp
            # pose in Rviz.
            print("publish grasp!! .")
            self.grasp_pose_publisher.publish(p)

        return grasp_msg


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

    if model_dir.lower() == "default":
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "models")

    model_dir = os.path.join(model_dir, model_name)

    # Instantiate the grasp planner.
    grasp_planner = DexnetGraspPlannerService(model_dir, fully_conv, cv_bridge,
                                              grasp_service_name, grasp_publisher_name)

    rospy.loginfo("Grasping Policy Initialized")

    # Spin forever.
    rospy.spin()
