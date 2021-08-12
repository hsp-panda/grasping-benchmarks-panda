#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import yaml
import math
import os
import time

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv_bridge
import rospy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from grasping_benchmarks.base.transformations import quaternion_to_matrix
from grasping_benchmarks.base.base_grasp_planner import CameraData
from grasping_benchmarks_ros.srv import GraspPlanner, GraspPlannerRequest, GraspPlannerResponse
from grasping_benchmarks_ros.msg import BenchmarkGrasp
from grasping_benchmarks.graspnet.graspnet_grasp_planner import GraspNetGraspPlanner

class GraspnetGraspPlannerService(GraspNetGraspPlanner):
    def __init__(self, cfg_file : str, cv_bridge : CvBridge, grasp_offset : np.array, grasp_service_name : str, grasp_publisher_name : str):
        """Constructor

        Parameters
        ----------
        cfg_file : str
            YAML file with GraspNet configuration
        cv_bridge : CvBridge
            Pointer to the ROS node cv_bridge
        grasp_offset : np.array
            3-d array of x,y,z offset to apply to every grasp in eef
            frame, by default np.zeros(3)
        grasp_service_name : str
            Name of the service spawned by this node
        grasp_publisher_name : str
            Name of the ROS publisher to send planned grasps for visualization
        """

        super(GraspNetGraspPlanner, self).__init__(cfg_file=cfg_file, grasp_offset=grasp_offset)
        self.cv_bridge = cv_bridge

        # Create publisher to publish pose of final grasp.
        self.grasp_pose_publisher = None
        if grasp_publisher_name is not None:
            self.grasp_pose_publisher = rospy.Publisher(grasp_publisher_name, PoseStamped, queue_size=10)

        # Initialize the ROS grasp planning service.
        self._grasp_planning_service = rospy.Service(grasp_service_name, GraspPlanner,
                                            self.plan_grasp_handler)

    def read_images(self, req : GraspPlannerRequest):
        """Read images as a CameraData class from a service request

        Parameters
        ----------
        req : GraspPlannerRequest
            ROS service request for the grasp planning service
        """

        # Get color, depth and camera parameters from request
        camera_info = req.camera_info
        viewpoint = req.view_point
        try:
            cv2_color = self.cv_bridge.imgmsg_to_cv2(req.color_image, desired_encoding='rgb8')

            raw_depth = self.cv_bridge.imgmsg_to_cv2(req.depth_image, desired_encoding='passthrough')
            cv2_depth = np.array(raw_depth, dtype=np.float32)

            cv2_depth *= 0.001

            # Fix NANs
            nans_idx = np.isnan(cv2_depth)
            cv2_depth[nans_idx] = 1000.0

        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

        # Check for image and depth size
        if cv2_color.shape != cv2_depth.shape:
            msg = "Mismatch between depth shape {}x{} and color shape {}x{}".format(cv2_depth.shape[0],
                                                                                    cv2_depth.shape[1],
                                                                                    cv2_color.shape[0],
                                                                                    cv2_color.shape[1])
            rospy.logerr(msg)
            raise rospy.ServiceException(msg)

        # Create CameraData structure
        # CameraInfo intrinsic camera matrix for the raw (distorted) images:
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        camera_position = np.array([viewpoint.pose.position.x,
                                    viewpoint.pose.position.y,
                                    viewpoint.pose.position.z])
        camera_orientation = quaternion_to_matrix(viewpoint.pose.quaternion.x,
                                           viewpoint.pose.quaternion.y,
                                           viewpoint.pose.quaternion.z,
                                           viewpoint.pose.quaternion.w)
        camera_extrinsic_matrix = np.eye(4)
        camera_extrinsic_matrix[:3,:3] = camera_orientation
        camera_extrinsic_matrix[:3,3] = camera_position
        camera_data = self.create_camera_data(rgb_image=cv2_color,
                                              depth_image=cv2_depth,
                                              cam_intrinsic_frame=camera_info.header.frame_id,
                                              cam_extrinsic_matrix=camera_extrinsic_matrix,
                                              fx=camera_info.K[0],
                                              fy=camera_info.K[4],
                                              cx=camera_info.K[2],
                                              cy=camera_info.K[5],
                                              w=camera_info.width,
                                              h=camera_info.height)

        return camera_data

    def plan_grasp_handler(self, req):

        raise NotImplementedError
        pass

    def transform_grasp_to_world(self, grasp_pose, camera_viewpoint):

        raise NotImplementedError
        pass

    def _create_grasp_planner_srv_msg(self):

        raise NotImplementedError
        pass

if __name__ == "__main__":

    # Initialize the ROS node
    rospy.init_node("Graspnet_Grasp_Planner")

    # Initialize CvBridge
    cv_bridge = CvBridge()

    # Get configuration options
    # model filename?
    # model options?
    # config filename and path?
    cfg_file = "cfg/config_graspnet.yaml"
    grasp_service_name = rospy.get_param("~grasp_planner_service_name")
    grasp_publisher_name = rospy.get_param("~grasp_publisher_name")
    grasp_offset = rospy.get_param("~grasp_pose_offset", [0.0, 0.0, 0.0])
    grasp_offset = np.array(grasp_offset[:3])

    # Initialize the grasp planner service
    grasp_planner = GraspnetGraspPlannerService(cfg_file=cfg_file,
                                                cv_bridge=cv_bridge,
                                                grasp_offset=grasp_offset,
                                                grasp_service_name=grasp_service_name,
                                                grasp_publisher_name=grasp_publisher_name)

    rospy.loginfo("Grasping Policy Initiated")
    rospy.spin()




