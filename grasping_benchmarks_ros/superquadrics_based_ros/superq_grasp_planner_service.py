#!/usr/bin/env python3
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
import ros_numpy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from grasping_benchmarks_ros.srv import GraspPlanner
from grasping_benchmarks_ros.msg import BenchmarkGrasp

import superquadric_bindings  as sb

from grasping_benchmarks.base.base_grasp_planner import CameraData

from grasping_benchmarks.superquadric_based.superquadrics_grasp_planner import SuperquadricsGraspPlanner


class SuperquadricGraspPlannerService(SuperquadricsGraspPlanner):
    def __init__(self, cfg_file, grasp_service_name, grasp_publisher_name, grasp_offset):
        """
        Parameters
        ----------
        model_config_file (str): path to model configuration file of type config.json
        fully_conv (bool): flag to use fully-convolutional network
        cv_bridge: (obj:`CvBridge`): ROS `CvBridge`

        grasp_pose_publisher: (obj:`Publisher`): ROS publisher to publish pose of planned grasp for visualization.
        """

        super(SuperquadricGraspPlannerService, self).__init__(cfg_file, grasp_offset)

        # Create publisher to publish pose of final grasp.
        self.grasp_pose_publisher = None
        if grasp_publisher_name is not None:
            self.grasp_pose_publisher = rospy.Publisher(grasp_publisher_name, PoseStamped, queue_size=10)

        # Initialize the ROS service.
        self._grasp_planning_service = rospy.Service(grasp_service_name, GraspPlanner,
                                            self.plan_grasp_handler)

        self._visualizer = []


    def read_images(self, req):
        """Reads images from a ROS service request.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service.
        """
        # Get the ROS data
        ros_cloud = req.cloud
        cam_position = np.array([req.view_point.position.x, req.view_point.position.y, req.view_point.position.z])
        cam_quat = np.array([req.view_point.orientation.x, req.view_point.orientation.y,
                             req.view_point.orientation.z, req.view_point.orientation.w])

        # pointcloud2 to numpy array
        pc_data = ros_numpy.numpify(ros_cloud) # TODO: fix color decoding as with this method colors result in NaNs

        points = np.array([pc_data['x'], pc_data['y'], pc_data['z']])
        camera_data = self.create_camera_data(points, cam_position, cam_quat)

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


    def _create_grasp_planner_srv_msg(self):
        if len(self.grasp_poses) == 0:
            return False

        # --- Create `BenchmarkGrasp` return message --- #
        grasp_msg = BenchmarkGrasp()

        # ----
        # The superquadric-based planner compute the grasp pose expressed wrt the robot base
        # To be compatible with the message expected by the benchmark manager,
        # we express it wrt the camera

        robot_T_gp = np.eye(4)
        robot_T_gp[:3,:3] = self.best_grasp.rotation
        robot_T_gp[:3,3] = self.best_grasp.position

        robot_T_cam = np.eye(4)
        robot_T_cam[:3,:3] = self._camera_data.extrinsic_params['rotation']
        robot_T_cam[:3,3] = self._camera_data.extrinsic_params['position']

        cam_T_gp = np.matmul(np.linalg.inv(robot_T_cam), robot_T_gp)

        # set pose...
        p = PoseStamped()
        p.header.frame_id = self._camera_data.intrinsic_params['cam_frame']
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
            self.grasp_pose_publisher.publish(p)

        return grasp_msg


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node("Superquadric_based_Grasp_Planner")


    # Get configs.
    config_file = rospy.get_param("~config_file")
    grasp_service_name = rospy.get_param("~grasp_planner_service_name")
    grasp_publisher_name = rospy.get_param("~grasp_publisher_name")
    grasp_offset = rospy.get_param("~grasp_pose_offset", [0.0, 0.0, 0.0])

    grasp_offset = np.array(grasp_offset[:3])

    # Instantiate the grasp planner.
    grasp_planner = SuperquadricGraspPlannerService(config_file,
                                              grasp_service_name, grasp_publisher_name, grasp_offset)

    rospy.loginfo("Superquadric-based grasp detection server initialized, waiting for a point cloud ...")

    # Spin forever.
    rospy.spin()
