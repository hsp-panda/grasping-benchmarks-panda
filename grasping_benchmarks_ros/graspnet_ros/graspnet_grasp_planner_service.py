#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

from typing import Tuple
import yaml
import math
import os
import time
from threading import Lock

from cv_bridge import CvBridge, CvBridgeError
import ros_numpy
import numpy as np
import cv_bridge
import rospy
import ctypes
import struct

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2

from grasping_benchmarks.base.transformations import quaternion_to_matrix, matrix_to_quaternion
from grasping_benchmarks.base.base_grasp_planner import CameraData
from grasping_benchmarks_ros.srv import GraspPlanner, GraspPlannerRequest, GraspPlannerResponse
from grasping_benchmarks_ros.msg import BenchmarkGrasp
from grasping_benchmarks.graspnet.graspnet_grasp_planner import GraspNetGraspPlanner

class VisuMutex:
    """Simple class to synchronize threads on whether to display stuff or not
    """
    def __init__(self):
        self._mutex = Lock()
        self._ready_to_visu = False

    @property
    def isReady(self):
        return self._ready_to_visu

    def setReadyState(self, state : bool):
        self._mutex.acquire()
        try:
            self._ready_to_visu = state
        finally:
            self._mutex.release()

DEBUG = False

visualization_mutex = VisuMutex()

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

        super(GraspnetGraspPlannerService, self).__init__(cfg_file, grasp_offset)
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
        if (cv2_color.shape[0] != cv2_depth.shape[0]) or (cv2_color.shape[1] != cv2_depth.shape[1]):
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
        camera_orientation = quaternion_to_matrix([viewpoint.pose.orientation.x,
                                           viewpoint.pose.orientation.y,
                                           viewpoint.pose.orientation.z,
                                           viewpoint.pose.orientation.w])
        camera_extrinsic_matrix = np.eye(4)
        camera_extrinsic_matrix[:3,:3] = camera_orientation
        camera_extrinsic_matrix[:3,3] = camera_position

        #  If available, get the object point cloud and transform it in the
        #  camera ref frame
        obj_cloud = self.npy_from_pc2(req.cloud)[0]
        obj_cloud = self.transform_pc_to_camera_frame(obj_cloud, camera_extrinsic_matrix) if obj_cloud is not None else None

        camera_data = self.create_camera_data(rgb_image=cv2_color,
                                              depth_image=cv2_depth,
                                              cam_intrinsic_frame=camera_info.header.frame_id,
                                              cam_extrinsic_matrix=camera_extrinsic_matrix,
                                              fx=camera_info.K[0],
                                              fy=camera_info.K[4],
                                              cx=camera_info.K[2],
                                              cy=camera_info.K[5],
                                              skew=0.0,
                                              w=camera_info.width,
                                              h=camera_info.height,
                                              obj_cloud=obj_cloud)

        return camera_data

    def plan_grasp_handler(self, req : GraspPlannerRequest) -> GraspPlannerResponse:

        # Read camera images from the request
        camera_data = self.read_images(req)

        # Set number of candidates
        n_of_candidates = req.n_of_candidates if req.n_of_candidates else 1

        ok = self.plan_grasp(camera_data, n_of_candidates)
        if ok:
            # Communicate to main thread that we are ready to visu
            visualization_mutex.setReadyState(True)
            self.camera_viewpoint = req.view_point
            return self._create_grasp_planner_srv_msg()
        else:
            return GraspPlannerResponse()

    def transform_pc_to_camera_frame(self, pc : np.ndarray, camera_pose : np.ndarray) -> np.ndarray:
        """Transform the point cloud from root to camera reference frame

        Parameters
        ----------
        pc : np.ndarray
            nx3, float64 array of points
        camera_pose : np.ndarray
            4x4 camera pose, affine transformation

        Returns
        -------
        np.ndarray
            [description]
        """

        # [F]_p         : point p in reference frame F
        # [F]_T_[f]     : frame f in reference frame F
        # r             : root ref frame
        # cam           : cam ref frame
        # p, pc         : point, point cloud (points as rows)
        # tr(r_pc) = r_T_cam * tr(cam_pc)
        # tr(cam_pc) = inv(r_T_cam) * tr(r_pc)

        r_pc = np.c_[pc, np.ones(pc.shape[0])]
        cam_T_r = np.linalg.inv(camera_pose)
        cam_pc = np.transpose(np.dot(cam_T_r, np.transpose(r_pc)))

        return cam_pc[:, :-1]

    def transform_grasp_to_world(self, grasp_pose : PoseStamped, camera_viewpoint : PoseStamped) -> PoseStamped:
        """Transform the 6D grasp pose in the world reference frame, given the
        camera viewpoint

        Parameters
        ----------
        grasp_pose : geometry_msgs/PoseStamped
            Candidate grasp pose, in the camera ref frame
        camera_viewpoint : geometry_msgs/PoseStamped
            Camera pose wrt world ref frame

        Returns
        -------
        PoseStamped
            Candidate grasp pose, in the world reference frame
        """

        # w_T_cam : camera pose in world ref frame
        # cam_T_grasp : grasp pose in camera ref frame
        # w_T_grasp = w_T_cam * cam_T_grasp

        # Construct the 4x4 affine transf matrices from ROS stamped poses
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

    def _create_grasp_planner_srv_msg(self) -> GraspPlannerResponse:
        """Create service response message

        Returns
        -------
        GraspPlannerResponse
            Service response message
        """

        response = GraspPlannerResponse()

        for grasp_candidate in self.grasp_poses:
            # Transform Grasp6D candidates in PoseStamped candidates
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

            # Set candidate score and width
            grasp_msg.score.data = grasp_candidate.score
            grasp_msg.width.data = grasp_candidate.width

            response.grasp_candidates.append(grasp_msg)

        if self.grasp_pose_publisher is not None:
            # Publish poses for direct rViz visualization
            # TODO: properly publish all the poses
            print("Publishing grasps on topic")
            self.grasp_pose_publisher.publish(response.grasp_candidates[0].pose)

        return response


    def npy_from_pc2(self, pc : PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
        """Conversion from PointCloud2 to a numpy format
        Parameters
        ----------
        pc : PointCloud2
            Scene or object pc
        Returns
        -------
        Tuple[np.array, np.array]
            Point cloud in a nx3 array, where rows are xyz, and nx3 array where
            rows are rgb
        """

        pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(pc)

        # Decode x,y,z
        # NaNs are removed later
        points_xyz = ros_numpy.point_cloud2.get_xyz_points(pc_data, remove_nans=False)

        # Decode r,g,b
        pc_data_rgb_split = ros_numpy.point_cloud2.split_rgb_field(pc_data)
        points_rgb = np.column_stack((pc_data_rgb_split['r'], pc_data_rgb_split['g'], pc_data_rgb_split['b']))

        # Find NaNs and get remove their indexes
        valid_point_indexes = np.argwhere(np.invert(np.bitwise_or.reduce(np.isnan(points_xyz), axis=1)))
        valid_point_indexes = np.reshape(valid_point_indexes, valid_point_indexes.shape[0])

        return points_xyz[valid_point_indexes], points_rgb[valid_point_indexes]


if __name__ == "__main__":

    # Initialize the ROS node
    rospy.init_node("graspnet_grasp_planner")

    # Initialize CvBridge
    cv_bridge = CvBridge()

    # Get configuration options
    cfg_file = rospy.get_param("~config_file")
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

    # Visualization needs to be called from the main thread with mayavi
    while(not rospy.is_shutdown()):
        if visualization_mutex.isReady:
            visualization_mutex.setReadyState(False)
            grasp_planner.visualize(visualize_all=False)
        rospy.sleep(1)





