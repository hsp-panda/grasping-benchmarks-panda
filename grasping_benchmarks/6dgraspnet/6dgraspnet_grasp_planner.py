# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import yaml
import os

import math
import numpy as np

from grasping_benchmarks.base.base_grasp_planner import BaseGraspPlanner, CameraData
from grasping_benchmarks.base.grasp import Grasp6D
from grasping_benchmarks.base import transformations as tr

class GraspNetGraspPlanner(BaseGraspPlanner):
    """Grasp planner based on 6Dof-GraspNet

    """
    def __init__(self, cfg_file="cfg/config_panda.yaml", grasp_offset=np.zeros(3)):
        """Constructor

        Parameters
        ----------
        cfg_file : str, optional Path to config YAML file, by default
            "cfg/config_panda.yaml" grasp_offset : np.array, optional 3-d array
            of x,y,z offset to apply to every grasp in eef frame, by default
            np.zeros(3)
        """

        # Parse config YAML
        with open(cfg_file) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        # Call parent constructor
        super(GraspNetGraspPlanner, self).__init__(self.cfg)

        self._camera_data = CameraData()
        self._point_cloud = None
        self._grasp_offset = grasp_offset

    def configure(self, cfg : dict):
        """Additional class configuration

        Parameters
        ----------
        cfg : dict configuration dict, as sourced from the YAML file
        """

        pass

    def create_camera_data(self, rgb_image : np.ndarray, depth_image : np.ndarray, cam_intrinsic_frame :str,
                           fx: float, fy: float, cx: float, cy: float, skew: float, w: int, h: int) -> CameraData:
        """Create the CameraData object in the format expected by the graspnet planner

        Parameters
        ----------
        rgb_image : np.ndarray
            RGB image
        depth_image : np.ndarray
            Depth (float) image
        cam_intrinsic_frame : str
            The reference frame of the images. Grasp poses are computed wrt this frame
        fx : float
            Focal length (x direction)
        fy : float
            Focal length (y direction)
        cx : float
            Principal point (x direction)
        cy : float
            Principal poin (y direction)
        skew : float
            Skew coefficient
        w : int
            Image width
        h : int
            Image height

        Returns
        -------
        CameraData
            Object storing info required by plan_grasp()
        """


        # CameraData also has fields for camera extrinsics, we should use them!


        pass

    def plan_grasp(self, camera_data : CameraData, n_candidates : int):
        """Plan grasps according to visual data. Grasps are returned with
        respect to the camera reference frame

        Parameters
        ----------
        camera_data : CameraData
        n_candidates : int
            Number of grasp candidates to plan and return

        Returns
        -------
        [type]
            [description]
        """



        return super().plan_grasp(camera_data, n_candidates=n_candidates)


    def visualize(self):
        """Visualize point cloud and last batch of computed grasps in a 3D visualizer

        """
        return super().visualize()



