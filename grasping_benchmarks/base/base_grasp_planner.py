# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import json
import math
import os
import time

import numpy as np
import rospy
from grasping_benchmarks.base.grasp import Grasp6D

class CameraData:
    rgb_img = None
    depth_img = None
    pc_img = None
    seg_img = None
    bounding_box = None
    intrinsic_params = None
    extrinsic_params = {'position': np.ndarray((3, 1), float), 'rotation': np.eye(3)}


class BaseGraspPlanner(object):
    """The base class for grasp planners

    """
    def __init__(self, cfg:dict):
        """
        Parameters
        ----------
        cfg : dict
            Dictionary of configuration parameters.
        """
        self.cfg = cfg
        self._grasp_poses = []
        self._best_grasp = None
        self._camera_data = CameraData()

    def reset(self):
        self.grasp_poses = []
        self._best_grasp = None
        self._camera_data = CameraData()

    def plan_grasp(self, camera_data, n_candidates=1):
        """Grasp Planner
            Compute candidate grasp poses

        Args:
            camera_data (obj): `CameraData`. Contains the data (img, camera params) necessary to compute the grasp poses
            n_candidates (int): number of candidate grasp poses to return

        Raises:
            NotImplementedError: [description]
        """

        raise NotImplementedError

    def visualize(self):
        """Plot the grasp poses
        """

        pass

    @property
    def grasp_poses(self):
        return self._grasp_poses

    @grasp_poses.setter
    def grasp_poses(self, grasp_poses:list):
        if len(grasp_poses) is 0:
            self._grasp_poses = []

        elif type(grasp_poses[0]) is not Grasp6D:
            raise ValueError('Invalid grasp type. Must be `benchmark_grasping.grasp.Grasp6D`')

        self._grasp_poses = grasp_poses

    @property
    def best_grasp(self):
        return self._best_grasp

    @best_grasp.setter
    def best_grasp(self, best_grasp:Grasp6D):
        if type(best_grasp) is not Grasp6D:
            raise ValueError('Invalid grasp type. Must be `benchmark_grasping.grasp.Grasp6D`')

        self._best_grasp = best_grasp
