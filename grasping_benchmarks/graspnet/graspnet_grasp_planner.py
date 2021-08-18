# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import yaml
import os
import sys
from types import SimpleNamespace

import math
import numpy as np

from grasping_benchmarks.base.base_grasp_planner import BaseGraspPlanner, CameraData
from grasping_benchmarks.base.grasp import Grasp6D
from grasping_benchmarks.base import transformations as tr

import mayavi.mlab as mlab

# Import the GraspNet implementation code
# Requires a GRASPNET_DIR environment variable pointing to the root of the repo
sys.path.append(os.environ['GRASPNET_DIR'])
os.chdir(os.environ['GRASPNET_DIR'])
from demo.main import get_color_for_pc, backproject
import grasp_estimator
import utils.utils as utils
import utils.visualization_utils as visu_utils

class GraspNetGraspPlanner(BaseGraspPlanner):
    """Grasp planner based on 6Dof-GraspNet

    """
    def __init__(self, cfg_file="cfg/config_graspnet.yaml", grasp_offset=np.zeros(3)):
        """Constructor

        Parameters
        ----------
        cfg_file : str, optional
            Path to config YAML file, by default
            "cfg/config_graspnet.yaml"
        grasp_offset : np.array, optional
            3-d array of x,y,z offset to apply to every grasp in eef
            frame, by default np.zeros(3)
        """

        # Parse config YAML
        with open(cfg_file) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        # Call parent constructor
        super(GraspNetGraspPlanner, self).__init__(self.cfg)

        # Additional configuration of the planner
        self._grasp_offset = grasp_offset
        self.configure(self.cfg)

        self.latest_grasps = []
        self.latest_grasp_scores = []
        self.n_of_candidates = 1


    def configure(self, cfg : dict):
        """Additional class configuration

        Parameters
        ----------
        cfg : dict configuration dict, as sourced from the YAML file
        """

        # Create a namespace from the config dict
        # Since the GraspNet implementation uses a namespace
        self.cfg_ns = SimpleNamespace(**self.cfg)

        self.grasp_sampler_args = utils.read_checkpoint_args(self.cfg_ns.grasp_sampler_folder)
        self.grasp_sampler_args.is_train = False
        self.grasp_evaluator_args = utils.read_checkpoint_args(self.cfg_ns.grasp_evaluator_folder)
        self.grasp_evaluator_args.continue_train = True

        self.estimator = grasp_estimator.GraspEstimator(self.grasp_sampler_args,
                                                        self.grasp_evaluator_args,
                                                        self.cfg_ns)


    def create_camera_data(self, rgb_image : np.ndarray, depth_image : np.ndarray, cam_intrinsic_frame :str, cam_extrinsic_matrix : np.ndarray,
                           fx: float, fy: float, cx: float, cy: float, skew: float, w: int, h: int, obj_cloud : np.ndarray = None) -> CameraData:

        """Create the CameraData object in the format expected by the graspnet planner

        Parameters
        ----------
        rgb_image : np.ndarray
            RGB image
        depth_image : np.ndarray
            Depth (float) image
        cam_intrinsic_frame : str
            The reference frame ID of the images. Grasp poses are computed wrt this
            frame
        cam_extrinsic_matrix : np.ndarray
            The 4x4 camera pose in world reference frame
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
        obj_cloud : np.ndarray, optional
            Object point cloud to use for grasp planning (a segmented portion of
            the point cloud could be given here)

        Returns
        -------
        CameraData
            Object storing info required by plan_grasp()
        """

        camera_data = CameraData()
        camera_data.rgb_img = rgb_image
        camera_data.depth_img = depth_image
        intrinsic_matrix = np.array([[fx, skew,  cx],
                                     [0,    fy,  cy],
                                     [0,     0,   1]])
        camera_data.intrinsic_params = {
                                        'fx' : fx,
                                        'fy' : fy,
                                        'cx' : cx,
                                        'cy' : cy,
                                        'skew' : skew,
                                        'w' : w,
                                        'h' : h,
                                        'frame' : cam_intrinsic_frame,
                                        'matrix' : intrinsic_matrix
                                        }
        camera_data.extrinsic_params['position'] = cam_extrinsic_matrix[:3,3]
        camera_data.extrinsic_params['rotation'] = cam_extrinsic_matrix[:3,:3]

        # Remove missing depth readouts
        # Filter out zeros and readings beyond 2m
        np.nan_to_num(camera_data.depth_img, copy=False, nan=0)
        invalid_mask = np.where(np.logical_or(camera_data.depth_img==0, camera_data.depth_img>2))
        camera_data.depth_img[invalid_mask] = np.nan

        # Obtain the (colored) scene point cloud from valid points
        self.scene_pc, selection = backproject(camera_data.depth_img,
                                          camera_data.intrinsic_params['matrix'],
                                          return_finite_depth = True,
                                          return_selection = True)
        pc_colors = camera_data.rgb_img.copy()
        pc_colors = np.reshape(pc_colors, [-1, 3])
        pc_colors = pc_colors[selection, :]
        self.scene_pc_colors = pc_colors

        # The algorithm requires an object point cloud, low-pass filtered over
        # 10 frames. For now, we just use what comes from the camera.
        # If a point cloud is passed to this function, use that as object pc for
        # planning. Otherwise, use the scene cloud as object cloud
        if obj_cloud is not None:
            self.object_pc = obj_cloud
        else:
            self.object_pc = self.scene_pc

        # Not sure if we should return a CameraData object or simply assign it
        self._camera_data = camera_data
        return camera_data


    def plan_grasp(self, camera_data : CameraData, n_candidates : int) -> bool:
        """Plan grasps according to visual data. Grasps are returned with
        respect to the camera reference frame

        Parameters
        ----------
        camera_data : CameraData
        n_candidates : int
            Number of grasp candidates to plan and return

        Returns
        -------
        bool
            True if any number of candidates could be retrieved, False otherwise
        """

        self.n_of_candidates = n_candidates

        # Compute grasps according to the pytorch implementation
        self.latest_grasps, self.latest_grasp_scores = self.estimator.generate_and_refine_grasps(self.object_pc)
        self.grasp_poses.clear()

        # Sort grasps from best to worst
        # (Assume grasps and scores are lists)
        if len(self.latest_grasps) >= n_candidates:
            sorted_grasps_quality_list = sorted(zip(self.latest_grasp_scores, self.latest_grasps), key=lambda pair: pair[0], reverse=True)
            self.latest_grasps = [g[1] for g in sorted_grasps_quality_list]
            self.latest_grasp_scores = [g[0] for g in sorted_grasps_quality_list]
        else:
            return False

        # Organize grasps in a Grasp class
        # Grasps are specified wrt the camera ref frame

        for grasp,score in zip(self.latest_grasps[:n_candidates], self.latest_grasp_scores[:n_candidates]):
            # Grasps should already be in 6D as output
            # A 90 degrees offset is applied to account for the difference in
            # reference frame (see
            # https://github.com/NVlabs/6dof-graspnet/issues/8)
            # when dealing with the real robot hand
            #TODO Offset should be applied here too
            offset_transform = np.array([[0,-1, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])
            grasp_with_offset = np.dot(grasp, offset_transform)
            grasp_6d = Grasp6D(position=grasp_with_offset[:3, 3],
                               rotation=grasp_with_offset[:3,:3],
                               width=0, score=score,
                               ref_frame=camera_data.intrinsic_params['frame'])
            self.grasp_poses.append(grasp_6d)

        self.best_grasp = self.grasp_poses[0]

        return True

    def visualize(self, visualize_all : bool = True):
        """Visualize point cloud and last batch of computed grasps in a 3D visualizer
        """

        candidates_to_display = self.n_of_candidates if ((self.n_of_candidates > 0) and (self.n_of_candidates < len(self.latest_grasps)) and not visualize_all) else len(self.latest_grasps)

        mlab.figure(bgcolor=(1,1,1))
        visu_utils.draw_scene(
            pc=self.scene_pc,
            grasps=self.latest_grasps[:candidates_to_display],
            grasp_scores=self.latest_grasp_scores[:candidates_to_display],
            pc_color=self.scene_pc_colors,
            show_gripper_mesh=True
        )
        print('[INFO] Close visualization window to proceed')
        mlab.show()

