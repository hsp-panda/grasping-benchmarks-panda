"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

ROS Server for planning GQ-CNN grasps.

Author
-----
Vishal Satish & Jeff Mahler
"""

# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import json
import math
import os
import time
import warnings
import numpy as np

from scipy.spatial.transform import Rotation as R

from autolab_core import YamlConfig
from perception import (CameraIntrinsics, ColorImage, DepthImage, BinaryImage,
                        RgbdImage)
from visualization import Visualizer2D as vis
from gqcnn.grasping import (Grasp2D, SuctionPoint2D, RgbdImageState,
                            RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode, NoValidGraspsException

from grasping_benchmarks.base import transformations
from grasping_benchmarks.base.base_grasp_planner import BaseGraspPlanner, CameraData
from grasping_benchmarks.base.grasp import Grasp6D



class DexnetGraspPlanner(BaseGraspPlanner):
    def __init__(self, model_dir, fully_conv):
        """
        Parameters
        ----------
            model_dir (str): path to model
            fully_conv (bool): flag to use fully-convolutional network
        """
        self.configure(model_dir, fully_conv)
        super(DexnetGraspPlanner, self).__init__(self.cfg)

        self._dexnet_gp = None


    def configure(self, model_dir, fully_conv):
        """Configure model and grasping policy

        Args:
            model_config_file (str): path to model configuration file of type config.json
            fully_conv (bool): if fully-convolutional network

        """

        # read model config.json file
        try:
            model_config = json.load(open(os.path.join(model_dir, "config.json"), "r"))
        except Exception:
            raise ValueError(
                    "Cannot open model config file {}".format(os.path.join(model_dir, "config.json")))

        # --- set gripper mode --- #
        if "gripper_mode" in model_config["gqcnn"]:
            gripper_mode = model_config["gqcnn"]["gripper_mode"]

        else:
            input_data_mode = model_config["gqcnn_config"]["input_data_mode"]

            if input_data_mode == "tf_image":
                gripper_mode = GripperMode.LEGACY_PARALLEL_JAW

            elif input_data_mode == "parallel_jaw":
                gripper_mode = GripperMode.PARALLEL_JAW

            else:
                raise ValueError(
                    "Input data mode {} not supported!".format(input_data_mode))

        if (gripper_mode != GripperMode.LEGACY_PARALLEL_JAW and gripper_mode != GripperMode.PARALLEL_JAW):
            raise ValueError("Gripper mode {} not supported!".format(gripper_mode))

        # --- Load config --- #
        config_file = "cfg/fc_gqcnn_pj.yaml" if fully_conv else "cfg/gqcnn_pj.yaml"

        try:
            config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)

        except Exception:
            print("cannot open configuration file {}".format(
                  os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)))

        # Read config
        self.cfg = YamlConfig(config_filename)

        # --- Define the grasping policy --- #
        policy_cfg = self.cfg["policy"]
        policy_cfg["metric"]["gqcnn_model"] = model_dir

        policy_type = "cem"
        if "type" in policy_cfg:
            policy_type = policy_cfg["type"]

        if policy_type == "fully_conv_pj":
            self.grasping_policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_cfg)

        elif policy_type == "ranking":
            self.grasping_policy = RobustGraspingPolicy(policy_cfg)

        elif policy_type == "cem":
            self.grasping_policy = CrossEntropyRobustGraspingPolicy(policy_cfg)

        else:
            fc = "fully-convolutional" if fully_conv else ""
            raise ValueError("Invalid {} policy type: {}".format(fc, policy_type))

        # --- Set minimum input dimensions --- #
        if policy_type is "fully_conv_pj":
            self.min_width = self.cfg["policy"]["gqcnn_recep_w"]
            self.min_height = self.cfg["policy"]["gqcnn_recep_h"]

        else:
            pad = max(math.ceil(np.sqrt(2) * (float(self.cfg["policy"]["metric"]["crop_width"]) / 2)),
                      math.ceil(np.sqrt(2) * (float(self.cfg["policy"]["metric"]["crop_height"]) / 2)))

            self.min_width = 2 * pad + self.cfg["policy"]["metric"]["crop_width"]
            self.min_height = 2 * pad + self.cfg["policy"]["metric"]["crop_height"]


    def create_camera_data(self, rgb: np.ndarray, depth: np.ndarray,
                           cam_intr_frame: str, fx: float, fy: float,
                           cx: float, cy: float, skew: float, w: int, h: int,
                           seg_mask: np.ndarray = np.empty(shape=(0,)), bbox: tuple = ()):

        """Create the CameraData object in the correct format expected by gqcnn

        Parameters
        ---------
        req: rgb: np.ndarray: rgb image
             depth: np.ndarray: depth image
             cam_intr_frame: str: the reference frame of the images. Grasp poses are computed wrt this frame
             fx: float: focal length (x)
             fy: float: focal length (y)
             cx: float: principal point (x)
             cy: float: principal point (y)
             skew: float: skew coefficient
             w: int: width
             h: int: height
        opt: seg_mask: np.ndarray: segmentation mask
             bbox: tuple: a tuple of 4 values that define the mask bounding box = (x_min, y_min, x_max, y_max)

        Returns:
            CameraData: object that stores the input data required by plan_grasp()
        """

        camera_data = CameraData()

        # Create images
        camera_data.rgb_img = ColorImage(rgb, frame=cam_intr_frame)
        camera_data.depth_img = DepthImage(depth, frame=cam_intr_frame)

        if seg_mask.size > 0:
            camera_data.seg_img = BinaryImage(seg_mask, cam_intr_frame)

        # Check image sizes
        if camera_data.rgb_img.height != camera_data.depth_img.height or \
           camera_data.rgb_img.width != camera_data.depth_img.width:

            msg = ("Color image and depth image must be the same shape! Color"
                   " is %d x %d but depth is %d x %d") % (
                       camera_data.rgb_img.height, camera_data.rgb_img.width,
                       camera_data.depth_img.height, camera_data.depth_img.width)

            raise AssertionError(msg)

        if (camera_data.rgb_img.height < self.min_height
                or camera_data.rgb_img.width < self.min_width):

            msg = ("Color image is too small! Must be at least %d x %d"
                   " resolution but the requested image is only %d x %d") % (
                       self.min_height, self.min_width,
                       camera_data.rgb_img.height, camera_data.rgb_img.width)

            raise AssertionError(msg)

        if camera_data.rgb_img.height != camera_data.seg_img.height or \
            camera_data.rgb_img.width != camera_data.seg_img.width:

            msg = ("Images and segmask must be the same shape! Color image is"
                " %d x %d but segmask is %d x %d") % (
                    camera_data.rgb_img.height, camera_data.rgb_img.width,
                    camera_data.seg_img.height, camera_data.seg_img.width)

            raise AssertionError(msg)

        # set intrinsic params
        camera_data.intrinsic_params = CameraIntrinsics(cam_intr_frame, fx, fy, cx, cy, skew, h, w)

        # set mask bounding box
        if len(bbox) == 4:
            camera_data.bounding_box = {'min_x': bbox[0], 'min_y': bbox[1], 'max_x': bbox[2], 'max_y': bbox[3]}

        return camera_data

    def transform_grasp_to_6D(self, grasp_pose, camera_intrinsics):
        """Planar to 6D grasp pose

        Args:
            grasp_pose (obj:`gqcnn.grasping.Grasp2D` or :obj:`gqcnn.grasping.SuctionPoint2D`)
            camera_intrinsics (obj: `CameraIntrinsics`)

        Returns:
            cam_T_grasp (np.array(shape=(4,4))): 6D transform of the grasp pose wrt camera frame
        """

        u = grasp_pose.center[0] - camera_intrinsics.cx
        v = grasp_pose.center[1] - camera_intrinsics.cy

        X = (grasp_pose.depth * u) / camera_intrinsics.fx
        Y = (grasp_pose.depth * v) / camera_intrinsics.fy

        grasp_pos = [X, Y, grasp_pose.depth]
        euler = [0, 0, -1.57 + grasp_pose.angle]

        rot = R.from_euler('xyz', euler)
        cam_R_grasp = rot.as_dcm()

        cam_T_grasp = np.append(cam_R_grasp, np.array([grasp_pos]).T, axis=1)
        cam_T_grasp = np.append(cam_T_grasp, np.array([[0, 0, 0, 1]]), axis=0)

        grasp_target_T_panda_ef = np.eye(4)
        grasp_target_T_panda_ef[2, 3] = -0.13

        cam_T_grasp = np.matmul(cam_T_grasp, grasp_target_T_panda_ef)

        return cam_T_grasp

    def plan_grasp(self, camera_data, n_candidates=1):
        """Grasp planner.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """

        self._camera_data = camera_data

        # --- Inpaint images --- #
        color_im = camera_data.rgb_img.inpaint(
            rescale_factor=self.cfg["inpaint_rescale_factor"])

        depth_im = camera_data.depth_img.inpaint(
            rescale_factor=self.cfg["inpaint_rescale_factor"])

        # --- Init segmask --- #
        if camera_data.seg_img is None:
            segmask = BinaryImage(255 *
                                  np.ones(depth_im.shape).astype(np.uint8),
                                  frame=color_im.frame)
        else:
            segmask = camera_data.seg_mask


        # --- Aggregate color and depth images into a single
        # BerkeleyAutomation/perception `RgbdImage` --- #
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)

        # --- Mask bounding box --- #
        if camera_data.bounding_box is not None:
            # Calc bb parameters.
            min_x = camera_data.bounding_box['min_x']
            min_y = camera_data.bounding_box['min_y']
            max_x = camera_data.bounding_box['max_x']
            max_y = camera_data.bounding_box['max_y']

            # Contain box to image->don't let it exceed image height/width
            # bounds.
            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0
            if max_x > rgbd_im.width:
                max_x = rgbd_im.width
            if max_y > rgbd_im.height:
                max_y = rgbd_im.height

            # Mask.
            bb_segmask_arr = np.zeros([rgbd_im.height, rgbd_im.width])
            bb_segmask_arr[min_y:max_y, min_x:max_x] = 255
            bb_segmask = BinaryImage(bb_segmask_arr.astype(np.uint8),
                                     segmask.frame)
            segmask = segmask.mask_binary(bb_segmask)

        # --- Create an `RgbdImageState` with the cropped `RgbdImage` and `CameraIntrinsics` --- #
        rgbd_state = RgbdImageState(rgbd_im, camera_data.intrinsic_params, segmask=segmask)

        # --- Execute policy --- #
        try:
            grasps_and_predictions = self.execute_policy(rgbd_state, self.grasping_policy,
                                                     camera_data.intrinsic_params.frame,
                                                     n_candidates)

            self._dexnet_gp = grasps_and_predictions

            # --- project planar grasps to 3D space --- #
            l = []
            for gp in grasps_and_predictions:

                # my method
                pose_6d = self.transform_grasp_to_6D(gp[0], camera_data.intrinsic_params)
                pos = pose_6d[:3,3]
                rot = pose_6d[:3, :3]
                grasp_6D = Grasp6D(position=pos, rotation=rot,
                                   width=gp[0].width, score= gp[1],
                                   ref_frame=camera_data.intrinsic_params.frame)

                l.append(grasp_6D)

                # dexnet method --> needs autolab_core installed as catkin package
                # 6D_gp = gp[0].pose()

            self.grasp_poses = l
            self.best_grasp = l[0]

            self.visualize()

            return True

        except NoValidGraspsException:
            warnings.warn(("While executing policy found no valid grasps from sampled antipodal point pairs!"))

            return False

    def visualize(self):

        vis.clf()

        # Visualize.
        if self.cfg["vis"]["color_image"]:
            vis.imshow(self._camera_data.rgb_img)
            vis.show()
        if self.cfg["vis"]["depth_image"]:
            vis.imshow(self._camera_data.depth_img)
            vis.show()

        if self.cfg["vis"]["segmask"]:
            vis.imshow(self._camera_data.seg_img)
            vis.show()

        if self.cfg["vis"]["best_grasp"]:
            vis.imshow(self._camera_data.rgb_img)
            vis.grasp(self._dexnet_gp[0][0], scale=2.5, show_center=True, show_axis=True)
            vis.show()



    def execute_policy(self, rgbd_image_state, grasping_policy, pose_frame, n_candidates):
        """Executes a grasping policy on an `RgbdImageState`.

        Parameters
        ----------
        rgbd_image_state: :obj:`RgbdImageState`
            `RgbdImageState` from BerkeleyAutomation/perception to encapsulate
            depth and color image along with camera intrinsics.
        grasping_policy: :obj:`GraspingPolicy`
            Grasping policy to use.
        grasp_pose_publisher: :obj:`Publisher`
            ROS publisher to publish pose of planned grasp for visualization.
        pose_frame: :obj:`str`
            Frame of reference to publish pose in.
        """

        print("executing policy..")

        # --- Plan grasp poses --- #
        grasp_planning_start_time = time.time()

        grasps, q_values = grasping_policy.action_set(rgbd_image_state)

        print("Total grasp planning time: {} secs.".format(str(time.time() - grasp_planning_start_time)))

        grasps_and_predictions = []
        for g, q in zip(grasps, q_values):
            grasps_and_predictions.append( [g,q] )

        sorted_grasps_and_predictions = sorted(grasps_and_predictions, key=lambda x: x[1], reverse=True)

        selected_grasps_and_predictions = sorted_grasps_and_predictions[:n_candidates]

        return selected_grasps_and_predictions
