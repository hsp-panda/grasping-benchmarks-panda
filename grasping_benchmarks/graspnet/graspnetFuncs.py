from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
from utils import utils
from data import DataLoader
import ipdb
from types import SimpleNamespace

import sys


from threading import Thread
from time import sleep




def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument('--input_file_name', type=str, default='cheezit')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=1024)
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    return parser

    
def grasp_estimator_args(graspnet_cfg):
    graspnet_args = SimpleNamespace(grasp_sampler_folder = graspnet_cfg['grasp_sampler_folder'], 
                                grasp_evaluator_folder = graspnet_cfg['grasp_evaluator_folder'],
                                refinement_method=graspnet_cfg['refinement_method'],
                                refine_steps=graspnet_cfg['refine_steps'],
                                npy_folder=graspnet_cfg['npy_folder'],
                                input_file_name=graspnet_cfg['input_file_name'],
                                threshold=graspnet_cfg['threshold'],
                                choose_fn=graspnet_cfg['choose_fn'],
                                target_pc_size=graspnet_cfg['target_pc_size'],
                                num_grasp_samples=graspnet_cfg['num_grasp_samples'],
                                generate_dense_grasps=graspnet_cfg['generate_dense_grasps'],
                                batch_size=graspnet_cfg['batch_size'],
                                train_data=graspnet_cfg['train_data'])

    return graspnet_args


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X



def graspnetfuncs(graspnet_cfg, data):
    ipdb.set_trace()

    graspnet_args = grasp_estimator_args(graspnet_cfg)

    grasp_sampler_args = utils.read_checkpoint_args(graspnet_args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(graspnet_args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args, grasp_evaluator_args, graspnet_args)

    # Depending on your numpy version you may need to change allow_pickle
    # from True to False.
    ipdb.set_trace()
    depth = data['depth']
    image = data['image']
    K = data['intrinsics_matrix']
    # Removing points that are farther than 1 meter or missing depth
    # values.
    #depth[depth == 0 or depth > 1] = np.nan

    np.nan_to_num(depth, copy=False)
    mask = np.where(np.logical_or(depth == 0, depth > 1))
    depth[mask] = np.nan
    pc, selection = backproject(depth,
                                K,
                                return_finite_depth=True,
                                return_selection=True)
    pc_colors = image.copy()
    pc_colors = np.reshape(pc_colors, [-1, 3])
    pc_colors = pc_colors[selection, :]

    # Smoothed pc comes from averaging the depth for 10 frames and removing
    # the pixels with jittery depth between those 10 frames.
    object_pc = data['smoothed_object_pc']
    ipdb.set_trace()
    generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
        object_pc)    

   
    # simple_draw_scene(
    #     pc,
    #     pc_color=pc_colors,
    #     grasps=generated_grasps,
    #     grasp_scores=generated_scores,
    # )

    return generated_grasps, generated_scores






