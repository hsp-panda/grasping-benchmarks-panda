# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import os
import numpy as np
from grasping_benchmarks.base import transformations


class Grasp6D(object):
    """6D cartesian grasp

    Attributes
    ----------
    position : (`numpy.ndarray` of float): 3-entry position vector wrt camera frame
    rotation (`numpy.ndarray` of float):3x3 rotation matrix wrt camera frame
    width : Distance between the fingers in meters.
    score: prediction score of the grasp pose
    ref_frame: frame of reference for camera that the grasp corresponds to.
    quaternion: rotation expressed as quaternion
    """

    def __init__(self,
                 position=np.zeros(3),
                 rotation=np.eye(3),
                 width=0.0,
                 score=0.0,
                 ref_frame="camera"):

        self._position = position
        self._rotation = rotation

        self._check_valid_position(self._position)
        self._check_valid_rotation(self._rotation)

        self.width = width
        self.ref_frame = ref_frame
        self.score = score

        self._quaternion = transformations.matrix_to_quaternion(self._rotation)


    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        # Convert lists and tuples
        if type(rotation) in (list, tuple):
            rotation = np.array(rotation).astype(np.float32)

        self._check_valid_rotation(rotation)
        self._rotation = rotation * 1.

        self._quaternion = transformations.matrix_to_quaternion(rotation)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        # Convert list to position array
        if type(position) in (list, tuple) and len(position) == 3:
            position = np.array([t for t in position]).astype(np.float32)

        self._check_valid_position(position)
        self._position = position.squeeze() * 1.

    @property
    def quaternion(self):
        return self._quaternion

    @quaternion.setter
    def quaternion(self, quat):
        # Convert quaternions
        if len(quat) != 4 or np.abs(np.linalg.norm(quat) - 1.0) > 1e-3:
                raise ValueError('Invalid quaternion')

        self._quaternion = np.array([q for q in quat])
        rotation = transformations.quaternion_to_matrix(q)

        self._check_valid_rotation(rotation)
        self._rotation = rotation * 1.


    def _check_valid_rotation(self, rotation):
        """Checks that the given rotation matrix is valid.
        """
        if not isinstance(rotation, np.ndarray) or not np.issubdtype(rotation.dtype, np.number):
            raise ValueError('Rotation must be specified as numeric numpy array')

        if len(rotation.shape) != 2 or rotation.shape[0] != 3 or rotation.shape[1] != 3:
            raise ValueError('Rotation must be specified as a 3x3 ndarray')

        if np.abs(np.linalg.det(rotation) - 1.0) > 1e-3:
            raise ValueError('Illegal rotation. Must have determinant == 1.0')

    def _check_valid_position(self, position):
        """Checks that the position vector is valid.
        """
        if not isinstance(position, np.ndarray) or not np.issubdtype(position.dtype, np.number):
            raise ValueError('Position must be specified as numeric numpy array')

        pos = position.squeeze()
        if len(pos.shape) != 1 or pos.shape[0] != 3:
            raise ValueError('position must be specified as a 3-vector, 3x1 ndarray, or 1x3 ndarray')
