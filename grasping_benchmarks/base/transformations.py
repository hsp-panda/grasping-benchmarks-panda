# Copyright (C) 2019 Istituto Italiano di Tecnologia (IIT)
# This software may be modified and distributed under the terms of the
# LGPL-2.1+ license. See the accompanying LICENSE file for details.

import numpy as np
import math as m
import warnings

import struct


def axis_angle_to_quaternion(vec_aa):
    """ Transform rotation from axis-angle to quaternion representation
    vec_aa: x, y ,z, a
    """
    qx = vec_aa[0] * m.sin(vec_aa[3] / 2)
    qy = vec_aa[1] * m.sin(vec_aa[3] / 2)
    qz = vec_aa[2] * m.sin(vec_aa[3] / 2)
    qw = m.cos(vec_aa[3] / 2)

    return [qx, qy, qz, qw]

def quaternion_to_axis_angle(quat):
    """ Transform rotation from quaternion to axis-angle representation
    quat: x, y ,z, w
    """
    angle = 2 * m.acos(quat[3])
    x = quat[0] / m.sqrt(1 - quat[3] * quat[3])
    y = quat[1] / m.sqrt(1 - quat[3] * quat[3])
    z = quat[2] / m.sqrt(1 - quat[3] * quat[3])

    return [x, y, z, angle]

def quaternion_to_matrix(quat):
    """ Transform rotation from quaternion to 3x3 matrix representation
    quat: x, y ,z, w
    """

    q = np.array(quat[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    q *= m.sqrt(2.0 / nq)
    q = np.outer(q, q)

    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])
        ), dtype=np.float64)

def matrix_to_quaternion(matrix):
    """Transform rotation from 3x3 matrix to quaternion representation
    matrix: 3x3
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2

        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0

        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1

        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]

        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]

    q *= 0.5 / m.sqrt(t * M[3, 3])
    return q

# TODO
def quaternion_to_eigen(quat):
    raise NotImplementedError

# TODO
def eigen_to_quat(roll, pitch, yaw):
    raise NotImplementedError

# TODO
def eigen_to_matrix(roll, pitch, yaw):
    raise NotImplementedError

# TODO
def matrix_to_eigen(matrix):
    raise NotImplementedError

def quat_multiplication(a: np.ndarray, b: np.ndarray):

    if not a.shape == b.shape and a.shape == 4:
        raise AssertionError("quat_distance(): wrong shape of points")
    elif not (np.linalg.norm(a) == 1.0 and np.linalg.norm(b) == 1.0):
        warnings.warn("quat_distance(): vector(s) without unitary norm {} , {}".format(np.linalg.norm(a), np.linalg.norm(b)))

    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]

    x12 = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y12 = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z12 = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w12 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x12, y12, z12, w12])

def quat_conjugate(a: np.ndarray):
    return np.array([-a[0], -a[1], -a[2], a[3]])


def quat_inverse(a: np.ndarray):
    conj = quat_conjugate(a)
    norm_a = np.linalg.norm(a)
    inv = np.divide(conj, np.linalg.norm(a)**norm_a)

    if not np.linalg.norm(inv) == 1.0:
        warnings.warn("quat_inverse(): computed vector has not unitary norm {}".format(np.linalg.norm(a)))

    return inv

def sph_coord(x: float, y: float, z: float):
    ro = m.sqrt(x*x + y*y + z*z)
    theta = m.acos(z/ro)
    phi = m.atan2(y,x)
    return [ro, theta, phi]
