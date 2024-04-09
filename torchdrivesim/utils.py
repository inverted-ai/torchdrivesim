"""
Miscellaneous utilities, including for geometric operations on tensors.
"""
import numpy as np
import collections
from functools import reduce
from typing import Tuple, List, Dict

import torch
from torch import Tensor

Resolution = collections.namedtuple('Resolution', ['width', 'height'])


def isin(x: Tensor, y: Tensor) -> Tensor:
    """
    Checks whether elements of tensor x are contained in tensor y.
    This function is built-in in torch >= 1.10
    and will be removed from here in the future.

    Args:
        x: any tensor
        y: a one-dimensional tensor
    Returns:
        a boolean tensor with the same shape as x
    """
    assert len(y.shape) == 1
    return (x[..., None] == y).any(-1)


def normalize_angle(angle):
    """
    Normalize to <-pi, pi) range by shifting by a multiple of 2*pi.
    Works with floats, numpy arrays, and torch tensors.
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def rotation_matrix(theta: Tensor) -> Tensor:
    """
    Counterclockwise rotation matrix in 2D.

    Args:
        theta: tensor of shape Sx1 with rotation angle in radians
    Returns:
        Sx2x2 tensor with the rotation matrix.
    """
    rot_mat = torch.stack([
        torch.cat([torch.cos(theta), - torch.sin(theta)], dim=-1),
        torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
    ], dim=-2)
    return rot_mat


def rotate(v: Tensor, angle: Tensor) -> Tensor:
    """
    Rotate the vector counterclockwise (from x towards y).
    Works correctly in batch mode.

    Args:
        v: tensor of shape Sx2 representing points
        angle: tensor of shape Sx1 representing rotation angle
    Returns:
        Sx2 tensor of rotated points
    """
    rot_mat = rotation_matrix(angle)
    rotated = torch.matmul(rot_mat, v.unsqueeze(-1)).squeeze(-1)
    return rotated


def relative(origin_xy: Tensor, origin_psi: Tensor, target_xy: Tensor, target_psi: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes position and orientation of the target relative to origin.
    Points are represented as Sx2 tensors of coordinates and Sx1 tensors of orientations in radians.
    """
    rel_xy = rotate(target_xy - origin_xy, - origin_psi)
    rel_psi = normalize_angle(target_psi - origin_psi)
    return rel_xy, rel_psi


def is_inside_polygon(point: Tensor, polygon: Tensor) -> Tensor:
    """
    Checks whether a given point is inside a given convex polygon.
    B and P can be zero or more batch dimensions, the former being batch and the latter points.

    Args:
        point: BxPx2 tensor of x-y coordinates
        polygon: BxNx2 tensor of points specifying a convex polygon in either clockwise or counter-clockwise fashion.
    Returns:
        boolean tensor of shape BxP indicating whether the point is inside the polygon
    """
    batch_dims = len(polygon.shape) - 2
    assert batch_dims >= 0
    assert polygon.shape[:batch_dims] == point.shape[:batch_dims]
    for _ in point.shape[batch_dims:-1]:
        polygon = polygon.unsqueeze(-3)
    edges = torch.stack([polygon, polygon.roll(-1, dims=-2)], dim=-2)
    a = edges[..., 1, 1] - edges[..., 0, 1]
    b = edges[..., 0, 0] - edges[..., 1, 0]
    c = - a * edges[..., 0, 0] - b * edges[..., 0, 1]
    is_right = a * point[..., None, 0] + b * point[..., None, 1] + c >= 0
    all_right = torch.all(is_right, dim=-1)
    all_left = torch.all(is_right.logical_not(), dim=-1)
    return torch.logical_or(all_right, all_left)


def merge_dicts(ds: List[Dict]) -> Dict:
    """
    Merges a sequence of dictionaries, giving preference to entries earlier in the sequence.
    """
    def f(x, y):
        x.update(y)
        return x
    return reduce(f, ds, dict())


def assert_equal(x, y):
    assert x == y
