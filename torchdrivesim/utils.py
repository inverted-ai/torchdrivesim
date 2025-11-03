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


def transform(points: Tensor, pose: Tensor) -> Tensor:
    """
    Given points relative to a pose, produce absolute positions of the points.
    There can be zero or more batch dimensions.

    Args:
        points: BxNx2 tensor
        pose: Bx3 tensor of position (x,y) and orientation (yaw angle in radians)

    Returns:
        Bx2 tensor of absolute positions
    """
    xy = pose[..., :2].unsqueeze(-2).expand_as(points)
    psi = pose[..., 2:3].unsqueeze(-2).expand_as(points[..., :1])
    return rotate(points, psi) + xy


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


def line_circle_intersection(p1: torch.Tensor, p2: torch.Tensor,
                           circle_center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """
    Determine intersections between batched line segments and circles in 2D.

    Args:
        p1 (torch.Tensor): Start points of line segments [..., 2]
        p2 (torch.Tensor): End points of line segments [..., 2]
        circle_center (torch.Tensor): Circle centers [..., 2]
        radius (torch.Tensor): Circle radii [..., 1]

    Returns:
        torch.Tensor: Boolean tensor indicating intersections [..., 1]
    """
    # Vector from p1 to p2
    d = p2 - p1  # [..., 2]

    # Vector from p1 to circle center
    f = p1 - circle_center  # [..., 2]

    # Coefficients of quadratic equation at^2 + bt + c = 0
    a = torch.sum(d * d, dim=-1)  # [...]
    b = 2 * torch.sum(f * d, dim=-1)  # [...]
    c = torch.sum(f * f, dim=-1) - (radius[..., 0] * radius[..., 0])  # [...]

    # Discriminant
    discriminant = b * b - 4 * a * c  # [...]

    # Check if discriminant is non-negative (potential intersection)
    has_intersection = discriminant >= 0  # [...]

    # Calculate intersection parameters
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0))
    # Avoid division by zero for degenerate lines (a = 0)
    a_safe = torch.where(torch.abs(a) < 1e-8, torch.ones_like(a) * 1e-8, a)

    t1 = (-b - sqrt_discriminant) / (2 * a_safe)  # [...]
    t2 = (-b + sqrt_discriminant) / (2 * a_safe)  # [...]

    # Check if line segment intersects circle
    # Intersection occurs if interval [0,1] overlaps with interval [min(t1,t2), max(t1,t2)]
    t_min = torch.min(t1, t2)
    t_max = torch.max(t1, t2)
    line_segment_intersection = (t_min <= 1) & (t_max >= 0)

    # Combine discriminant check with line segment constraint
    result = has_intersection & line_segment_intersection

    return result[..., None]  # Add dimension to match expected output shape
