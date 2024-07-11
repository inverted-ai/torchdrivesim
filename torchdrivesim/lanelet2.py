"""
All utilities for working with maps in Lanelet2 format.
This module imports correctly and exports all the usual signatures even
when the lanelet2 package is not installed, but in that case all calls
to its functions raise the Lanelet2NotFound exception.
"""
import os
from random import random
from typing import Any, Tuple, List, Optional

import numpy as np
import scipy.spatial
from scipy.sparse import lil_matrix
import torch
from torch import Tensor

from torchdrivesim.mesh import BaseMesh, logger, BirdviewMesh, rendering_mesh

try:
    import lanelet2
    is_available = True
    LaneletMap = lanelet2.core.LaneletMap
except ImportError:
    lanelet2 = None
    is_available = False
    LaneletMap = Any


class Lanelet2NotFound(ImportError):
    """
    Python bindings for Lanelet2 are not installed.
    """
    pass


class LaneletError(RuntimeError):
    """
    Some function related to Lanelet2 failed.
    """
    pass


def load_lanelet_map(map_path: str, origin: Tuple[float, float] = (0, 0)) -> LaneletMap:
    """
    Load a Lanelet2 map from an OSM file on disk.

    Args:
        map_path: local path to OSM file containing the map
        origin: latitude and longitude of the origin to use with UTM projector
    Raises:
        Lanelet2NotFound: if lanelet2 package is not available
        FileNotFoundError: if specified file doesn't exist
    """
    if not is_available:
        raise Lanelet2NotFound()
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(*origin))
    lanelet_map = lanelet2.io.load(map_path, projector)
    return lanelet_map


def find_lanelet_directions(lanelet_map: LaneletMap, x: float, y: float, tags_to_exclude: Optional[List[str]] = None,
                            lanelet_dist_tolerance: float = 1.0) -> List[float]:
    """
    For a given point, find local orientations of all lanelets that contain it.

    Args:
        lanelet_map: the map with a lanelet layer
        x: first coordinate of the point in the map's coordinate frame
        y: second coordinate of the point in the map's coordinate frame
        tags_to_exclude: lanelets tagged with any of those tags will be omitted as if they didn't exist
        lanelet_dist_tolerance: how far out of a lanelet a car can be for the lanelet to still be containing it
    Returns:
        for each lanelet, the angle representing its local orientation in radians
    Raises:
        Lanelet2NotFound: if lanelet2 package is not available
    """
    if not is_available:
        raise Lanelet2NotFound()
    if tags_to_exclude is None:
        tags_to_exclude = []
    location = lanelet2.core.BasicPoint2d(float(x), float(y))
    location3d = lanelet2.core.BasicPoint3d(float(x), float(y), 0)
    all_selected_lanelets = lanelet2.geometry.findWithin2d(lanelet_map.laneletLayer, location, lanelet_dist_tolerance)
    directions = []
    for distance, lanelet in all_selected_lanelets:
        centerline = lanelet.centerline
        if len(centerline) < 2:
            continue
        if any([lanelet_attr in lanelet.attributes for lanelet_attr in tags_to_exclude]):
            directions = []
            break
        direction = find_direction(centerline, location3d)
        directions.append(direction)
    return directions


def find_direction(linestring, location3d) -> float:
    """
    For a given linestring and a point near it, finds the local orientation of the linestring.

    Returns:
        the linestring's local orientation in radians
    Raises:
        Lanelet2NotFound: if lanelet2 package is not available
        LaneletError: when the method fails, usually because the linestring has a weird shape
    """
    if not is_available:
        raise Lanelet2NotFound()
    projected_reference = lanelet2.geometry.project(linestring, location3d)
    first, second = float("inf"), float("inf")
    closest_point_idx, second_closest_point_idx = 0, 0

    for i, point in enumerate(linestring):
        point_dist = lanelet2.geometry.distance(projected_reference, point)
        if point_dist < first:
            second = first
            first = point_dist
            second_closest_point_idx = closest_point_idx
            closest_point_idx = i
        elif point_dist < second:
            second = point_dist
            second_closest_point_idx = i

    if not abs(closest_point_idx - second_closest_point_idx) == 1:
        raise LaneletError('Failed to find direction of the linestring at a given point')

    if closest_point_idx > second_closest_point_idx:
        point_a, point_b = linestring[second_closest_point_idx], linestring[closest_point_idx]
    else:
        point_b, point_a = linestring[second_closest_point_idx], linestring[closest_point_idx]
    direction = np.arctan2(point_b.y - point_a.y, point_b.x - point_a.x)

    return direction


def pick_random_point_and_orientation(lanelet_map: LaneletMap) -> Tuple[float, float, float]:
    """
    Picks a point on the map by randomly selecting a lanelet and then randomly selecting a point
    along its centerline, both using `random` package.

    Returns:
        tuple (x, y, orientation), using the map's coordinate frame and radians
    Raises:
        Lanelet2NotFound: if lanelet2 package is not available
    """
    if not is_available:
        raise Lanelet2NotFound()
    spawn_lanelet = random.choice(list(lanelet_map.laneletLayer))
    centerline = spawn_lanelet.centerline
    lanelet_length = lanelet2.geometry.length(centerline)
    distance = random.uniform(0, lanelet_length)
    selected_point = lanelet2.geometry.interpolatedPointAtDistance(centerline, distance)
    following_point = lanelet2.geometry.interpolatedPointAtDistance(centerline, min(distance + 1, lanelet_length))
    orientation = np.arctan2(following_point.y - selected_point.y, following_point.x - selected_point.x)
    return selected_point.x, selected_point.y, orientation


def road_mesh_from_lanelet_map(lanelet_map: LaneletMap, lanelets: Optional[List[int]] = None) -> BaseMesh:
    """
    Creates a road mesh by triangulating all lanelets in a given map.

    Args:
        lanelet_map: map to triangulate
        lanelets: if specified, only use lanelets with those ids
    """
    # Each point in the lanelet map becomes a vertex of the road mesh
    n_points = len(lanelet_map.pointLayer)
    vertices = np.ndarray(shape=(n_points, 2), dtype=np.float32)
    point_idx = dict()
    i = 0
    for p in lanelet_map.pointLayer:
        point_idx[p.id] = i
        vertices[i] = [p.x, p.y]
        i += 1
    verts = torch.from_numpy(vertices)

    # Lanelets are triangulated into faces
    # Triangulation may not be correct for very bendy lanelets
    # with unbalanced spacing between the left and the right boundary
    lanelet_faces = []
    for l in lanelet_map.laneletLayer:
        if lanelets is not None and l.id not in lanelets:
            continue
        lb = l.leftBound
        rb = l.rightBound
        n_faces = len(lb) + len(rb) - 2
        if n_faces < 1:
            # TODO: CARLA maps tend to triger this warning unnecessarily many times
            logger.debug("Fewer than 3 points forming lanelet boundary, will not render")
            continue
        faces = np.ndarray(shape=(n_faces, 3), dtype=np.int_)
        i, j = 0, 0
        while i + j < n_faces:
            if i < len(lb) - 1:
                faces[i + j] = [point_idx[p.id] for p in [lb[i], rb[j], lb[i + 1]]]
                i += 1
            if j < len(rb) - 1:
                faces[i + j] = [point_idx[p.id] for p in [lb[i], rb[j], rb[j + 1]]]
                j += 1
        faces = torch.from_numpy(faces)
        lanelet_faces.append(faces)
    faces = torch.cat(lanelet_faces, dim=0)
    return BaseMesh(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0))


def line_segments_to_mesh(points: Tensor, line_width: float = 0.3, eps: float = 1e-6) -> BaseMesh:
    """
    Converts a given collection of line segments to a mesh visualizing them.

    Args:
        points: BxNx2x2 tensor specifying N line segments, each consisting of a pair of points
        line_width: width of the line in meters
        eps: small value to add for avoiding division by zero
    Return:
        mesh visualizing the line, with 6*N verts and 4*N faces
    """
    batch_size = points.shape[0]
    # TODO: Since I'm not actually using multiple batches here I don't bother with masking between meshes
    # but maybe I should add support for optionally passing a mask and appropriately compute the norm with padding.
    d = points[:, :, 1] - points[:, :, 0]
    d_hat = d / (torch.norm(d, p=2, dim=2, keepdim=True) + eps)
    d_perp = torch.stack([-d_hat[:, :, 1], d_hat[:, :, 0]], dim=2).unsqueeze(2)

    verts = torch.cat([
        points + d_perp * line_width,
        points,
        points - d_perp * line_width,
    ], dim=2).reshape(batch_size, -1, 2)

    faces = torch.tensor([[[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]]], dtype=torch.int32, device=points.device)
    faces = faces.repeat_interleave(points.shape[1], dim=1).repeat_interleave(batch_size, dim=0).reshape(batch_size, -1, 3)
    offset = torch.arange(start=0, end=faces.shape[1]//4, step=1, device=faces.device, dtype=faces.dtype)*6
    offset = offset.unsqueeze(0).repeat_interleave(batch_size, dim=0)
    offset = torch.stack([offset, offset, offset, offset], dim=2).view(batch_size, -1, 1)
    faces = faces + offset
    return BaseMesh(verts=verts, faces=faces)


def lanelet_map_to_lane_mesh(lanelet_map: LaneletMap, left_handed: bool = False, batch_size: int = 50000,
                             left_right_marking_join_threshold: float = 0.1, lanelets: Optional[List[int]] = None) -> BirdviewMesh:
    """
    Creates a lane marking mesh from a given map.

    Args:
        lanelet_map: map to use
        left_handed: whether the map's coordinate system is left-handed (flips the left and right boundary designations)
        batch_size: controls the amount of points processed in parallel
        left_right_marking_join_threshold: if left and right markings are this close, they will be treated as joint
    """
    # Each point in the lanelet map becomes a vertex of the road mesh
    n_points = len(lanelet_map.pointLayer)
    vertices = np.ndarray(shape=(n_points, 2), dtype=np.float32)
    point_idx = dict()
    i = 0
    for p in lanelet_map.pointLayer:
        point_idx[p.id] = i
        vertices[i] = [p.x, p.y]
        i += 1
    left_segments_t = set()
    right_segments_t = set()
    for j, l in enumerate(lanelet_map.laneletLayer):
        if lanelets is not None and l.id not in lanelets:
            continue
        lb = l.leftBound
        rb = l.rightBound

        for i in range(len(rb) - 1):
            segment = tuple(sorted([rb[i].id, rb[i + 1].id]))
            right_segments_t.add(segment)

        for i in range(len(lb) - 1):
            segment = tuple(sorted([lb[i].id, lb[i + 1].id]))
            left_segments_t.add(segment)
    left_points_t = []
    for segment in left_segments_t:
        p1 = vertices[point_idx[segment[0]]]
        p2 = vertices[point_idx[segment[1]]]
        left_points_t.append(np.array([p1, p2], dtype=np.float32))
    left_points_t = np.stack(left_points_t, axis=0)
    right_points_t = []
    for segment in right_segments_t:
        p1 = vertices[point_idx[segment[0]]]
        p2 = vertices[point_idx[segment[1]]]
        right_points_t.append(np.array([p1, p2], dtype=np.float32))
    right_points_t = np.stack(right_points_t, axis=0)
    def calc_distance_sparse(vec_1, vec_2):
        res = lil_matrix((vec_1.shape[0], vec_2.shape[0]), dtype = np.int8)
        for i in range(0, vec_1.shape[0], batch_size):
            for j in range(0, vec_2.shape[0], batch_size):
                res[i:i+batch_size, j:j+batch_size] = scipy.spatial.distance.cdist(\
                    vec_1[i:i+batch_size], vec_2[j:j+batch_size]) < left_right_marking_join_threshold
        return res.tocsr()
    p00 = calc_distance_sparse(left_points_t[:,0], right_points_t[:,0])
    p11 = calc_distance_sparse(left_points_t[:,1], right_points_t[:,1])
    p01 = calc_distance_sparse(left_points_t[:,0], right_points_t[:,1])
    p10 = calc_distance_sparse(left_points_t[:,1], right_points_t[:,0])
    joint_indexes = p00.multiply(p11) + p01.multiply(p10)
    left_common_indexes_binary = (np.asarray(joint_indexes.sum(axis=1)) > 0).astype(np.float32).flatten()
    right_common_indexes_binary = (np.asarray(joint_indexes.sum(axis=0)) > 0).astype(np.float32).flatten()
    left_indexes = (1 - left_common_indexes_binary).nonzero()[0]
    right_indexes = (1 - right_common_indexes_binary).nonzero()[0]
    left_common_indexes = left_common_indexes_binary.nonzero()[0]
    right_common_indexes = right_common_indexes_binary.nonzero()[0]
    left_points = left_points_t[left_indexes]
    right_points = right_points_t[right_indexes]
    joint_points = left_points_t[left_common_indexes]  # Keep common vertices from the left

    if left_handed:
        left_points, right_points = right_points, left_points

    left_points = torch.tensor(left_points).unsqueeze(0)
    right_points = torch.tensor(right_points).unsqueeze(0)
    if joint_points is not None:
        joint_points = torch.tensor(joint_points).unsqueeze(0)
    batch_size = left_points.shape[0]
    if joint_points is not None and joint_points.shape[1] > 0:
        joint_mesh = rendering_mesh(
            line_segments_to_mesh(joint_points, line_width=0.275),
            category='joint_lane'
        )
    else:
        joint_mesh = BirdviewMesh.empty(dim=2, batch_size=batch_size).to(left_points.device)
    left_mesh = rendering_mesh(
        line_segments_to_mesh(left_points, line_width=0.275), category='left_lane'
    )
    right_mesh = rendering_mesh(
        line_segments_to_mesh(right_points, line_width=0.275), category='right_lane'
    )
    lane_mesh = BirdviewMesh.concat([joint_mesh, left_mesh, right_mesh])
    return lane_mesh
