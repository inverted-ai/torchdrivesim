import importlib
from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import torch
from shapely.geometry import Polygon
from torch import Tensor, relu, cosine_similarity
from torch.nn import functional as F

import torchdrivesim
from torchdrivesim._iou_utils import box2corners_th, iou_differentiable_fast, iou_non_differentiable
from torchdrivesim.lanelet2 import LaneletMap, find_lanelet_directions, LaneletError
from torchdrivesim.mesh import BaseMesh
from torchdrivesim import assert_pytorch3d_available
from torchdrivesim.rendering.pytorch3d import Pytorch3DNotFound
from torchdrivesim.utils import normalize_angle

logger = logging.getLogger(__name__)

LANELET_TAGS_TO_EXCLUDE = ['parking']


def point_mesh_face_distance(meshes: "pytorch3d.structures.Meshes", pcls: "pytorch3d.structures.Pointclouds",
                             reduction: str = 'sum', weighted: bool = False, threshold: float = 0) -> Tensor:
    """
    Computes the distance between a pointcloud and a mesh, defined as the L2 distance
    from each point to the closest face in the mesh, reduced across the points in the cloud.
    Runs in batch mode for both mesh and pointcloud.

    Args:
        meshes: a batch of B meshes
        pcls: a batch of B pointcouds
        reduction: 'none' | 'sum' | 'mean' | 'min' | 'max'
        weighted: weight by the inverse of number of points in the pointcloud
        threshold: reduce result by this amount and clip at zero from below
    Returns:
        BxP tensor if reduction is 'none', else Bx1 tensor
    """

    assert_pytorch3d_available()
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    if len(meshes) != len(pcls):
        raise ValueError(f"The batch has {len(meshes)} but {len(pcls)} pointclouds")
    batch_size = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    if weighted:
        # weight by the inverse of number of points
        point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
        num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
        weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
        weights_p = 1.0 / weights_p.float()
        point_to_face = point_to_face * weights_p
    point_dist = point_to_face.view(batch_size, -1)
    point_dist = torch.nan_to_num(point_dist, nan=0.0)
    point_dist = F.threshold(point_dist, threshold, 0)
    if reduction == 'sum':
        point_dist = point_dist.sum(dim=-1, keepdim=True)
    elif reduction == 'mean':
        point_dist = point_dist.mean(dim=-1, keepdim=True)
    elif reduction == 'min':
        point_dist = point_dist.min(dim=-1, keepdim=True)[0]
    elif reduction == 'max':
        point_dist = point_dist.max(dim=-1, keepdim=True)[0]

    return point_dist


def point_to_mesh_distance_pt(points: torch.Tensor, tris: torch.Tensor, threshold: float = 0):
    """
    Computes the distance between points and mesh triangles, defined as the L2 distance
    from each point to the closest face in the mesh. This function only uses native Pytorch operations
    and is an alternative version of the function `point_mesh_face_distance` that relies on Pytorch3D.

    Args:
        points: Bx3 tensor of points
        tris: BxFx3x3 tensor of mesh triangles
        threshold: reduce result by this amount and clip at zero from below
    Returns:
        Bx1 tensor
    """

    p = points.unsqueeze(1)
    v0, v1, v2 = tris.unbind(dim=-2)
    cross = torch.cross(v2 - v0, v1 - v0, dim=-1)
    norm_normal = cross.norm(dim=-1, keepdim=True)
    normal = cross / (norm_normal + 1e-8)

    t = ((v0 - p)[..., None, :] @ normal[..., :, None]).squeeze(-1)
    p0 = p + t * normal

    def area_of_triangle(v0, v1, v2):
        p0 = v1 - v0
        p1 = v2 - v0

        # compute the hypotenus of the scross product (p0 x p1)
        dd = torch.hypot(
          p0[..., 1] * p1[..., 2] - p0[..., 2] * p1[..., 1],
          torch.hypot(p0[..., 2] * p1[..., 0] - p0[..., 0] * p1[..., 2],
                      p0[..., 0] * p1[..., 1] - p0[..., 1] * p1[..., 0]))
        return dd / 2.0

    def bary_centric_coords_3d(p, v0, v1, v2, eps=1e-8):
        p0 = v1 - v0
        p1 = v2 - v0
        p2 = p - v0

        d00 = (p0[..., None, :] @ p0[..., :, None]).squeeze(-1)
        d01 = (p0[..., None, :] @ p1[..., :, None]).squeeze(-1)
        d11 = (p1[..., None, :] @ p1[..., :, None]).squeeze(-1)
        d20 = (p2[..., None, :] @ p0[..., :, None]).squeeze(-1)
        d21 = (p2[..., None, :] @ p1[..., :, None]).squeeze(-1)

        denom = d00 * d11 - d01 * d01 + eps
        w1 = (d11 * d20 - d01 * d21) / denom
        w2 = (d00 * d21 - d01 * d20) / denom
        w0 = 1.0 - w1 - w2
        return w0, w1, w2

    def is_inside_triangle(p, v0, v1, v2, min_triangle_area=5e-3):
        w0, w1, w2 = bary_centric_coords_3d(p, v0, v1, v2)
        x_in = (0.0 <= w0) & (w0 <= 1.0)
        y_in = (0.0 <= w1) & (w1 <= 1.0)
        z_in = (0.0 <= w2) & (w2 <= 1.0)
        inside = x_in & y_in & z_in

        invalid_triangles = (area_of_triangle(v0, v1, v2) < min_triangle_area).unsqueeze(-1)
        return inside & invalid_triangles.logical_not()

    def point_line_distance(p, v0, v1):
        v1v0 = v1 - v0
        l2 = (v1v0[..., None, :] @ v1v0[..., :, None]).squeeze(-1)

        t = (v1v0[..., None, :] @ (p - v0)[..., :, None]).squeeze(-1) / (l2 + 1e-8)
        tt = torch.clamp(t, min=0, max=1)
        p_proj = v0 + tt * v1v0
        dist = ((p - p_proj)[..., None, :] @ (p - p_proj)[..., :, None]).squeeze(-1)

        small_dist = (l2 <= 1e-8).to(l2.dtype)
        if small_dist.any():
            dist = dist * (1 - small_dist) + ((p - v1)[..., None, :] @ (p - v1)[..., :, None]).squeeze(-1) * small_dist
        return dist

    e01 = point_line_distance(p, v0, v1)
    e02 = point_line_distance(p, v0, v2)
    e12 = point_line_distance(p, v1, v2)

    dist = torch.minimum(torch.minimum(e01, e02), e12)

    inside_triangle = is_inside_triangle(p0, v0, v1, v2)
    condition = (inside_triangle & (norm_normal > 1e-8)).to(dist.dtype)

    dist, _ = ((t*t)*condition + dist*(1-condition)).min(dim=-2)
    dist = torch.nan_to_num(dist, nan=0.0)
    dist = F.threshold(dist, threshold, 0)
    return dist


def offroad_infraction_loss(agent_states: Tensor, lenwid: Tensor,
                            driving_surface_mesh: Union["pytorch3d.structures.Meshes", BaseMesh],
                            threshold: float = 0, use_pytorch3d: Optional[bool] = None) -> Tensor:
    """
    Calculates off-road infraction loss, defined as the sum of thresholded distances
    from agent corners to the driving surface mesh.

    Args:
        agent_states: BxAx4 tensor of agent states x,y,orientation,speed
        lenwid: BxAx2 tensor providing length and width of each agent
        driving_surface_mesh: a batch of B meshes defining driving surface
        threshold: if given, the distance of each corner is thresholded using this value
        use_pytorch3d: whether to use the optimized and differentiable pytorch3d code, or the
            simpler pure pytorch implementation - defaults to true iff pytorch3d is installed
    Returns:
        BxA tensor of offroad losses for each agent
    """

    if use_pytorch3d is None:
        use_pytorch3d = torchdrivesim.rendering.pytorch3d.is_available
    batch_size, num_agents = agent_states.shape[:2]
    if num_agents == 0 or driving_surface_mesh.faces_count == 0:
        return torch.zeros_like(agent_states[..., 0])
    if len(lenwid.shape) == 2:
        lenwid = lenwid.unsqueeze(-2).expand((lenwid.shape[0], num_agents, lenwid.shape[1]))
    predicted_rectangles = torch.cat([
        agent_states[..., :2],
        lenwid,
        agent_states[..., 2:3],
    ], dim=-1)
    ego_verts = box2corners_th(predicted_rectangles)
    ego_verts = F.pad(ego_verts, (0,1))
    if use_pytorch3d:
        assert_pytorch3d_available()
        import pytorch3d
        ego_verts = ego_verts.view(-1, *ego_verts.shape[2:])  # B*A x 4 x 3
        ego_pointclouds = pytorch3d.structures.Pointclouds(ego_verts)
        if isinstance(driving_surface_mesh, BaseMesh):
            driving_surface_mesh = driving_surface_mesh.pytorch3d(include_textures=False)
        driving_surface_mesh_extended = driving_surface_mesh.extend(num_agents)
        offroad_loss = point_mesh_face_distance(driving_surface_mesh_extended, ego_pointclouds, threshold=threshold)
        offroad_loss = offroad_loss.view(batch_size, num_agents)
    else:
        ego_verts = ego_verts.view(-1, *ego_verts.shape[3:])  # B*A*4 x 3
        driving_surface_mesh_extended = driving_surface_mesh.expand(num_agents*4)
        mesh_verts = F.pad(driving_surface_mesh_extended.verts[..., :2], (0,1)).reshape(-1, 3)
        mesh_faces = driving_surface_mesh_extended.faces.clone()
        mesh_faces += driving_surface_mesh_extended.verts_count * torch.arange(driving_surface_mesh_extended.batch_size,
                                                            dtype=mesh_faces.dtype, device=mesh_faces.device)[:, None, None]
        mesh_faces = mesh_faces.reshape(-1, 3)
        mesh_tris = mesh_verts[mesh_faces].reshape(-1, driving_surface_mesh_extended.faces_count, 3, 3)
        offroad_loss = point_to_mesh_distance_pt(ego_verts, mesh_tris, threshold=threshold)
        offroad_loss = offroad_loss.reshape(batch_size, num_agents, 4).sum(dim=-1)
    return offroad_loss


def lanelet_orientation_loss(lanelet_maps: List[Optional[LaneletMap]], agents_state: Tensor,
                             recenter_offset: Optional[Tensor] = None,
                             direction_angle_threshold: float = np.pi / 2,
                             lanelet_dist_tolerance: float = 1.0) -> Tensor:
    """
    Calculate a loss value of the agent orientation regarding to the direction of the lanelet segment.
    The loss is computed by finding the direction of the lanelet the agent is currently on, and then taking
    -cos(δ) given δ is the orientation difference between the lanelet orientation and the agent orientation,
    unless abs(δ) is smaller or equal to `direction_angle_threshold`, in which case the loss is 0.
    This normalizes the loss range to [0,1].
    In case it is ambiguous which lanelet the agent is on, the one best matching agent orientation is on.

    Args:
        lanelet_maps: a list of B maps; None is means orientation loss won't be computed for that batch element
        agents_state: BxAx4 tensor of agent states x,t,orientation,speed
        recenter_offset: Bx2 tensor that needs to be added to agent coordinates to match them to map
        direction_angle_threshold: minimal angle between the lane direction and agent direction that's considered
            an infraction (needs to be at least pi / 2)
        lanelet_dist_tolerance: how far away from a lanelet the agent can be to still be considered inside
    Returns:
        BxA tensor of orientation losses for all agents
    """
    batch_results = []
    device = agents_state.device

    assert len(lanelet_maps) == agents_state.shape[0]
    if recenter_offset is not None:
        assert len(lanelet_maps) == recenter_offset.shape[0]
    assert direction_angle_threshold >= np.pi / 2,\
        'direction_angle_threshold smaller than pi / 2 will produce false positives'

    for batch_idx in range(len(lanelet_maps)):
        lanelet_map = lanelet_maps[batch_idx]
        agents_results = []
        for a, agent_state in enumerate(agents_state[batch_idx]):
            if not lanelet_map:
                agents_results.append(torch.tensor(0.0).to(device))
                continue
            x, y = float(agent_state[0]), float(agent_state[1])
            agent_psi = agent_state[2]
            if recenter_offset is not None:
                x = x + recenter_offset[batch_idx, 0]
                y = y + recenter_offset[batch_idx, 1]
            try:
                directions = [
                    direction for direction in find_lanelet_directions(
                        lanelet_map=lanelet_map, x=x, y=y, tags_to_exclude=LANELET_TAGS_TO_EXCLUDE,
                        lanelet_dist_tolerance=lanelet_dist_tolerance,
                    )
                ]
                if len(directions) > 0:
                    direction_angle_difference = normalize_angle(torch.tensor(directions).to(device) - agent_psi)
                    losses = - torch.cos(direction_angle_difference) * (
                            torch.abs(direction_angle_difference) > direction_angle_threshold
                    )
                    loss = losses.min()
                else:
                    loss = torch.tensor(0.0).to(device)
            except LaneletError:
                logger.debug(
                    "Lanelet errors occurred during computing the orientation losses."
                    " Setting the wrong way loss for the agent to be zero.")
                loss = torch.tensor(0.0).to(device)
            agents_results.append(loss)
        if not agents_results:
            # In case agent count is zero, a.k.a A=0
            batch_results.append(torch.empty(0))
        else:
            batch_result = torch.stack(agents_results, dim=-1)  # Tensor of shape [A]
            batch_results.append(batch_result)
    results = torch.stack(batch_results).to(device).to(torch.float)

    return results


def iou_differentiable(box1: Tensor, box2: Tensor, fast: bool = True) -> Tensor:
    """
    Computes an approximate, differentiable IoU between two oriented bounding boxes.
    Accepts multiple batch dimensions.

    Args:
        box1: BxAx5 tensor x,y,length,width,orientation
        box2: BxAx5 tensor
        fast: whether to use faster but less accurate method
    Returns:
        tensor of shape (B,A) with IoU values
    """
    if fast:
        iou = iou_differentiable_fast(box1, box2)
    else:
        from driving_models.helpers.iou_slow_utils import iou_differentiable_slow
        iou = iou_differentiable_slow(box1, box2)
    return iou


def compute_agent_collisions_metric_pytorch3d(all_rects: Tensor, masks: Tensor) -> Tensor:
    """
    Computes the number of collisions per agent in batch.

    Args:
        all_rects : BxAx5 tensor of oriented rectangles x,y,length,width,orientation
        masks : BxA boolean tensor, indicating whether to compute collisions for this agent
    Returns:
        BxA tensor with collision counts per agent
    """
    with torch.no_grad():
        all_scores = []
        agent_count = masks.shape[1]
        for batch, rect in enumerate(all_rects):
            all_rect = all_rects[batch]
            masks_i = masks[batch].expand(agent_count, agent_count)
            masks_i = masks_i.clone().fill_diagonal_(0.0).T  # Make sure we don't count collisions with itself
            iou = iou_non_differentiable(all_rect)
            intersects = ((iou > 0.0) * (iou <= 1)).to(all_rect.dtype)
            intersects = (intersects * masks_i).sum(dim=-1)
            all_scores.append(intersects)
        all_scores = torch.stack(all_scores, dim=0)
    return all_scores


def compute_agent_collisions_metric(all_rects: np.ndarray, collision_masks: np.ndarray,
                                    present_masks: np.ndarray) -> np.ndarray:
    """
    Returns the number of collisions per agent in a batch.

    Args:
        all_rects: BxAx5 array of oriented rectangles x,y,length,width,orientation
        collision_masks: BxA boolean array, indicating whether to compute collisions for this agent
        present_masks: BxA boolean array, indicating which rectangles are not padding
    Returns:
        BxA array with collision counts per agent
    """
    all_scores = []
    agent_count = present_masks.shape[1]
    for batch, (rect, mask_i) in enumerate(zip(all_rects, collision_masks)):
        intersects = get_all_intersections(rect)  # (PRESENT x PRESENT) upper triangular numpy array
        intersects[~mask_i] = 0
        intersects = intersects + intersects.T - np.diag(np.diag(intersects))  # Mirror upper to lower triangle
        all_agent_scores = intersects.sum(axis=-1)
        padded_all_agent_scores = np.zeros(agent_count)
        padded_all_agent_scores[present_masks[batch]] = all_agent_scores  # xA
        all_scores.append(padded_all_agent_scores)
    all_scores = np.array(all_scores)
    return all_scores


def bbox2discs(box: Tensor, num_discs: int = 5, backend: str = 'torch') -> Tuple[Tensor, Tensor]:
    """
    Converts a bounding box to the specified number of equally spaced discs with radius half the width.
    B is the batch size. A bounding box is described by 5 values in this order: (x, y, length, width, orientation).)
    Args:
        box: Bx5 tensor or bounding boxes
        num_discs : The number of discs used to represent a vehicle. Must be a positive odd number. (default: 5)
        backend : Either torch or numpy. (default: 'torch')
    Returns:
        A tuple (agent_disc_center, agent_r) where agent_disc_center is the coordinates of the center of each disc
        (B, num_discs, 2) and agent_r is the radius of the discs (B, 1)
    """
    assert isinstance(num_discs, int) and num_discs > 1 and num_discs % 2 != 0
    num_discs_per_side = int((num_discs - 1)/2)
    agent_xy = box[..., 0:2]
    agent_len = box[..., 2:3]
    agent_wid = box[..., 3:4]
    agent_yaw = box[..., 4:5]

    if backend == 'torch':
        agent_r = torch.minimum(agent_len, agent_wid) / 2
        agent_disc_xy = torch.stack([
                i * ((torch.maximum(agent_len, agent_wid) / 2) - agent_r) / num_discs_per_side
                for i in range(-num_discs_per_side, num_discs_per_side+1)], dim=-2)
        agent_disc_xy = F.pad(agent_disc_xy, (0,1)) # We don't modify the y axis

        agent_yaw = (agent_yaw + (np.pi / 2) * (agent_wid > agent_len)).unsqueeze(-2)
        agent_disc_center = torch.cat([
            agent_disc_xy[...,0:1] * torch.cos(agent_yaw) - agent_disc_xy[...,1:2] * torch.sin(agent_yaw),
            agent_disc_xy[...,0:1] * torch.sin(agent_yaw) + agent_disc_xy[...,1:2] * torch.cos(agent_yaw),
        ], dim=-1)
        agent_disc_center += agent_xy.unsqueeze(-2)
    elif backend == 'numpy':
        agent_r = np.minimum(agent_len, agent_wid) / 2
        agent_disc_xy = np.stack([
                i * ((np.maximum(agent_len, agent_wid) / 2) - agent_r) / num_discs_per_side
                for i in range(-num_discs_per_side, num_discs_per_side+1)], axis=-2)
        # We don't modify the y axis
        agent_disc_xy = np.pad(agent_disc_xy, ((0,0), (0,0), (0,1)), 'constant', constant_values=0.0)

        agent_yaw = np.expand_dims(agent_yaw + (np.pi / 2) * (agent_wid > agent_len), -2)
        agent_disc_center = np.concatenate([
            agent_disc_xy[...,0:1] * np.cos(agent_yaw) - agent_disc_xy[...,1:2] * np.sin(agent_yaw),
            agent_disc_xy[...,0:1] * np.sin(agent_yaw) + agent_disc_xy[...,1:2] * np.cos(agent_yaw),
        ], axis=-1)
        agent_disc_center += np.expand_dims(agent_xy, -2)
    else:
        raise ValueError('Unknown backend framework.')
    return agent_disc_center, agent_r


def get_all_intersections(rects: np.ndarray, ego_idx: Optional[int] = None) -> np.ndarray:
    """
    Calculates an upper triangular matrix of size equal to the number of rotated rectangles which
    indicating whether one rectangle intersects with another.
    Args:
        rects: (agents_count x 5) numpy array
        ego_idx: Optional ego index for ego only mode
    Returns:
        An upper triangular matrix of size equal to the number of rotated rectangles
    """
    m = len(rects)
    polys = rectangle_vertices(*np.split(rects, rects.shape[-1], axis=-1))
    polys = [Polygon([p[0], p[1], p[2], p[3]]) for p in
             polys]  # Create polygons for fast calculation of the intersection
    if ego_idx is None:
        intersections = np.zeros((m, m))
        if m > 100 and importlib.util.find_spec('rtree') is not None:
            # If the number of rectangles is large, we use spatial index for fast filtering of non-overlapping rectangles.
            import rtree.index
            # Build a spatial index for rects.
            index_a = rtree.index.Index()
            for i, a in enumerate(polys):
                index_a.insert(i, a.bounds)
            # Find candidate intersections using the spatial index.
            all_intersection_indices = (index_a.intersection(polys[i].bounds) for i in range(m))
        else:
            all_intersection_indices = (range(m) for _ in range(m))
        for i in range(m):
            intersection_indices = next(all_intersection_indices)  # Make sure to consume the indices generator
            b = polys[i]
            for j in intersection_indices:
                if j < i:  # Calculate only the upper triangle
                    a = polys[j]
                    intersection_area = a.intersection(b).area
                    if intersection_area:
                        intersections[j, i] = 1
                else:
                    break
    else:
        intersections = np.zeros((m - 1))
        ego_poly = polys.pop(ego_idx)
        for i in range(len(polys)):
            intersection_area = polys[i].intersection(ego_poly).area
            if intersection_area:
                intersections[i] = 1
    return intersections


def rectangle_vertices(cx: np.ndarray, cy: np.ndarray, w: np.ndarray, h: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Calculate the 4 corners of the rotated rectangles given the center point, the size and the orientation.
    Args:
        cx: Bx1 array with x coordinate of the center point
        cy: Bx1 array with y coordinate of the center point
        w: Bx1 array with width of the rectangle
        h: Bx1 array with height of the rectangle
        angle: Bx1 array with yaw angle of the rectangle
    Returns:
        Bx4 corners of the rectangle
    """
    dx = w/2
    dy = h/2
    dxcos = dx*np.cos(angle)
    dxsin = dx*np.sin(angle)
    dycos = dy*np.cos(angle)
    dysin = dy*np.sin(angle)
    return np.stack([
        np.concatenate([cx, cy], axis=-1) + np.concatenate([-dxcos - -dysin, -dxsin + -dycos], axis=-1),
        np.concatenate([cx, cy], axis=-1) + np.concatenate([ dxcos - -dysin,  dxsin + -dycos], axis=-1),
        np.concatenate([cx, cy], axis=-1) + np.concatenate([ dxcos -  dysin,  dxsin +  dycos], axis=-1),
        np.concatenate([cx, cy], axis=-1) + np.concatenate([-dxcos -  dysin, -dxsin +  dycos], axis=-1)
    ], axis=1)


def collision_detection_with_discs(box1: Tensor, box2: Tensor, num_discs: int = 5, backend: str = 'torch'):
    """
    Calculate the differentiable collision loss as described in the paper TrafficSim.
    Accepts multiple batch dimensions.
    A bounding box is described by 5 values in this order: (x, y, length, width, orientation)

    Args:
        box1: BxAx5 tensor
        box2: BxAx5 tensor
        num_discs: The number of discs used to represent a vehicle. Must be a positive odd number.
        backend: Either torch or numpy.

    Returns:
        tensor of shape (B,A) with collision values between corresponding agents
    """
    batch_size = box1.shape[0]
    num_agents = box1.shape[1]

    agent1_disc_center, agent1_r = bbox2discs(box1.reshape(batch_size*num_agents, 5), num_discs, backend=backend)
    agent2_disc_center, agent2_r = bbox2discs(box2.reshape(batch_size*num_agents, 5), num_discs, backend=backend)

    agent1_r, agent2_r = agent1_r.reshape(batch_size, num_agents), agent2_r.reshape(batch_size, num_agents)

    if backend == 'torch':
        d = torch.cdist(agent1_disc_center, agent2_disc_center, p=2.0) # Euclidean distance
        d = d.reshape(batch_size, num_agents, num_discs*num_discs)
        d, _ = torch.min(d, dim=-1)

        loss_value = torch.relu(1 - (d / (agent1_r + agent2_r))) # ReLU for cases when d > r_i + r_j
    elif backend == 'numpy':
        from scipy.spatial.distance import cdist

        d = []
        for i in range(agent1_disc_center.shape[0]):
            d.append(cdist(agent1_disc_center[i], agent2_disc_center[i], metric='euclidean')) # Euclidean distance
        d = np.stack(d, axis=0)
        d = d.reshape(batch_size, num_agents, num_discs*num_discs)
        d = np.amin(d, axis=-1)

        loss_value = np.maximum(1 - (d / (agent1_r + agent2_r)), 0) # ReLU for cases when d > r_i + r_j
    else:
        raise ValueError('Unknown backend framework.')
    return loss_value
