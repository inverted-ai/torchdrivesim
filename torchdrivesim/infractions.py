import importlib
from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import torch
from shapely.geometry import Polygon
from torch import Tensor, relu, cosine_similarity
from torch.nn import functional as F

from torchdrivesim._iou_utils import box2corners_th, iou_differentiable_fast, iou_non_differentiable
from torchdrivesim.lanelet2 import LaneletMap, find_lanelet_directions, LaneletError
from torchdrivesim.mesh import BaseMesh, check_pytorch3d_available

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

    check_pytorch3d_available()
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


def offroad_infraction_loss(agent_states: Tensor, lenwid: Tensor,
                            driving_surface_mesh: Union["pytorch3d.structures.Meshes", BaseMesh],
                            threshold: float = 0) -> Tensor:
    """
    Calculates off-road infraction loss, defined as the sum of thresholded distances
    from agent corners to the driving surface mesh.

    Args:
        agent_states: BxAx4 tensor of agent states x,y,orientation,speed
        lenwid: BxAx2 tensor providing length and width of each agent
        driving_surface_mesh: a batch of B meshes defining driving surface
        threshold: if given, the distance of each corner is thresholded using this value
    Returns:
        BxA tensor of offroad losses for each agent
    """
    check_pytorch3d_available()
    import pytorch3d
    batch_size, sequence_length = agent_states.shape[:2]
    if sequence_length == 0:
        return torch.zeros_like(agent_states[..., 0])
    if len(lenwid.shape) == 2:
        lenwid = lenwid.unsqueeze(-2).expand((lenwid.shape[0], sequence_length, lenwid.shape[1]))
    predicted_rectangles = torch.cat([
        agent_states[..., :2],
            lenwid,
        agent_states[..., 2:3],
        ], dim=-1)
    ego_verts = box2corners_th(predicted_rectangles)
    ego_verts = F.pad(ego_verts, (0,1))
    ego_verts = ego_verts.view(-1, *ego_verts.shape[2:])
    ego_pointclouds = pytorch3d.structures.Pointclouds(ego_verts)
    if isinstance(driving_surface_mesh, BaseMesh):
        driving_surface_mesh = driving_surface_mesh.pytorch3d(include_textures=False)
    driving_surface_mesh_extended = driving_surface_mesh.extend(sequence_length)
    offroad_loss = point_mesh_face_distance(driving_surface_mesh_extended, ego_pointclouds, threshold=threshold)
    offroad_loss = offroad_loss.view(batch_size, sequence_length)
    return offroad_loss


def lanelet_orientation_loss(lanelet_maps: List[Optional[LaneletMap]], agents_state: Tensor,
                             recenter_offset: Optional[Tensor] = None) -> Tensor:
    """
    Calculate a loss value of the agent orientation regarding to the direction of the lanelet segment.
    The loss is computed by finding the direction of the lanelet the agent is currently on, and then taking
    max(0, -cos(δ)) given δ is the orientation difference between the lanelet orientation and the agent orientation.
    This normalizes the loss range from 0-1 while ignoring the angle difference less than 90 degrees.
    In case it is ambiguous which lanelet the agent is on, the one best matching agent orientation is on.

    Args:
        lanelet_maps: a list of B maps; None is means orientation loss won't be computed for that batch element
        agents_state: BxAx4 tensor of agent states x,t,orientation,speed
        recenter_offset: Bx2 tensor that needs to be added to agent coordinates to match them to map
    Returns:
        BxA tensor of orientation losses for all agents
    """
    batch_results = []
    device = agents_state.device

    assert len(lanelet_maps) == agents_state.shape[0]
    if recenter_offset is not None:
        assert len(lanelet_maps) == recenter_offset.shape[0]

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
            agent_direction = torch.stack([torch.cos(agent_psi), torch.sin(agent_psi)])
            try:
                directions = [
                    torch.tensor(direction) for direction in find_lanelet_directions(
                        lanelet_map=lanelet_map, x=x, y=y, tags_to_exclude=LANELET_TAGS_TO_EXCLUDE
                    )
                ]
                losses = [
                    relu(-cosine_similarity(
                        agent_direction, torch.stack([torch.cos(psi), torch.sin(psi)]).to(device), dim=0
                    )) for psi in directions
                ]
                if len(losses) > 0:
                    loss = torch.min(torch.stack(losses), dim=0).values
                else:
                    loss = torch.tensor(0.0).to(device)
            except LaneletError:
                logger.error(
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
        box1: Bx5 tensor x,y,length,width,orientation
        box2: Bx5 tensor
        fast: whether to use faster but less accurate method
    Returns:
        tensor of shape (B,) with IoU values
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

    agent_r = agent_wid/2

    if backend == 'torch':
        agent_disc_xy = torch.stack([
                i * ((agent_len / 2) - agent_r) / num_discs_per_side
                for i in range(-num_discs_per_side, num_discs_per_side+1)], dim=-2)
        agent_disc_xy = F.pad(agent_disc_xy, (0,1)) # We don't modify the y axis

        agent_yaw = agent_yaw.unsqueeze(-2)
        agent_disc_center = torch.cat([
            agent_disc_xy[...,0:1] * torch.cos(agent_yaw) - agent_disc_xy[...,1:2] * torch.sin(agent_yaw),
            agent_disc_xy[...,0:1] * torch.sin(agent_yaw) + agent_disc_xy[...,1:2] * torch.cos(agent_yaw),
        ], dim=-1)
        agent_disc_center += agent_xy.unsqueeze(-2)
    elif backend == 'numpy':
        agent_disc_xy = np.stack([
                i * ((agent_len / 2) - agent_r) / num_discs_per_side
                for i in range(-num_discs_per_side, num_discs_per_side+1)], axis=-2)
        # We don't modify the y axis
        agent_disc_xy = np.pad(agent_disc_xy, ((0,0), (0,0), (0,1)), 'constant', constant_values=0.0)

        agent_yaw = np.expand_dims(agent_yaw, -2)
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
        box1: Bx5 tensor
        box2: Bx5 tensor
        num_discs: The number of discs used to represent a vehicle. Must be a positive odd number.
        backend: Either torch or numpy.

    Returns:
        tensor of shape (B,) with collision values between corresponding agents
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
