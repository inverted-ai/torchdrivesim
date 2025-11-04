import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from torchdrivesim.utils import line_circle_intersection
from torchdrivesim.lanelet2 import LaneFeatures


@dataclass
class ObservationNoiseConfig:
    _type_: str = 'base'


@dataclass
class StandardSensingObservationNoiseConfig:
    _type_: str = 'standard_sensing'


@dataclass
class EgocentricLanePruningObservationNoiseConfig(ObservationNoiseConfig):
    _type_: str = 'egocentric_lane_pruning'
    # Agent-visibility settings (in ego frame)
    max_forward_m: float = 50.0       # Max distance to look ahead for agents
    max_backward_m: float = 0.0        # Small look-behind window
    require_ahead: bool = True         # Only keep agents with rel_x > 0
    fov_angle_deg: float = 22.5        # Field of view angle
    use_fov_constraint: bool = True    # Whether to apply FOV
    agent_heading_align_cos_min: float = 0.85

    # Lane settings
    lane_max_lateral_m: float = 2.5    # Lane half-width for both lanes and agents
    lane_max_forward_m: float = 100.0
    lane_max_backward_m: float = 0.5
    heading_align_cos_min: float = 0.7

    # Randomization
    lane_lateral_noise_std: float = 0.0    # Noise for lateral threshold
    lane_lateral_noise_min: float = 1.5    # Min lateral (always see current lane)
    lane_lateral_noise_max: float = 8.0    # Max lateral (cap the noise)
    random_per_agent: bool = True          # Different noise per agent

    # Progressive lane noise for distant features
    lane_distance_noise_enabled: bool = True      # Enable distance-based noise
    lane_distance_noise_start_m: float = 10.0     # Start adding noise after this distance
    lane_distance_noise_max_m: float = 100.0      # Maximum noise distance (full noise)
    lane_distance_noise_max_lateral: float = 5.0  # Max lateral deviation at max distance
    lane_distance_noise_smoothness: float = 40.0  # Smoothness of noise (higher = smoother curves)
    lane_distance_noise_seed_change: bool = True  # Change noise pattern each timestep


class ObservationNoise:
    def __init__(self, cfg: ObservationNoiseConfig):
        self.cfg = cfg

    def get_noisy_state(self, simulator):
        return torch.cat([
            simulator.get_state()[:, None, :, :].expand(-1, simulator.agent_count, -1, -1),
            simulator.get_npc_state()[:, None, :, :].expand(-1, simulator.agent_count, -1, -1),
        ], dim=-2)

    def get_noisy_present_mask(self, simulator):
        return torch.cat([
            simulator.get_present_mask()[:, None, :].expand(-1, simulator.agent_count, -1),
            simulator.get_npc_present_mask()[:, None, :].expand(-1, simulator.agent_count, -1),
        ], dim=-1)

    def get_noisy_agent_size(self, simulator):
        return torch.cat([
            simulator.get_agent_size()[:, None, :, :].expand(-1, simulator.agent_count, -1, -1),
            simulator.get_npc_size()[:, None, :, :].expand(-1, simulator.agent_count, -1, -1),
        ], dim=-2)

    def get_noisy_lane_features(self, simulator) -> "LaneFeatures":
        return simulator.lane_features


class StandardSensingObservationNoise(ObservationNoise):
    def __init__(self, cfg: StandardSensingObservationNoiseConfig):
        self.cfg = cfg

    def get_noisy_state(self, simulator):
        exposed_states = simulator.get_state()   # [B, A, 4]
        all_states = super().get_noisy_state(simulator)  # [B, A, A+Npc, 4]

        exposed_xy = exposed_states[..., :2]
        all_xy = all_states[..., :2]
        distance_from_ego = torch.norm(exposed_xy[..., None, :] - all_xy, dim=-1)
        deviation = torch.max(torch.stack([
            0.19 * (distance_from_ego > 0.5),
            1.6 * (distance_from_ego > 25),
            3.2 * (distance_from_ego > 50),
            3.83 * (distance_from_ego > 100),
        ], dim=-1), dim=-1, keepdim=True).values
        noisy_states = all_states + torch.randn_like(all_states) * deviation
        return noisy_states

    def get_noisy_present_mask(self, simulator):
        base_mask = super().get_noisy_present_mask(simulator)  # [B, A, A+Npc]

        states = super().get_noisy_state(simulator)  # [B, A, A+Npc, 4]
        sizes = super().get_noisy_agent_size(simulator)  # [B, A, A+Npc, 2]

        batch_size, agent_count, total_entities = base_mask.shape

        agent_indices = torch.arange(agent_count, device=states.device)
        ego_pos = states[:, agent_indices, agent_indices, :2]  # [B, A, 2]

        # Create expanded tensors for all pairwise occlusion calculations
        ego_expanded = ego_pos[:, :, None, None, :].expand(-1, -1, total_entities, total_entities, -1)  # [B, A, E, E, 2]

        # Target entity positions (entities being potentially occluded)
        target_pos = states[:, :, :, None, :2].expand(-1, -1, -1, total_entities, -1)  # [B, A, E, E, 2]

        # Occluder entity positions (entities doing the occluding)
        occluder_pos = states[:, :, None, :, :2].expand(-1, -1, total_entities, -1, -1)  # [B, A, E, E, 2]

        # Occluder radii
        occluder_radius = sizes[:, :, None, :, 1:2].expand(-1, -1, total_entities, -1, -1) / 2  # [B, A, E, E, 1]

        # Check line-circle intersections for all combinations
        occluding = line_circle_intersection(
            ego_expanded, target_pos, occluder_pos, occluder_radius
        ).squeeze(-1)  # [B, A, E, E]

        # Exclude self-occlusion (entity cannot occlude itself)
        self_mask = torch.eye(total_entities, device=occluding.device, dtype=torch.bool)[None, None, :, :]

        # Exclude ego agents being occluded from their own perspective
        ego_target_mask = torch.zeros(batch_size, agent_count, total_entities, total_entities, dtype=torch.bool, device=occluding.device)
        ego_target_mask[:, agent_indices, :, agent_indices] = True

        occluding = occluding & ~self_mask & ~ego_target_mask

        # An entity is occluded if ANY other entity occludes it from the ego's perspective
        occluded = occluding.any(dim=-1)  # [B, A, E]

        # Apply occlusion to the base mask (hide occluded entities)
        final_mask = base_mask & ~occluded

        return final_mask


class EgocentricLanePruningObservationNoise(ObservationNoise):
    def __init__(self, cfg: EgocentricLanePruningObservationNoiseConfig):
        super().__init__(cfg)
        self.cfg = cfg

    # 1) Limit visibility to only the car in front (if any)
    def get_noisy_present_mask(self, simulator):
        # Start with vanilla present mask (agents + NPCs) tiled per-exposed agent
        base_mask = super().get_noisy_present_mask(simulator)  # [B, A, A+Npc]

        states = super().get_noisy_state(simulator)            # [B, A, A+Npc, 4]
        B, A, E, _ = states.shape

        # Ego poses (one per exposed agent)
        agent_idx = torch.arange(A, device=states.device)
        ego_xy  = states[:, agent_idx, agent_idx, :2]          # [B, A, 2]
        ego_psi = states[:, agent_idx, agent_idx, 2:3]         # [B, A, 1]

        # All entities as seen by each exposed agent
        all_xy  = states[:, :, :, :2]                          # [B, A, E, 2]
        all_psi = states[:, :, :, 2:3]                         # [B, A, E, 1] - heading of all entities

        # Transform to ego frame: R * (p - ego)
        s, c = torch.sin(ego_psi), torch.cos(ego_psi)
        R_T = torch.stack([torch.stack([c, -s], dim=-1),
                           torch.stack([s, c], dim=-1)], dim=-2)  # [B, A, 2, 2]
        rel_xy = torch.matmul(all_xy - ego_xy.unsqueeze(-2), R_T)   # [B, A, E, 2]

        rel_x = rel_xy[..., 0]  # [B, A, E]
        rel_y = rel_xy[..., 1]  # [B, A, E]

        # Calculate distance and bearing angle for FOV constraint
        distance = torch.sqrt(rel_x**2 + rel_y**2)  # [B, A, E]
        bearing_angle_rad = torch.atan2(rel_y, rel_x)  # [B, A, E]

        # Forward gate - how far ahead/behind we look
        forward_gate = (rel_x <= self.cfg.max_forward_m) & (rel_x >= -self.cfg.max_backward_m)

        # Same-lane lateral gate
        lane_width = 2.0
        lane_lateral_threshold = torch.full((B, A, 1), lane_width, device=states.device)
        same_lane_gate = rel_y.abs() <= lane_lateral_threshold

        # Only look ahead if configured
        ahead_gate = (rel_x > 0) if self.cfg.require_ahead else torch.ones_like(rel_x, dtype=torch.bool)

        # Field of view constraint (if enabled)
        if self.cfg.use_fov_constraint:
            half_fov_rad = torch.deg2rad(torch.tensor(self.cfg.fov_angle_deg / 2.0, device=states.device))
            fov_gate = torch.abs(bearing_angle_rad) <= half_fov_rad
        else:
            fov_gate = torch.ones_like(rel_x, dtype=torch.bool)

        # HEADING ALIGNMENT GATE
        # Check if the other agent's heading is aligned with ego's heading
        heading_diff = all_psi - ego_psi.unsqueeze(-2)  # [B, A, E, 1]

        # Normalize heading difference to [-pi, pi]
        heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))

        # Use cosine similarity for alignment check (same as lane features)
        # cos(heading_diff) >= threshold means aligned
        # Default threshold: 0.7 ≈ ±45°, 0.85 ≈ ±32°, 0.9 ≈ ±25°
        heading_align_threshold = getattr(self.cfg, 'agent_heading_align_cos_min',
                                          self.cfg.heading_align_cos_min)  # Use same as lanes or separate
        alignment_gate = torch.cos(heading_diff).squeeze(-1) >= heading_align_threshold

        # Combine all visibility constraints
        vis_gate = forward_gate & same_lane_gate & ahead_gate & fov_gate & alignment_gate & base_mask

        # Don't consider the ego itself
        self_mask = torch.zeros((B, A, E), device=states.device, dtype=torch.bool)
        batch_idx = torch.arange(B, device=states.device)[:, None]
        self_mask[batch_idx, agent_idx[None, :], agent_idx[None, :]] = True

        # Set ego's own position to inf so it won't be selected
        candidates = torch.where(self_mask, torch.inf, rel_x)

        # Set non-visible candidates to inf
        candidates = torch.where(vis_gate, candidates, torch.inf)

        # Pick nearest-in-front: argmin rel_x among valid candidates
        front_idx = candidates.argmin(dim=-1)  # [B, A]

        # Check if we actually found a valid front entity
        has_front = torch.isfinite(candidates.min(dim=-1).values)  # [B, A]

        # Build a mask that keeps ONLY that front entity
        keep = torch.zeros_like(base_mask, dtype=torch.bool)  # [B, A, E]

        # Set the front entity to True for each ego agent that has one
        valid_idx = has_front.nonzero(as_tuple=True)
        if len(valid_idx[0]) > 0:
            keep[valid_idx[0], valid_idx[1], front_idx[valid_idx]] = True

        # Keep the ego itself visible (for self-awareness)
        keep[batch_idx, agent_idx[None, :], agent_idx[None, :]] = True

        # Final mask is intersection with base present mask
        final_mask = keep & base_mask

        return final_mask

    # 2) Prune lane features to ego’s "current lane"
    def get_noisy_lane_features(self, simulator) -> "LaneFeatures":
        lf = simulator.lane_features
        if lf is None:
            return None

        states = simulator.get_state()            # [B, A, 4]
        B, A, _ = states.shape
        ego_xy  = states[..., :2]                 # [B, A, 2]
        ego_psi = states[..., 2:3]                # [B, A, 1]

        # Generate random lateral thresholds if configured
        if self.cfg.lane_lateral_noise_std > 0:
            if self.cfg.random_per_agent:
                lateral_thresholds = self.cfg.lane_max_lateral_m + \
                    torch.randn(B, A, 1, device=states.device) * self.cfg.lane_lateral_noise_std
            else:
                lateral_thresholds = self.cfg.lane_max_lateral_m + \
                    torch.randn(B, 1, 1, device=states.device) * self.cfg.lane_lateral_noise_std

            lateral_thresholds = torch.clamp(
                lateral_thresholds,
                min=self.cfg.lane_lateral_noise_min,
                max=self.cfg.lane_lateral_noise_max
            )
        else:
            lateral_thresholds = torch.full((B, A, 1), self.cfg.lane_max_lateral_m, device=states.device)

        def _generate_smooth_curve_noise(rel_x, B, A, M, device):
            """Generate smooth, distance-based noise that simulates curve misperception"""
            if not self.cfg.lane_distance_noise_enabled:
                return torch.zeros(B, A, M, device=device)

            # Distance factor: 0 at start_distance, 1 at max_distance
            distance_factor = (rel_x - self.cfg.lane_distance_noise_start_m) / \
                             (self.cfg.lane_distance_noise_max_m - self.cfg.lane_distance_noise_start_m)
            distance_factor = torch.clamp(distance_factor, 0, 1)

            # Generate smooth base noise using multiple sine waves (creates realistic curve variations)
            # This simulates misperceiving the road curvature
            if self.cfg.lane_distance_noise_seed_change:
                # Different noise each timestep
                phase_offsets = torch.rand(B, A, 3, device=device) * 2 * np.pi
            else:
                # Consistent noise (you might want to cache this)
                if not hasattr(self, '_noise_phase_cache'):
                    self._noise_phase_cache = torch.rand(1, 1, 3, device=device) * 2 * np.pi
                phase_offsets = self._noise_phase_cache.expand(B, A, -1)

            # Create smooth curves using different frequency components
            # Low frequency for general curve, higher frequencies for smaller variations
            smoothness = self.cfg.lane_distance_noise_smoothness

            # Primary curve component (low frequency)
            curve1 = torch.sin(rel_x.unsqueeze(-1) / smoothness + phase_offsets[..., 0:1])

            # Secondary variation (medium frequency)
            curve2 = 0.3 * torch.sin(rel_x.unsqueeze(-1) * 2 / smoothness + phase_offsets[..., 1:2])

            # Tertiary detail (higher frequency, smaller amplitude)
            curve3 = 0.1 * torch.sin(rel_x.unsqueeze(-1) * 4 / smoothness + phase_offsets[..., 2:3])

            # Combine curves
            combined_curve = (curve1 + curve2 + curve3).squeeze(-1)

            # Add some randomness per agent for variation
            agent_variation = torch.randn(B, A, 1, device=device) * 0.3
            combined_curve = combined_curve * (1 + agent_variation)

            # Scale by distance and max lateral deviation
            # Quadratic scaling can make distant noise more pronounced
            distance_scaling = distance_factor ** 1.5  # Slightly superlinear growth
            lateral_noise = combined_curve * distance_scaling * self.cfg.lane_distance_noise_max_lateral

            return lateral_noise

        def _prune_block(feat: Optional[Tensor], feat_mask: Optional[Tensor]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
            if feat is None:
                return None, None

            M = feat.shape[1]  # Number of lane points
            xy  = feat[..., :2]                   # [B, M, 2]
            psi = feat[..., 2:3] if feat.shape[-1] > 2 else None

            # Expand per exposed agent
            xy_e   = xy.unsqueeze(1).expand(B, A, -1, -1)           # [B, A, M, 2]
            psi_e  = psi.unsqueeze(1).expand(B, A, -1, -1) if psi is not None else None
            ego_xy_e  = ego_xy.unsqueeze(-2)                        # [B, A, 1, 2]
            ego_psi_e = ego_psi                                     # [B, A, 1]

            # Rotate points to ego frame
            s, c = torch.sin(ego_psi_e), torch.cos(ego_psi_e)
            R_T = torch.stack([torch.stack([c, -s], dim=-1),
                              torch.stack([s, c], dim=-1)], dim=-2)  # [B, A, 2, 2]
            rel_xy = torch.matmul(xy_e - ego_xy_e, R_T)            # [B, A, M, 2]
            rel_x, rel_y = rel_xy[..., 0], rel_xy[..., 1]

            # APPLY PROGRESSIVE CURVE NOISE
            if self.cfg.lane_distance_noise_enabled:
                # Generate smooth lateral noise that increases with distance
                lateral_noise = _generate_smooth_curve_noise(rel_x, B, A, M, states.device)

                # Apply noise to lateral positions (creates curve misperception)
                rel_y_noisy = rel_y + lateral_noise

                # Optionally add small longitudinal noise for realism
                if hasattr(self.cfg, 'lane_distance_noise_longitudinal_factor'):
                    long_noise = lateral_noise * self.cfg.lane_distance_noise_longitudinal_factor
                    rel_x = rel_x + long_noise * 0.1  # Much smaller longitudinal effect
            else:
                rel_y_noisy = rel_y

            # Forward/backward gate
            forward_gate = (rel_x <= self.cfg.lane_max_forward_m) & (rel_x >= -self.cfg.lane_max_backward_m)

            # Lateral gate with noise (use noisy rel_y)
            lateral_gate = rel_y_noisy.abs() <= lateral_thresholds

            # Heading alignment gate
            if psi_e is not None:
                delta = (psi_e - ego_psi_e)

                # Optionally add noise to perceived lane heading for distant points
                if self.cfg.lane_distance_noise_enabled and hasattr(self.cfg, 'lane_distance_noise_heading'):
                    distance_factor = torch.clamp(
                        (rel_x - self.cfg.lane_distance_noise_start_m) /
                        (self.cfg.lane_distance_noise_max_m - self.cfg.lane_distance_noise_start_m),
                        0, 1
                    )
                    heading_noise = torch.randn_like(delta) * distance_factor.unsqueeze(-1) * \
                                   self.cfg.lane_distance_noise_heading
                    delta = delta + heading_noise

                align = torch.cos(delta).squeeze(-1) >= self.cfg.heading_align_cos_min
            else:
                align = torch.ones_like(forward_gate)

            # Combine all gates
            gate = forward_gate & lateral_gate & align

            # Apply existing mask if provided
            if feat_mask is not None:
                mask_e = gate & feat_mask.unsqueeze(1).expand_as(gate)
            else:
                mask_e = gate

            # Apply mask to features and add noise to the actual positions
            feat_e = feat.unsqueeze(1).expand(B, A, -1, -1).clone()

            if self.cfg.lane_distance_noise_enabled:
                # Transform noise back to global frame and apply to actual features
                # This ensures the returned features have the noise baked in
                noise_ego_frame = torch.zeros_like(rel_xy)
                noise_ego_frame[..., 1] = lateral_noise  # Apply noise to lateral component

                # Rotate noise back to global frame
                R = torch.stack([torch.stack([c, s], dim=-1),
                                torch.stack([-s, c], dim=-1)], dim=-2)
                noise_global = torch.matmul(noise_ego_frame, R)

                # Apply noise to feature positions
                feat_e[..., :2] = feat_e[..., :2] + noise_global * mask_e.unsqueeze(-1)

            pruned = torch.where(mask_e.unsqueeze(-1), feat_e, torch.zeros_like(feat_e))

            return pruned, mask_e

        dense_f, dense_m = _prune_block(lf.dense_lane_features, lf.dense_lane_features_mask)
        sparse_f, sparse_m = _prune_block(lf.sparse_lane_features, lf.sparse_lane_features_mask)

        out = LaneFeatures(
            dense_lane_features=dense_f.squeeze(2),
            dense_lane_features_mask=dense_m.squeeze(2),
            sparse_lane_features=sparse_f.squeeze(2),
            sparse_lane_features_mask=sparse_m.squeeze(2),
        )
        return out
