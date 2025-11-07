import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

import torch

from torchdrivesim.mesh import BirdviewMesh, set_colors_with_defaults
from torchdrivesim.utils import line_circle_intersection
from torchdrivesim.lanelet2 import LaneFeatures


@dataclass
class ObservationNoiseConfig:
    _type_: str = 'base'


@dataclass
class StandardSensingObservationNoiseConfig:
    _type_: str = 'standard_sensing'


@dataclass
class MapObservationNoiseFromLogConfig:
    _type_: str = 'map_observation_noise_from_log'


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

    def get_noisy_lane_features(self, simulator):
        # TODO: This should return (noisy) lane features for each agent separately
        # TODO: (the LaneFeatures class should be expanded to support this).
        return simulator.lane_features

    def get_noisy_background_mesh(self, simulator):
        return simulator.birdview_mesh_generator.background_mesh


class StandardSensingObservationNoise(ObservationNoise):
    def __init__(self, cfg: StandardSensingObservationNoiseConfig):
        super().__init__(cfg)

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


class MapObservationNoiseFromLog(ObservationNoise):
    def __init__(self, cfg: StandardSensingObservationNoiseConfig,
                 noisy_lane_features: Optional[List[LaneFeatures]] = None,
                 noisy_background_mesh: Optional[List[BirdviewMesh]] = None):
        super().__init__(cfg)
        self.noisy_lane_features = noisy_lane_features
        self.noisy_background_mesh = noisy_background_mesh

    def get_noisy_lane_features(self, simulator):
        if self.noisy_lane_features is not None and simulator.internal_time < len(self.noisy_lane_features):
            return self.noisy_lane_features[simulator.internal_time]
        else:
            return simulator.lane_features

    def get_noisy_background_mesh(self, simulator):
        if self.noisy_background_mesh is not None and simulator.internal_time < len(self.noisy_background_mesh):
            bg_mesh = self.noisy_background_mesh[simulator.internal_time]
            background_mesh = set_colors_with_defaults(bg_mesh.clone(), color_map=simulator.birdview_mesh_generator.color_map,
                                                       rendering_levels=simulator.birdview_mesh_generator.rendering_levels)
            return background_mesh
        else:
            return simulator.birdview_mesh_generator.background_mesh
