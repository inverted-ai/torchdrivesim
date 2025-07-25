import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

import torch

from torchdrivesim.utils import line_circle_intersection


@dataclass
class ObservationNoiseConfig:
    _type_: str = 'base'


@dataclass
class StandardSensingObservationNoiseConfig:
    _type_: str = 'standard_sensing'


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
        ego_target_mask[:, agent_indices, agent_indices, :] = True

        occluding = occluding & ~self_mask & ~ego_target_mask

        # An entity is occluded if ANY other entity occludes it from the ego's perspective
        occluded = occluding.any(dim=-1)  # [B, A, E]

        # Apply occlusion to the base mask (hide occluded entities)
        final_mask = base_mask & ~occluded

        return final_mask
