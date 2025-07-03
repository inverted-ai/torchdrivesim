import os
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torchdrivesim.behavior.common import InitializationFailedError
from torchdrivesim.simulator import NPCController, Simulator, SpawnController


def interaction_replay(location, dataset_path, initial_frame=1, segment_length=40, recording=0):
    recording_path = os.path.join(dataset_path, 'recorded_trackfiles', location, 'vehicle_tracks_{:03d}.csv'.format(recording))
    df = pd.read_csv(recording_path)
    final_frame = initial_frame + segment_length - 1
    for frame in [initial_frame, final_frame]:
        if frame not in df.frame_id.unique():
            raise InitializationFailedError(f'Frame {frame} not available in {recording_path}')
    df = df[(df.frame_id >= initial_frame) & (df.frame_id <= final_frame)].copy()
    df = df.sort_values(['track_id', 'frame_id'])

    df['rear_offset'] = 1.4
    agent_ids = sorted(df.track_id.unique())
    agent_attributes = []
    for agent_id in agent_ids:
        attr = df[df.track_id == agent_id][['length', 'width', 'rear_offset']].to_numpy()
        attr = torch.from_numpy(attr).mean(dim=-2)
        agent_attributes.append(attr)
    agent_attributes = torch.stack(agent_attributes, dim=-2).unsqueeze(0)

    df['present'] = True
    df['speed'] = np.sqrt(df.vx ** 2 + df.vy ** 2)
    frame_ids = sorted(df.frame_id.unique())
    dense_index = pd.MultiIndex.from_product([agent_ids, frame_ids], names=["track_id", "frame_id"])
    padding = pd.DataFrame(index=dense_index, data=dict(x=0.0, y=0.0, psi_rad=0.0, speed=0.0, present=False))
    daug = df.set_index(['track_id', 'frame_id']).reindex(dense_index).combine_first(padding)
    agent_states = torch.from_numpy(daug[['x', 'y', 'psi_rad', 'speed']].to_numpy()).unsqueeze(0)
    agent_states = agent_states.reshape(1, len(agent_ids), len(frame_ids), 4)
    present_mask = torch.from_numpy(daug['present'].astype(bool).to_numpy()).unsqueeze(0)
    present_mask = present_mask.reshape(1, len(agent_ids), len(frame_ids))

    return agent_attributes, agent_states, present_mask


class ReplayController(NPCController):
    def __init__(self, npc_size, npc_states, npc_present_masks: Optional[torch.Tensor] = None, time: int = 0,
                 npc_types: Optional[Tensor] = None, agent_type_names: Optional[List[str]] = None, spawn_controller: Optional[SpawnController] = None):
        self.time = time
        self.npc_states = npc_states
        self.npc_present_masks = npc_present_masks
        if self.npc_present_masks is None:
            self.npc_present_masks = torch.ones_like(self.npc_states[..., 0], dtype=torch.bool)
        super().__init__(npc_size, self.npc_states[..., self.time, :], self.npc_present_masks[..., self.time], npc_types, agent_type_names, spawn_controller)

    def advance_npcs(self, simulator: Simulator) -> None:
        self.time += 1
        if self.time == self.npc_states.shape[-2]:
            self.time = 0
        self.npc_state = self.npc_states[..., self.time, :]
        self.npc_present_mask = self.npc_present_masks[..., self.time]
        self.spawn_despawn_npcs(simulator)

    def to(self, device):
        self.npc_size = self.npc_size.to(device)
        self.npc_state = self.npc_state.to(device)
        self.npc_present_mask = self.npc_present_mask.to(device)
        self.npc_types = self.npc_types.to(device)
        self.npc_states = self.npc_states.to(device)
        self.npc_present_masks = self.npc_present_masks.to(device)
        self.spawn_controller.to(device)
        return self

    def copy(self):
        obj = self.__class__(self.npc_size, self.npc_states, self.npc_present_masks, self.time, self.npc_types, self.agent_type_names, self.spawn_controller.copy())
        obj.npc_state = self.npc_state.clone()
        obj.npc_present_mask = self.npc_present_mask.clone()
        return obj

    def extend(self, n, in_place=True):
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other

        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        self.npc_size = enlarge(self.npc_size)
        self.npc_state = enlarge(self.npc_state)
        self.npc_present_mask = enlarge(self.npc_present_mask)
        self.npc_types = enlarge(self.npc_types)
        self.npc_states = enlarge(self.npc_states)
        self.npc_present_masks = enlarge(self.npc_present_masks)
        self.spawn_controller.extend(n, in_place=True)
        return self

    def select_batch_elements(self, idx, in_place=True):
        if not in_place:
            return self.copy().select_batch_elements(idx, in_place=True)

        self.npc_size = self.npc_size[idx]
        self.npc_state = self.npc_state[idx]
        self.npc_present_mask = self.npc_present_mask[idx]
        self.npc_types = self.npc_types[idx]
        self.npc_states = self.npc_states[idx]
        self.npc_present_masks = self.npc_present_masks[idx]
        self.spawn_controller.select_batch_elements(idx, in_place=True)
        return self
