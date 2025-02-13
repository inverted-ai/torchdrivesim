import os
from typing import Optional, List
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torchdrivesim.behavior.common import InitializationFailedError
from torchdrivesim.simulator import NPCWrapper, SimulatorInterface, NPCController, Simulator
from torchdrivesim.utils import assert_equal


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


class ReplayWrapper(NPCWrapper):
    """
    Performs log replay for a subset of agents based on their recorded trajectories.
    The log has a finite length T, after which the replay will loop back from the beginning.

    Args:
        simulator: existing simulator to wrap
        npc_mask: A functor of tensors with a single dimension of size A, indicating which agents to replay.
            The tensors can not have batch dimensions.
        agent_states: a functor of BxAxTxSt tensors with states to replay across time,
            which should be padded with arbitrary values for non-replay agents
        present_masks: indicates when replay agents appear and disappear; by default they're all present at all times
        time: initial index into the time dimension for replay, incremented at every step
        replay_mask: A tensor of shape BxA indicating which agents are replayed, i.e. their actions are ignored
    """
    def __init__(self, simulator: SimulatorInterface, npc_mask: torch.Tensor,
                 agent_states: torch.Tensor, present_masks: Optional[torch.Tensor] = None, time: int = 0,
                 replay_mask: Optional[torch.Tensor] = None):
        super().__init__(simulator=simulator, npc_mask=npc_mask)

        # TODO: add time dimension to replay mask
        if replay_mask is None:
            replay_mask = npc_mask.unsqueeze(0).expand((self.batch_size,) + npc_mask.shape)

        self.replay_states = agent_states
        self.present_masks = present_masks
        self.replay_mask = replay_mask
        self.time = time
        self.max_time_step = agent_states.shape[-2]

        if self.present_masks is None:
            # by default all replay agents are always present
            self.present_masks = torch.ones_like(self.replay_states[..., 0], dtype=torch.bool),

        self.validate_tensor_shapes()

    def _npc_teleport_to(self):
        current_replay_state = self.replay_states[..., self.time, :]
        return current_replay_state

    def _update_npc_present_mask(self):
        return self.present_masks[..., self.time]

    def step(self, action):
        self.time += 1
        # reset time if needed
        if self.time == self.max_time_step:
            self.time = 0
        super().step(action)
        updated_state = self.replay_states[..., self.time, :].where(self.replay_mask.unsqueeze(-1), self.inner_simulator.get_state())
        self.inner_simulator.set_state(updated_state)
        updated_present_mask = self.present_masks[..., self.time].where(self.replay_mask, self.inner_simulator.get_present_mask())
        self.inner_simulator.update_present_mask(updated_present_mask)

    def to(self, device) -> Self:
        super().to(device)
        self.replay_states = self.replay_states.to(device)
        self.present_masks = self.present_masks.to(device)
        return self

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, npc_mask=self.npc_mask, agent_states=self.replay_states,
                               present_masks=self.present_masks, time=self.time)
        return other

    def extend(self, n, in_place=True):
        if not in_place:
            return super().extend(n, in_place=in_place)

        self.inner_simulator.extend(n)

        enlarge = lambda x: x.unsqueeze(1).expand((x.shape[0], n) + x.shape[1:]).reshape((n * x.shape[0],) + x.shape[1:])
        self.replay_states = enlarge(self.replay_states)
        self.present_masks = enlarge(self.present_masks)
        self.replay_mask = enlarge(self.replay_mask)
        return self

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.replay_states = other.replay_states[idx]
        other.present_masks = other.present_masks[idx]
        other.replay_mask = other.replay_mask[idx]
        other._batch_size = len(idx)
        return other

    def validate_tensor_shapes(self):
        # check that tensors have the expected number of dimensions
        assert_equal(len(self.npc_mask.shape), 1)
        assert_equal(len(self.replay_states.shape), 4)
        assert_equal(len(self.present_masks.shape), 3)

        # check that batch size is the same everywhere
        b = self.batch_size
        assert_equal(self.replay_states.shape[0], b)
        assert_equal(self.present_masks.shape[0], b)

        # check that the number of agents in replay is the same as in underlying simulator
        check_counts = lambda i: lambda x, y: assert_equal(x.shape[i], y)
        assert_equal(self.npc_mask.shape[0], self.inner_simulator.agent_count)
        assert_equal(self.replay_states.shape[1], self.inner_simulator.agent_count)
        assert_equal(self.present_masks.shape[1], self.inner_simulator.agent_count)


class ReplayController(NPCController):
    def __init__(self, npc_size, npc_states, npc_present_masks: Optional[torch.Tensor] = None, time: int = 0,
                 npc_types: Optional[Tensor] = None, agent_type_names: Optional[List[str]] = None):
        self.time = time
        self.npc_states = npc_states
        self.npc_present_masks = npc_present_masks
        if self.npc_present_masks is None:
            self.npc_present_masks = torch.ones_like(self.npc_states[..., 0], dtype=torch.bool)
        super().__init__(npc_size, self.npc_states[..., self.time, :], self.npc_present_masks[..., self.time], npc_types, agent_type_names)

    def advance_npcs(self, simulator: Simulator) -> None:
        self.time += 1
        if self.time == self.npc_states.shape[-2]:
            self.time = 0
        self.npc_state = self.npc_states[..., self.time, :]
        self.npc_present_mask = self.npc_present_masks[..., self.time]
        return None

    def to(self, device):
        self.npc_size = self.npc_size.to(device)
        self.npc_state = self.npc_state.to(device)
        self.npc_present_mask = self.npc_present_mask.to(device)
        self.npc_types = self.npc_types.to(device)
        self.npc_states = self.npc_states.to(device)
        self.npc_present_masks = self.npc_present_masks.to(device)
        return self
    
    def copy(self):
        return self.__class__(self.npc_size, self.npc_states, self.npc_present_masks, self.time, self.npc_types, self.agent_type_names)
    
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
        return self