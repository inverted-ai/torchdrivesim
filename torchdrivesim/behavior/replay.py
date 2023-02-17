import os
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch

from torchdrivesim.behavior.common import InitializationFailedError
from torchdrivesim.simulator import NPCWrapper, SimulatorInterface, TensorPerAgentType
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
    """
    def __init__(self, simulator: SimulatorInterface, npc_mask: TensorPerAgentType,
                 agent_states: TensorPerAgentType, present_masks: TensorPerAgentType = None, time: int = 0):
        super().__init__(simulator=simulator, npc_mask=npc_mask)

        self.replay_states = agent_states
        self.present_masks = present_masks
        self.time = time
        self.max_time_step = max(self.across_agent_types(lambda x: x.shape[-2], agent_states).values())

        if self.present_masks is None:
            # by default all replay agents are always present
            self.present_masks = self.across_agent_types(
                lambda states: torch.ones_like(states[..., 0], dtype=torch.bool), self.replay_states
            )

        self.validate_agent_types()
        self.validate_tensor_shapes()

    def _npc_teleport_to(self):
        current_replay_state = self.across_agent_types(
            lambda states: states[..., self.time, :], self.replay_states
        )
        return current_replay_state

    def _update_npc_present_mask(self):
        return self.across_agent_types(lambda pm: pm[..., self.time], self.present_masks)

    def step(self, action):
        self.time += 1
        # reset time if needed
        if self.time == self.max_time_step:
            self.time = 0
        super().step(action)

    def to(self, device) -> Self:
        super().to(device)
        self.replay_states = self.agent_functor.to_device(self.replay_states, device)
        self.present_masks = self.agent_functor.to_device(self.present_masks, device)

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
        self.replay_states = self.across_agent_types(enlarge, self.replay_states)
        self.present_masks = self.across_agent_types(enlarge, self.present_masks)

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.replay_states = other.across_agent_types(lambda x: x[idx], other.replay_states)
        other.present_masks = other.across_agent_types(lambda x: x[idx], other.present_masks)
        other._batch_size = len(idx)
        return other

    def validate_agent_types(self):
        assert list(self.npc_mask.keys()) == self.agent_types
        assert list(self.replay_states.keys()) == self.agent_types
        assert list(self.present_masks.keys()) == self.agent_types

    def validate_tensor_shapes(self):
        # check that tensors have the expected number of dimensions
        self.across_agent_types(lambda m: assert_equal(len(m.shape), 1), self.npc_mask)
        self.across_agent_types(lambda s: assert_equal(len(s.shape), 4), self.replay_states)
        self.across_agent_types(lambda m: assert_equal(len(m.shape), 3), self.present_masks)

        # check that batch size is the same everywhere
        b = self.batch_size
        self.across_agent_types(lambda s: assert_equal(s.shape[0], b), self.replay_states)
        self.across_agent_types(lambda m: assert_equal(m.shape[0], b), self.present_masks)

        # check that the number of agents in replay is the same as in underlying simulator
        check_counts = lambda i: lambda x, y: assert_equal(x.shape[i], y)
        self.across_agent_types(check_counts(0), self.npc_mask, self.inner_simulator.agent_count)
        self.across_agent_types(check_counts(-3), self.replay_states, self.inner_simulator.agent_count)
        self.across_agent_types(check_counts(-2), self.present_masks, self.inner_simulator.agent_count)
