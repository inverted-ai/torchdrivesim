"""
Definitions of traffic controls. Currently, we support traffic lights, stop signs, and yield signs.
"""
from typing import List, Optional

import torch
from torch import Tensor

from torchdrivesim._iou_utils import oriented_box_intersection_2d, box2corners_th, box2corners_with_rear_factor


class BaseTrafficControl:
    """
    Base traffic control class. All traffic controls are static rectangular stoplines with additional discrete state.
    States are advanced by calling `step` and can be set either from a given recorded history of states or
    by `compute_state` if current timestep exceeds the given recorded history.

    Args:
        pos: BxNx5 stoplines tensor [x, y, length/thickness, width, orientation]
        allowed_states: Allowed values of state, e.g. colors for traffic lights.
        replay_states: BxNxT tensor of states to replay, represented as indices into `allowed_states`. Default T=0.
        mask: BxN boolean tensor indicating whether a given traffic control element is present and not a padding.
    """
    def __init__(self, pos: Tensor, allowed_states: Optional[List[str]] = None,
                 replay_states: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        self.pos = pos
        self.allowed_states = allowed_states if allowed_states is not None else self._default_allowed_states()
        self.replay_states = replay_states if replay_states is not None else self._default_replay_states()
        self.mask = mask if mask is not None else self._default_mask()

        self.corners = box2corners_th(self.pos)
        corner_mask = self.mask.to(self.corners.dtype).reshape(self.mask.shape[0], self.mask.shape[1], 1, 1)
        self.corners = self.corners * corner_mask + (1 - corner_mask) * -1000  # Set masked bboxes far from center
        self.state = self._default_state()

    @classmethod
    def _default_allowed_states(cls) -> List[str]:
        return ['none']

    def _default_replay_states(self) -> Tensor:
        return torch.zeros(*self.pos.shape[:2] + (0,), dtype=torch.long, device=self.pos.device)

    def _default_mask(self) -> Tensor:
        return torch.ones(*self.pos.shape[:2], dtype=torch.bool, device=self.pos.device)

    def _default_state(self) -> Tensor:
        if self.replay_states.shape[-1] > 0:
            return self.replay_states[..., 0]
        else:
            return torch.zeros(*self.pos.shape[:2], dtype=torch.long, device=self.pos.device)

    @property
    def total_replay_time(self) -> int:
        return self.replay_states.shape[-1]

    def copy(self):
        """
        Duplicates this object, allowing for independent subsequent execution.
        """
        other = self.__class__(
            pos=self.pos.clone(), allowed_states=self.allowed_states.copy(),
            replay_states=self.replay_states.clone(), mask=self.mask.clone(),
        )
        other.state = self.state.clone()
        return other

    def to(self, device: torch.device):
        """
        Modifies the object in-place, putting all tensors on the device provided.
        """
        self.pos = self.pos.to(device)
        self.corners = self.corners.to(device)
        self.replay_states = self.replay_states.to(device)
        self.mask = self.mask.to(device)
        self.state = self.state.to(device)
        return self

    def extend(self, n: int, in_place: bool = True):
        """
        Multiplies the first batch dimension by the given number.
        This is equivalent to introducing extra batch dimension on the right and then flattening.
        """
        if not in_place:
            other = self.copy()
            other.extend(n, in_place=True)
            return other

        enlarge = lambda x: x.unsqueeze(1).expand(
            (x.shape[0], n) + x.shape[1:]
        ).reshape((n * x.shape[0],) + x.shape[1:])
        self.pos = enlarge(self.pos)
        self.corners = enlarge(self.corners)
        self.mask = enlarge(self.mask)
        self.replay_states = enlarge(self.replay_states)
        self.state = enlarge(self.state)
        return self

    def select_batch_elements(self, idx: Tensor, in_place: bool = True):
        """
        Picks selected elements of the batch.
        The input is a tensor of indices into the batch dimension.
        """
        if not in_place:
            other = self.copy()
            other.select_batch_elements(idx, in_place=True)
            return other

        self.pos = self.pos[idx]
        self.corners = self.corners[idx]
        self.mask = self.mask[idx]
        self.replay_states = self.replay_states[idx]
        self.state = self.state[idx]
        return self

    def set_state(self, state: Tensor) -> None:
        """
        Sets control state tensor.
        """
        self.state = state

    def compute_state(self, time: int) -> Tensor:
        """
        Computes the state at a given time index. By default, it repeats the current state. Ignores replay states.
        """
        return self.state

    def step(self, time: int) -> None:
        """
        Advances the state of the control given a time step. If the time value is within the range of replayable states
        then this state will be used otherwise the function `compute_state(time)` will be used to calculate a new state.
        """
        if time < self.total_replay_time:
            self.set_state(self.replay_states[..., time])
        else:
            state = self.compute_state(time)
            self.set_state(state)

    def compute_violation(self, agent_state: Tensor) -> Tensor:
        """
        Given a collection of agents, computes which agents violate the traffic control.
        As deciding violations is typically complex, this function provides an approximate answer only.
        The base class reports no violations.

        Args:
            agent_state: BxAx5 tensor of agents states - x,y,length,width,orientation
        Returns:
            BxA boolean tensor indicating which agents violated the traffic control
        """
        return torch.zeros(agent_state.shape[0], agent_state.shape[1], dtype=torch.bool, device=agent_state.device)


class TrafficLightControl(BaseTrafficControl):
    """
    Traffic lights, with default allowed states of ['red', 'yellow', 'green'].
    An agent is regarded to violate the traffic light if the light is red and the agent
    bounding box substantially overlaps with the stop line.
    """
    violation_rear_factor = 0.1

    @classmethod
    def _default_allowed_states(cls) -> List[str]:
        return ['red', 'yellow', 'green']

    def compute_violation(self, agent_state: Tensor) -> Tensor:
        batch_size, num_agents = agent_state.shape[0], agent_state.shape[1]
        num_lights = self.corners.shape[1]
        if batch_size > 0 and num_agents > 0 and num_lights > 0:
            agent_corners = box2corners_with_rear_factor(agent_state, rear_factor=self.violation_rear_factor)
            agent_corners = agent_corners.reshape(-1, 1, 4, 2).expand(-1, num_lights, -1, -1)
            control_corners = self.corners.unsqueeze(1).expand(-1, num_agents, -1, -1, -1).reshape(-1, num_lights, 4, 2)
            overlap = (oriented_box_intersection_2d(agent_corners, control_corners)[0] > 0)
            is_red_light = (self.state.to(overlap.device) == self.allowed_states.index('red'))
            is_red_light = is_red_light.unsqueeze(1).expand(-1, num_agents, -1).reshape(-1, num_lights)
            red_light_violations = torch.logical_and(overlap, is_red_light)
            red_light_violations = red_light_violations.any(dim=-1).reshape(batch_size, num_agents)
        else:
            red_light_violations = torch.zeros(batch_size, num_agents, dtype=torch.bool, device=agent_state.device)
        return red_light_violations


class YieldControl(BaseTrafficControl):
    """
    Yield sign, indicating that cross traffic has priority.
    Violations are not computed.
    """
    pass


class StopSignControl(BaseTrafficControl):
    """
    Stop sign, indicating the vehicle should stop and yield to cross traffic.
    Violations are not computed.
    """
    pass
