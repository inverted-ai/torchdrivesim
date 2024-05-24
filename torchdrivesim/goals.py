"""
Definitions of goal conditions. Currently, we support waypoint conditions.
"""
import copy
import dataclasses
from typing import List, Optional, Union, Dict

import torch
import torch.nn.functional as F
from torch import Tensor


TensorPerAgentType = Union[Tensor, Dict[str, Tensor]]


class WaypointGoal:
    """
    Waypoints can be given either per agent type or directly as tensors. The waypoint tensor can contain `M` waypoints
    for each collection of `N` waypoints that progressively get enabled. The provided mask should indicated which waypoints
    are padding elements. A waypoint is marked as succesful when the agent state reaches the center of the waypoint
    within some distance.

    Args:
        waypoints: a functor of BxAxNxMx2 waypoint tensors [x, y]
        mask: a functor of BxAxNxM boolean tensor indicating whether a given waypoint element is present and not a padding.
    """
    def __init__(self, waypoints: TensorPerAgentType, mask: Optional[TensorPerAgentType] = None):
        self.waypoints = waypoints
        self.mask = mask if mask is not None else self._default_mask()
        if isinstance(self.waypoints, dict):
            self.max_goal_idx = max([v.shape[2] for v in self.waypoints.values()])
        else:
            self.max_goal_idx = self.waypoints.shape[2]
        self.state = self._default_state()  # BxAx1


    def _default_mask(self) -> Tensor:
        if isinstance(self.waypoints, dict):
            return {k: torch.ones(*v.shape[:-1], dtype=torch.bool, device=v.device) for k,v in self.waypoints.items()}
        else:
            return torch.ones(*self.waypoints.shape[:-1], dtype=torch.bool, device=self.waypoints.device)

    def _default_state(self) -> Tensor:
        if isinstance(self.waypoints, dict):
            return {k: torch.zeros(*v.shape[:2] + (1, ), dtype=torch.long, device=v.device) for k,v in self.waypoints.items()}
        else:
            return torch.zeros(*self.waypoints.shape[:2] + (1, ), dtype=torch.long, device=self.waypoints.device)

    def get_masks(self):
        """
        Returns the waypoint mask according to the current state value.
        """
        if isinstance(self.waypoints, dict):
            return {k: torch.gather(v, 2, self.state[k][..., None].expand(-1, -1, -1, *v.shape[3:])).squeeze(2)
                    for k,v in self.mask.items()}
        else:
            return torch.gather(self.mask, 2, self.state[..., None].expand(-1, -1, -1, *self.mask.shape[3:])).squeeze(2)

    def get_waypoints(self):
        """
        Returns the waypoints according to the current state value.
        """
        if isinstance(self.waypoints, dict):
            return {k: torch.gather(v, 2, self.state[k][..., None, None].expand(-1, -1, -1, *v.shape[3:])).squeeze(2)
                    for k,v in self.waypoints.items()}
        else:
            return torch.gather(self.waypoints, 2, self.state[..., None, None]\
                .expand(-1, -1, -1, *self.waypoints.shape[3:])).squeeze(2)

    def copy(self):
        """
        Duplicates this object, allowing for independent subsequent execution.
        """
        other = self.__class__(
            waypoints=copy.deepcopy(self.waypoints), mask=copy.deepcopy(self.mask),
        )
        other.state = copy.deepcopy(self.state)
        return other

    def to(self, device: torch.device):
        """
        Modifies the object in-place, putting all tensors on the device provided.
        """
        if isinstance(self.waypoints, dict):
            self.waypoints = {k: v.to(device) for k, v in self.waypoints.items()}
        else:
            self.waypoints = self.waypoints.to(device)

        if isinstance(self.mask, dict):
            self.mask = {k: v.to(device) for k, v in self.mask.items()}
        else:
            self.mask = self.mask.to(device)

        if isinstance(self.state, dict):
            self.state = {k: v.to(device) for k, v in self.state.items()}
        else:
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
        if isinstance(self.waypoints, dict):
            self.waypoints = {k: enlarge(v) for k, v in self.waypoints.items()}
        else:
            self.waypoints = enlarge(self.waypoints)

        if isinstance(self.mask, dict):
            self.mask = {k: enlarge(v) for k, v in self.mask.items()}
        else:
            self.mask = enlarge(self.mask)

        if isinstance(self.state, dict):
            self.state = {k: enlarge(v) for k, v in self.state.items()}
        else:
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

        if isinstance(self.waypoints, dict):
            self.waypoints = {k: v[idx] for k, v in self.waypoints.items()}
        else:
            self.waypoints = self.waypoints[idx]

        if isinstance(self.mask, dict):
            self.mask = {k: v[idx] for k, v in self.mask.items()}
        else:
            self.mask = self.mask[idx]

        if isinstance(self.state, dict):
            self.state = {k: v[idx] for k, v in self.state.items()}
        else:
            self.state = self.state[idx]
        return self

    def step(self, agent_states: TensorPerAgentType, time: int = 0, threshold: float = 2.0) -> None:
        """
        Advances the state of the waypoints.
        """
        if isinstance(agent_states, dict):
            assert isinstance(self.waypoints, dict)
            assert sum([a.shape[1] for a in agent_states.values()]) == sum(w.shape[1] for w in self.waypoints.values())
            assert list(agent_states.keys()) == list(self.waypoints.keys())
            waypoints = self.get_waypoints()
            masks = self.get_masks()
            for agent_type, agent_state in agent_states.items():
                agent_overlap = self._agent_waypoint_overlap(agent_state[..., :2], waypoints[agent_type], threshold=threshold)
                agent_overlap = agent_overlap.any(dim=-1, keepdim=True).expand_as(agent_overlap)
                agent_overlap = torch.where(masks[agent_type], agent_overlap, True)
                self.mask[agent_type] = self._update_mask(self.state[agent_type], self.mask[agent_type],
                                                          agent_overlap.logical_not())
                self.state[agent_type] = self._advance_state(self.state[agent_type], agent_overlap)
        else:
            assert torch.is_tensor(self.waypoints)
            assert agent_states.shape[1] == self.waypoints.shape[1]
            waypoints = self.get_waypoints()
            masks = self.get_masks()
            agent_overlap = self._agent_waypoint_overlap(agent_states[..., :2], waypoints, threshold=threshold)
            agent_overlap = agent_overlap.any(dim=-1, keepdim=True).expand_as(agent_overlap)
            agent_overlap = torch.where(masks, agent_overlap, True)
            self.mask = self._update_mask(self.state, self.mask, agent_overlap.logical_not())
            self.state = self._advance_state(self.state, agent_overlap)

    def _update_mask(self, state, mask, new_mask):
        """
        Helper function to update the global masks on the current state with the `new_mask` values.

        Args:
            state: BxAx1 int tensor
            mask: BxAxNxM boolean tensor
            new_mask: BxAxM boolean tensor
        Returns:
            BxAxNxM boolean tensor updated with the `new_mask` values
        """
        return torch.scatter(mask, 2, state[..., None].expand(-1, -1, -1, mask.shape[-1]), new_mask.unsqueeze(2))

    def _advance_state(self, state, agent_overlap):
        """
        Helper function to advance the state when agents overlap with waypoints.

        Args:
            state: BxAx1 int tensor
            agent_overlap: BxAxM boolean tensor
        Returns:
            BxAxM boolean tensor indicating the state of the waypoints
        """
        return torch.clamp(state + agent_overlap.any(dim=-1, keepdim=True), min=0, max=self.max_goal_idx-1)

    def _agent_waypoint_overlap(self, agent_states: Tensor, waypoints: Tensor, threshold = 2.0) -> Tensor:
        """
        Helper function to detect if agent states overlap with the current waypoints.

        Args:
            agent_states: BxAx2 tensor of agents states - x,y
            waypoints: BxAxMx2 tensor of waypoint
        Returns:
            BxAxM boolean tensor indicating which agents overlap with the waypoints
        """
        agent_states = agent_states[..., None, :].expand_as(waypoints)
        dist = (((agent_states[..., 0] - waypoints[..., 0])**2) + ((agent_states[..., 1] - waypoints[..., 1])**2))**0.5
        return dist <= threshold
