"""
Definitions of goal conditions. Currently, we support waypoint conditions.
"""
import copy
from typing import List, Optional, Union, Dict

import torch
from torch import Tensor


class WaypointGoal:
    """
    The waypoint tensor can contain `M` waypoints
    for each collection of `N` waypoints that progressively get enabled. The provided mask should indicated which waypoints
    are padding elements. A waypoint is marked as succesful when the agent state reaches the center of the waypoint
    within some distance.

    Args:
        waypoints: a BxAxNxMx2 tensor of waypoint [x, y]
        mask: a BxAxNxM boolean tensor indicating whether a given waypoint element is present and not a padding.
    """
    def __init__(self, waypoints: Tensor, mask: Optional[Tensor] = None):
        self.waypoints = waypoints
        self.mask = mask if mask is not None else self._default_mask()
        self.max_goal_idx = self.waypoints.shape[2]
        self.state = self._default_state()  # BxAx1


    def _default_mask(self) -> Tensor:
        return torch.ones(*self.waypoints.shape[:-1], dtype=torch.bool, device=self.waypoints.device)

    def _default_state(self) -> Tensor:
        return torch.zeros(*self.waypoints.shape[:2] + (1, ), dtype=torch.long, device=self.waypoints.device)

    def get_masks(self):
        """
        Returns the waypoint mask according to the current state value.
        """
        return torch.gather(self.mask, 2, self.state[..., None].expand(-1, -1, -1, *self.mask.shape[3:])).squeeze(2)

    def get_waypoints(self):
        """
        Returns the waypoints according to the current state value.
        """
        return torch.gather(self.waypoints, 2, self.state[..., None, None].expand(-1, -1, -1, *self.waypoints.shape[3:])).squeeze(2)

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
        self.waypoints = self.waypoints.to(device)
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
        self.waypoints = enlarge(self.waypoints)
        self.mask = enlarge(self.mask)
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

        self.waypoints = self.waypoints[idx]
        self.mask = self.mask[idx]
        self.state = self.state[idx]
        return self

    def step(self, agent_states: Tensor, time: int = 0, threshold: float = 2.0) -> None:
        """
        Advances the state of the waypoints.
        """
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
