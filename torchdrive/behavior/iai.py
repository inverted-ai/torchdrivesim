from typing import List, Optional

import torch
from torch import Tensor
from typing_extensions import Self

from torchdrive.behavior.common import InitializationFailedError
from torchdrive.simulator import NPCWrapper, SimulatorInterface, TensorPerAgentType, HomogeneousWrapper


def iai_initialize(location, agent_count, center=(0, 0)):
    import invertedai
    try:
        response = invertedai.api.initialize(
            location=location, agent_count=agent_count, location_of_interest=center

        )
    except invertedai.error.InvalidRequestError:
        raise InitializationFailedError()
    agent_attributes = torch.stack(
        [torch.tensor(at.tolist()) for at in response.agent_attributes], dim=-2
    )
    agent_states = torch.stack(
        [torch.tensor(st.tolist()) for st in response.agent_states], dim=-2
    )
    return agent_attributes, agent_states, response.recurrent_states


def iai_drive(location: str, agent_states: Tensor, agent_attributes: Tensor, recurrent_states: List):
    import invertedai
    from invertedai.common import AgentState, AgentAttributes, Point
    agent_attributes = [AgentAttributes(length=at[0], width=at[1], rear_axis_offset=at[2]) for at in agent_attributes]
    agent_states = [AgentState(center=Point(x=st[0], y=st[1]), orientation=st[2], speed=st[3]) for st in agent_states]
    response = invertedai.api.drive(
        location=location, agent_states=agent_states, agent_attributes=agent_attributes,
        recurrent_states=recurrent_states
    )
    agent_states = torch.stack(
        [torch.tensor(st.tolist()) for st in response.agent_states], dim=-2
    )
    return agent_states, response.recurrent_states


class IAIWrapper(NPCWrapper):
    """
    Uses IAI API to control NPCs, making a call every time step.
    Requires IAI_API_KEY to be set.
    This wrapper should be applied before any other agent-hiding wrappers.

    Args:
        simulator: existing simulator to wrap
        npc_mask: A functor of tensors with a single dimension of size A, indicating which agents to replay.
            The tensors can not have batch dimensions.
        recurrent_states: A functor of BxAxN tensors with NPC recurrent states, usually obtained from `iai_initialize`.
        rear_axis_offset: A BxAx1 tensor, concatenated across agent types, specifying the rear axis offset parameter
            used in the kinematic bicycle model. By default, a realistic value based on the vehicle length is used.
    """
    def __init__(self, simulator: SimulatorInterface, npc_mask: TensorPerAgentType,
                 recurrent_states: List[List], locations: List[str], rear_axis_offset: Optional[Tensor] = None):
        super().__init__(simulator=simulator, npc_mask=npc_mask)

        self._locations = locations
        self._npc_predictions = None
        self._recurrent_states = recurrent_states

        lenwid = HomogeneousWrapper(self.inner_simulator).get_agent_size()[..., :2]
        if rear_axis_offset is None:
            rear_axis_offset = 1.4 * torch.ones_like(lenwid[..., :1])  # TODO: use value proportional to length
        self._agent_attributes = torch.cat([lenwid, rear_axis_offset], dim=-1)

        # self.validate_agent_types()
        # self.validate_tensor_shapes()

    def _get_npc_predictions(self):
        states, recurrent = [], []
        agent_states = HomogeneousWrapper(self.inner_simulator).get_state()
        for i in range(self.batch_size):
            s, r = iai_drive(location=self._locations[i], agent_states=agent_states[i],
                             agent_attributes=self._agent_attributes[i], recurrent_states=self._recurrent_states[i])
            states.append(s)
            recurrent.append(r)
        states = torch.stack(states, dim=0).to(self.get_state().device)
        return states, recurrent

    def _npc_teleport_to(self):
        return self._npc_predictions

    def step(self, action):
        self._npc_predictions, self._recurrent_states = self._get_npc_predictions()  # TODO: run async after step
        super().step(action)
        self._npc_predictions = None

    def to(self, device) -> Self:
        super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, npc_mask=self.npc_mask, recurrent_states=self._recurrent_states,
                               locations=self._locations)
        return other

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.replay_states = other.across_agent_types(lambda x: x[idx], other.replay_states)
        other.present_masks = other.across_agent_types(lambda x: x[idx], other.present_masks)
        other._batch_size = len(idx)
        return other
    #
    # def validate_agent_types(self):
    #     assert list(self.npc_mask.keys()) == self.agent_types
    #     assert list(self.replay_states.keys()) == self.agent_types
    #     assert list(self.present_masks.keys()) == self.agent_types
    #
    # def validate_tensor_shapes(self):
    #     # check that tensors have the expected number of dimensions
    #     self.across_agent_types(lambda m: assert_equal(len(m.shape), 1), self.npc_mask)
    #     self.across_agent_types(lambda s: assert_equal(len(s.shape), 4), self.replay_states)
    #     self.across_agent_types(lambda m: assert_equal(len(m.shape), 3), self.present_masks)
    #
    #     # check that batch size is the same everywhere
    #     b = self.batch_size
    #     self.across_agent_types(lambda s: assert_equal(s.shape[0], b), self.replay_states)
    #     self.across_agent_types(lambda m: assert_equal(m.shape[0], b), self.present_masks)
    #
    #     # check that the number of agents in replay is the same as in underlying simulator
    #     check_counts = lambda i: lambda x, y: assert_equal(x.shape[i], y)
    #     self.across_agent_types(check_counts(0), self.npc_mask, self.inner_simulator.agent_count)
    #     self.across_agent_types(check_counts(-3), self.replay_states, self.inner_simulator.agent_count)
    #     self.across_agent_types(check_counts(-2), self.present_masks, self.inner_simulator.agent_count)