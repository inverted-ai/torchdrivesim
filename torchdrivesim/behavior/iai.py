from typing import List, Optional, Dict
from enum import Enum

import os
import math
import pickle
import random
import torch
from torch import Tensor
from typing_extensions import Self

from torchdrivesim.behavior.common import InitializationFailedError, LocationInfoFailedError
from torchdrivesim.simulator import NPCWrapper, SimulatorInterface, TensorPerAgentType, HomogeneousWrapper
from torchdrivesim.utils import TrafficLightState

IAI_LOCATION_INFO_DIR = "location_info"


def iai_conditional_initialize(location, agent_count, agent_attributes=None, agent_states=None, recurrent_states=None, center=(0, 0), traffic_light_state_history=None):
    import invertedai
    try:
        INITIALIZE_FOV = 120
        conditional_agent_attributes = []
        conditional_agent_states = []
        conditional_recurrent_states = []
        outside_agent_states = []
        outside_agent_attributes = []
        outside_recurrent_states = []

        for i in range(len(agent_states)):
            agent_state = agent_states[i]
            dist = math.dist(center, (agent_state.center.x, agent_state.center.y))
            if dist < INITIALIZE_FOV:
                conditional_agent_states.append(agent_state)
                conditional_agent_attributes.append(agent_attributes[i])
                conditional_recurrent_states.append(recurrent_states[i])
            else:
                outside_agent_states.append(agent_state)
                outside_agent_attributes.append(agent_attributes[i])
                outside_recurrent_states.append(recurrent_states[i])

        agent_count -= len(conditional_agent_states)
        if agent_count > 0:
            try:
                seed = random.randint(1, 10000)
                response = invertedai.api.initialize(
                    location=location,
                    agent_attributes=conditional_agent_attributes,
                    states_history=[conditional_agent_states],
                    agent_count=agent_count,
                    location_of_interest=center,
                    traffic_light_state_history=traffic_light_state_history,
                    random_seed = seed
                )
                agent_attribute_list = response.agent_attributes + outside_agent_attributes
                agent_state_list = response.agent_states + outside_agent_states
                recurrent_state_list = response.recurrent_states + outside_recurrent_states
            except Exception:
                agent_attribute_list = agent_attributes
                agent_state_list = agent_states
                recurrent_state_list = recurrent_states
        else:
            agent_attribute_list = agent_attributes
            agent_state_list = agent_states
            recurrent_state_list = recurrent_states

    except invertedai.error.InvalidRequestError:
        raise InitializationFailedError()

    agent_attributes = torch.stack(
        [torch.tensor(at.tolist()[:3]) for at in agent_attribute_list], dim=-2
    )
    agent_states = torch.stack(
        [torch.tensor(st.tolist()) for st in agent_state_list], dim=-2
    )
    return agent_attributes, agent_states, recurrent_state_list


def iai_drive(location: str, agent_states: Tensor, agent_attributes: Tensor, recurrent_states: List, traffic_lights_states: Dict = None):
    import invertedai
    from invertedai.common import AgentState, AgentAttributes, Point
    try:
        agent_attributes = [AgentAttributes(length=at[0], width=at[1], rear_axis_offset=at[2]) for at in agent_attributes]
        agent_states = [AgentState(center=Point(x=st[0], y=st[1]), orientation=st[2], speed=st[3]) for st in agent_states]
        seed = random.randint(1, 10000)
        response = invertedai.api.drive(
            location=location, agent_states=agent_states, agent_attributes=agent_attributes,
            recurrent_states=recurrent_states,
            traffic_lights_states=traffic_lights_states,
            random_seed=seed
        )
        agent_states = torch.stack(
            [torch.tensor(st.tolist()) for st in response.agent_states], dim=-2
        )
    except Exception as e:
        print(e)
        print(location)
        print(agent_attributes)
        print(agent_states)
        raise e
    return agent_states, response.recurrent_states


def iai_location_info(location: str):
    import invertedai
    try:
        response = invertedai.api.location_info(
            location=location
        )
    except invertedai.error.InvalidRequestError:
        raise LocationInfoFailedError()
    return response


def iai_location_info_from_local(location: str):
    with open(f"{IAI_LOCATION_INFO_DIR}/{location}.pkl", "rb") as f:
        response = pickle.load(f)
    return response


def cache_iai_location_info(locations: List[str]):
    import invertedai
    try:
        if not os.path.isdir(IAI_LOCATION_INFO_DIR):
            os.mkdir(IAI_LOCATION_INFO_DIR)
        for location in locations:
            response = invertedai.api.location_info(
                location=location
            )
            with open(f"{IAI_LOCATION_INFO_DIR}/{location}.pkl", "wb") as f:
                pickle.dump(response, f)
    except invertedai.error.InvalidRequestError:
        raise LocationInfoFailedError()


def get_static_actors(location_info_response):
    static_actors = dict()
    for actor in location_info_response.static_actors:
        static_actors[actor.actor_id] = {'pos': torch.Tensor([actor.center.x, actor.center.y, actor.length, actor.width, actor.orientation]),
                                         'agent_type': actor.agent_type}
    return static_actors


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
                 recurrent_states: List[List], locations: List[str], rear_axis_offset: Optional[Tensor] = None, car_sequences: Optional[Dict[int, List[List[float]]]] = None):
        super().__init__(simulator=simulator, npc_mask=npc_mask)

        self._locations = locations
        self._npc_predictions = None
        self._recurrent_states = recurrent_states
        self._car_sequences = car_sequences
        self._iai_timestep = 0

        lenwid = HomogeneousWrapper(self.inner_simulator).get_agent_size()[..., :2]
        if rear_axis_offset is None:
            rear_axis_offset = 1.4 * torch.ones_like(lenwid[..., :1])  # TODO: use value proportional to length
        self._agent_attributes = torch.cat([lenwid, rear_axis_offset], dim=-1)

    def _get_npc_predictions(self):
        states, recurrent = [], []
        agent_states = HomogeneousWrapper(self.inner_simulator).get_state()
        traffic_controls = self.get_innermost_simulator().get_traffic_controls()
        if (traffic_controls is not None) and ("traffic_light" in traffic_controls):
            traffic_light_control = traffic_controls["traffic_light"]
            traffic_lights_states = dict(zip(traffic_light_control.ids, traffic_light_control.state.squeeze()))
            traffic_lights_states = {k:TrafficLightState(traffic_light_control.allowed_states[int(traffic_lights_states[k])]) for k in traffic_lights_states}
        else:
            traffic_lights_states = [None]

        for i in range(self.batch_size):
            s, r = iai_drive(location=self._locations[i], agent_states=agent_states[i],
                             agent_attributes=self._agent_attributes[i], recurrent_states=self._recurrent_states[i],
                             traffic_lights_states=traffic_lights_states)
            if self._car_sequences is not None:
                for agent_idx in self._car_sequences:
                    if self._iai_timestep < len(self._car_sequences[agent_idx]):
                        s[agent_idx] = torch.Tensor(self._car_sequences[agent_idx][self._iai_timestep]).cuda()
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
        self._iai_timestep += 1

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
