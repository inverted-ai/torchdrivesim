from typing import List, Optional, Dict

import random
import torch
from torch import Tensor
from typing_extensions import Self

import invertedai as iai
from invertedai.common import AgentState, AgentProperties, Point, TrafficLightState, RecurrentState

from torchdrivesim.behavior.common import InitializationFailedError
from torchdrivesim.simulator import NPCController, NPCWrapper, Simulator, SimulatorInterface, SpawnController
from torchdrivesim.traffic_lights import TrafficLightController, current_light_state_tensor_from_controller


def unpack_attributes(attributes) -> torch.Tensor:
    return torch.tensor([attributes.length, attributes.width, attributes.rear_axis_offset])

def agent_attributes_to_basic_agent_properties(agent_attributes: Tensor) -> Dict[str, float]:
    return {
        'length': agent_attributes[0],
        'width': agent_attributes[1],
        'rear_axis_offset': agent_attributes[2],
    }

def agent_properties_to_agent_attributes(agent_properties: Dict[str, float]) -> Tensor:
    return torch.Tensor([agent_properties['length'], agent_properties['width'], agent_properties['rear_axis_offset']])

def iai_initialize(location, agent_count, center=(0, 0), traffic_light_state_history: Optional[List[Dict[int, TrafficLightState]]] = None) -> (Tensor, Tensor, List[RecurrentState]):
    import invertedai
    try:
        seed = random.randint(1, 10000)
        response = invertedai.api.initialize(
            location=location, agent_count=agent_count, location_of_interest=center,
            traffic_light_state_history=traffic_light_state_history,
            random_seed=seed,
        )
    except invertedai.error.InvalidRequestError:
        raise InitializationFailedError()
    agent_properties_dict = [dict(ap) for ap in response.agent_properties]
    agent_properties = [agent_properties_to_agent_attributes(ap) for ap in agent_properties_dict]
    agent_attributes = torch.stack(agent_properties, dim=-2)
    agent_states = torch.stack(
        [torch.tensor(st.tolist()) for st in response.agent_states], dim=-2
    )
    return agent_attributes, agent_states, response.recurrent_states


def iai_drive(location: str, agent_states: Tensor, agent_attributes: Tensor, recurrent_states: List[RecurrentState], traffic_lights_states: Optional[Dict[int, TrafficLightState]] = None) -> (Tensor, List[RecurrentState]):
    import invertedai
    from invertedai.common import AgentState, AgentProperties, Point
    agent_properties = [AgentProperties(**agent_attributes_to_basic_agent_properties(at)) for at in agent_attributes]
    agent_states = [AgentState(center=Point(x=st[0], y=st[1]), orientation=st[2], speed=st[3]) for st in agent_states]
    seed = random.randint(1, 10000)
    response = invertedai.api.drive(
        location=location, agent_states=agent_states, agent_properties=agent_properties,
        recurrent_states=recurrent_states,
        traffic_lights_states=traffic_lights_states,
        random_seed=seed
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
    def __init__(self, simulator: SimulatorInterface, npc_mask: torch.Tensor,
                 recurrent_states: List[List], locations: List[str], rear_axis_offset: Optional[Tensor] = None,
                 replay_states: Optional[Tensor] = None, replay_mask: Optional[Tensor] = None,
                 traffic_light_controller: Optional[TrafficLightController] = None, traffic_light_ids: Optional[List[int]] = None):
        super().__init__(simulator=simulator, npc_mask=npc_mask)

        self._locations = locations
        self._npc_predictions = None
        self._recurrent_states = recurrent_states
        self._replay_states = replay_states
        self._replay_mask = replay_mask
        self._replay_timestep = 0
        self._traffic_light_controller = traffic_light_controller
        self._traffic_light_ids = traffic_light_ids
        assert (self._traffic_light_controller is None) == (self._traffic_light_ids is None), \
                "Both _traffic_light_controller and _traffic_light_ids should be either None or not None"

        lenwid = self.inner_simulator.get_agent_size()[..., :2]
        if rear_axis_offset is None:
            rear_axis_offset = 1.4 * torch.ones_like(lenwid[..., :1])  # TODO: use value proportional to length
        self._agent_attributes = torch.cat([lenwid, rear_axis_offset], dim=-1)

    def _get_npc_predictions(self):
        states, recurrent = [], []
        agent_states = self.inner_simulator.get_state()
        for i in range(self.batch_size):
            s, r = iai_drive(location=self._locations[i], agent_states=agent_states[i],
                             agent_attributes=self._agent_attributes[i], recurrent_states=self._recurrent_states[i],
                             traffic_lights_states=self._traffic_light_controller.current_state_with_name if self._traffic_light_controller is not None else None)

            if self._replay_states is not None:
                if self._replay_timestep < self._replay_states.shape[2]:
                    s[self._replay_mask[i, :, self._replay_timestep]] = self._replay_states[i, self._replay_mask[i, :, self._replay_timestep], self._replay_timestep, :]

            states.append(s)
            recurrent.append(r)
        states = torch.stack(states, dim=0).to(self.get_state().device)
        return states, recurrent

    def _npc_teleport_to(self):
        return self._npc_predictions

    def step(self, action):
        self._npc_predictions, self._recurrent_states = self._get_npc_predictions()  # TODO: run async after step
        if self._traffic_light_controller is not None:
            self._traffic_light_controller.tick(0.1)
            self.get_innermost_simulator().traffic_controls['traffic_light'].set_state(
                current_light_state_tensor_from_controller(
                    self._traffic_light_controller, self._traffic_light_ids).unsqueeze(0).expand(self.batch_size, -1).to(self.get_state().device)
            )
        super().step(action)
        self._npc_predictions = None
        self._replay_timestep += 1

    def to(self, device) -> Self:
        super().to(device)

    def copy(self):
        inner_copy = self.inner_simulator.copy()
        other = self.__class__(inner_copy, npc_mask=self.npc_mask, recurrent_states=self._recurrent_states,
                               locations=self._locations)
        return other

    def select_batch_elements(self, idx, in_place=True):
        other = super().select_batch_elements(idx, in_place=in_place)
        other.replay_states = other.replay_states[idx]
        other.present_masks = other.present_masks[idx]
        other._batch_size = len(idx)
        return other


class IAINPCController(NPCController):
    def __init__(self, npc_size, npc_state, npc_lr, location, drive_model_version: Optional[str] = None, npc_present_mask: Optional[torch.Tensor] = None, time: int = 0,
                 npc_types: Optional[Tensor] = None, agent_type_names: Optional[List[str]] = None, traffic_light_controller: Optional[TrafficLightController] = None,
                 light_states_all_timesteps: Optional[List[Dict[int, TrafficLightState]]] = None, spawn_controller: Optional[SpawnController] = None):
        """
            Calls IAI API to predict the next states of NPCs, to make the NPCs aware of ego, also passes ego states during the call, but removes it afterwards.
            npc_size: BxAx2 tensor with the length and width of agents
            npc_state: BxAx4 tensor with the states of agents
            npc_lr: BxA tensor with the rear axis offsets of agents
            npc_present_mask: BxA boolean tensor with the presence masks of agents
            npc_types: BxA tensor with the types of agents
        """
        super().__init__(npc_size, npc_state, npc_present_mask, npc_types, agent_type_names, spawn_controller)
        self.location = location
        self.agent_type_names = agent_type_names
        self.npc_attribute = torch.cat([npc_size, npc_lr.unsqueeze(-1)], dim=-1) # npc_attribute is of shape BxAx3
        self.recurrent_state = None
        self.drive_model_version = drive_model_version
        self.npc_state_shape = self.npc_state.shape
        self._traffic_light_controller = traffic_light_controller
        self.light_states_all_timesteps = light_states_all_timesteps
        self.time = time


    def advance_npcs(self, simulator: Simulator) -> None:
        ego_state = simulator.get_state()
        ego_size = simulator.get_agent_size()
        ego_type = simulator.get_agent_type()

        def get_iai_agent_type(npc_type):
            agent_type = self.agent_type_names[npc_type]
            if agent_type != "pedestrian":
                return "car"
            return agent_type

        def iai_drive(location, agent_states, agent_attributes, recurrent_states, drive_model_version, present_mask):
            agent_states = agent_states[:, present_mask[0], :]
            agent_attributes = agent_attributes[:, present_mask[0], :]
            npc_types = self.npc_types[:, present_mask[0]]

            if agent_states.shape[-2] == 0 or agent_attributes.shape[-2] == 0:
                return agent_states, []

            agent_properties = [AgentProperties(length=agent_attributes[0][i][0],
                                                width=agent_attributes[0][i][1],
                                                rear_axis_offset=agent_attributes[0][i][2] if not agent_attributes[0][i][2].isnan() else 0,
                                                agent_type=get_iai_agent_type(npc_types[0][i])) for i in range(agent_attributes.shape[-2])]
            agent_states = [AgentState(center=Point(x=agent_states[0][i][0], y=agent_states[0][i][1]),
                                       orientation=agent_states[0][i][2],
                                       speed=agent_states[0][i][3]) for i in range(agent_states.shape[-2])]

            # insert ego vehicle to API payload
            agent_properties.insert(0, AgentProperties(length=ego_size[0, 0, 0],
                                                    width=ego_size[0, 0, 1],
                                                    rear_axis_offset=ego_size[0, 0, 0].item() / 4,
                                                    agent_type=get_iai_agent_type(ego_type[0, 0])))
            agent_states.insert(0, (AgentState(center=Point(x=ego_state[0, 0, 0], y=ego_state[0, 0, 1]),
                                           orientation=ego_state[0, 0, 2],
                                           speed=ego_state[0, 0, 3])))

            seed = random.randint(1, 10000)

            if self.light_states_all_timesteps is not None:
                traffic_light_states = self.light_states_all_timesteps[self.time]
            elif self._traffic_light_controller is not None:
                self._traffic_light_controller.tick(0.1)
                traffic_light_states = self._traffic_light_controller.current_state_with_name
            else:
                traffic_light_states = None
            if len(agent_states) < 100:
                response = iai.drive(location=location, agent_states=agent_states, agent_properties=agent_properties,
                                            recurrent_states=recurrent_states, random_seed=seed, api_model_version=drive_model_version, traffic_lights_states=traffic_light_states)
            else:
                response = iai.large_drive(location=location, agent_states=agent_states, agent_properties=agent_properties,
                                            recurrent_states=recurrent_states, random_seed=seed, api_model_version=drive_model_version, traffic_lights_states=traffic_light_states)

            agent_states = torch.stack([torch.tensor(st.tolist()) for st in response.agent_states[1:]], dim=-2)
            return agent_states, response.recurrent_states

        self.time += 1
        predicted_npc_state, predicted_npc_recurrent_state = iai_drive(location=self.location, agent_states=self.npc_state, agent_attributes=self.npc_attribute,
                                                                       recurrent_states=self.recurrent_state, drive_model_version=self.drive_model_version, present_mask=self.npc_present_mask)
        self.npc_state = predicted_npc_state.clone().detach().unsqueeze(0).to(self.npc_state.device)
        reconstructed_state = torch.full(self.npc_state_shape, 0.0).to(self.npc_state.device)
        reconstructed_state[:, self.npc_present_mask[0], :] = self.npc_state
        self.npc_state = reconstructed_state
        self.recurrent_state = predicted_npc_recurrent_state
        self.spawn_despawn_npcs(simulator)
    
    def to(self, device):
        super().to(device)
        self.npc_size = self.npc_size.to(device)
        self.npc_state = self.npc_state.to(device)
        self.npc_present_mask = self.npc_present_mask.to(device)
        self.npc_types = self.npc_types.to(device)
        self.spawn_controller.to(device)
        return self
