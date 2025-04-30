from typing import List, Optional, Dict

import random
import torch
from torch import Tensor

import invertedai as iai
from invertedai.common import AgentState, AgentProperties, Point, TrafficLightState, RecurrentState

from torchdrivesim.behavior.common import InitializationFailedError
from torchdrivesim.simulator import NPCController, Simulator, SpawnController
from torchdrivesim.traffic_lights import TrafficLightController


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
        if self.agent_type_names is None:
            self.agent_type_names = ['vehicle']
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
    
    def copy(self):
        """Create a deep copy of the controller."""
        other = self.__class__(
            npc_size=self.npc_size.clone(),
            npc_state=self.npc_state.clone(),
            npc_lr=self.npc_attribute[..., 2].clone(),
            location=self.location,
            drive_model_version=self.drive_model_version,
            npc_present_mask=self.npc_present_mask.clone() if self.npc_present_mask is not None else None,
            time=self.time,
            npc_types=self.npc_types.clone() if self.npc_types is not None else None,
            agent_type_names=self.agent_type_names.copy() if self.agent_type_names is not None else None,
            traffic_light_controller=self._traffic_light_controller,
            light_states_all_timesteps=self.light_states_all_timesteps,
            spawn_controller=self.spawn_controller.copy() if self.spawn_controller is not None else None
        )
        other.npc_attribute = self.npc_attribute.clone()
        other.recurrent_state = self.recurrent_state
        other.npc_state_shape = self.npc_state_shape
        return other

    def to(self, device):
        super().to(device)
        self.npc_size = self.npc_size.to(device)
        self.npc_state = self.npc_state.to(device)
        self.npc_present_mask = self.npc_present_mask.to(device)
        self.npc_types = self.npc_types.to(device)
        self.spawn_controller.to(device)
        return self
