from typing import List, Optional, Dict

import random
import torch
from torch import Tensor

from invertedai.common import TrafficLightState, RecurrentState

from torchdrivesim.behavior.common import InitializationFailedError


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
