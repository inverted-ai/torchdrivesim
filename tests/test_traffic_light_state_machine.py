import os
import pytest
from torchdrivesim.traffic_lights import TrafficLightStateMachine

@pytest.fixture
def traffic_light_state_machine():
    state_machine_file_path = os.path.join(os.path.dirname(__file__), "resources", "traffic_lights", 'intersection_1.json')
    return TrafficLightStateMachine(state_machine_file_path)

def test_reset(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.reset()
    assert traffic_light_state_machine.time_remaining >= 1

def test_set_to(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(2, time_remaining=3)
    assert traffic_light_state_machine.time_remaining == 3
    assert traffic_light_state_machine.current_state ==  {
            "actor_states": {
                "4411": "green",
                "3411": "red",
                "4399": "yellow",
                "3399": "yellow"
            },
            "state": "2",
            "duration": 5,
            "next_state": "3"
        }
    traffic_light_state_machine.set_to(2, time_remaining=1)
    assert traffic_light_state_machine.time_remaining == 1
    assert traffic_light_state_machine.current_state == {
            "actor_states": {
                "4411": "green",
                "3411": "red",
                "4399": "yellow",
                "3399": "yellow"
            },
            "state": "2",
            "duration": 5,
            "next_state": "3"
        }


def test_tick(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(0, 1)
    traffic_light_state_machine.tick(9)
    assert traffic_light_state_machine.time_remaining <= 0.1
    assert traffic_light_state_machine.current_state == {
            "actor_states": {
                "4411": "red",
                "3411": "red",
                "4399": "red",
                "3399": "red"
            },
            "state": "0",
            "duration": 10,
            "next_state": "1"
        }
    traffic_light_state_machine.tick(1)
    assert traffic_light_state_machine.time_remaining == 10
    assert traffic_light_state_machine.current_state == {
            "actor_states": {
                "4411": "green",
                "3411": "red",
                "4399": "green",
                "3399": "green"
            },
            "state": "1",
            "duration": 10,
            "next_state": "2"
        }

def test_get_current_actor_states(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(4, 3)
    actor_states = traffic_light_state_machine.get_current_actor_states()
    assert actor_states == {
                "4411": "yellow",
                "3411": "yellow",
                "4399": "red",
                "3399": "red"
            }
