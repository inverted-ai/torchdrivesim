import os
import pytest
from torchdrivesim.traffic_lights import TrafficLightStateMachine, TrafficLightGroupState, TrafficLightState

@pytest.fixture
def traffic_light_state_machine():
    state_machine_file_path = os.path.join(os.path.dirname(__file__), "resources", "traffic_lights", 'intersection_1.json')
    return TrafficLightStateMachine.from_json(state_machine_file_path)

def test_reset(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.reset()
    assert traffic_light_state_machine.time_remaining >= 1

def test_set_to(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(2, time_remaining=3)
    assert traffic_light_state_machine.time_remaining == 3
    assert traffic_light_state_machine.current_state == TrafficLightGroupState(
        actor_states={
                "4411": TrafficLightState.green,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.yellow,
                "3399": TrafficLightState.yellow
            },
        sequence_number=2,
        duration=5,
        next_state=3)
    
    traffic_light_state_machine.set_to(2, time_remaining=1)
    assert traffic_light_state_machine.time_remaining == 1
    assert traffic_light_state_machine.current_state == TrafficLightGroupState(
        actor_states={
                "4411": TrafficLightState.green,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.yellow,
                "3399": TrafficLightState.yellow
            },
        sequence_number=2,
        duration=5,
        next_state=3)


def test_tick(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(0, 1)
    traffic_light_state_machine.tick(0.9)
    assert traffic_light_state_machine.time_remaining <= 0.1
    assert traffic_light_state_machine.current_state == TrafficLightGroupState(
        actor_states={
                "4411": TrafficLightState.red,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.red,
                "3399": TrafficLightState.red
            },
        sequence_number=0,
        duration=10,
        next_state=1)

    traffic_light_state_machine.tick(0.1)
    assert traffic_light_state_machine.time_remaining == 10
    assert traffic_light_state_machine.current_state == TrafficLightGroupState(
        actor_states={
                "4411": TrafficLightState.green,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.green,
                "3399": TrafficLightState.green
            },
        sequence_number=1,
        duration=10,
        next_state=2)


def test_tick_large_dt_middle_of_state(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(0, 10)
    traffic_light_state_machine.tick(23)
    assert traffic_light_state_machine.current_state == TrafficLightGroupState(
        actor_states={
                "4411": TrafficLightState.green,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.yellow,
                "3399": TrafficLightState.yellow
            },
        sequence_number=2,
        duration=5,
        next_state=3)
    assert traffic_light_state_machine.time_remaining == 2


def test_tick_large_dt_end_of_state(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(0, 10)
    traffic_light_state_machine.tick(25)
    assert traffic_light_state_machine.current_state == TrafficLightGroupState(
        actor_states={
                "4411": TrafficLightState.green,
                "3411": TrafficLightState.green,
                "4399": TrafficLightState.red,
                "3399": TrafficLightState.red
            },
        sequence_number=3,
        duration=10,
        next_state=4)
    assert traffic_light_state_machine.time_remaining == 10


def test_tick_large_dt_wraps_around_to_earlier_state(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(0, 10)
    traffic_light_state_machine.tick(45)
    assert traffic_light_state_machine.current_state == TrafficLightGroupState(
        actor_states={
                "4411": TrafficLightState.red,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.red,
                "3399": TrafficLightState.red
            },
        sequence_number=0,
        duration=10,
        next_state=1)
    assert traffic_light_state_machine.time_remaining == 5


def test_get_current_actor_states(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.set_to(4, 3)
    actor_states = traffic_light_state_machine.get_current_actor_states()
    assert actor_states == {
                "4411": TrafficLightState.yellow,
                "3411": TrafficLightState.yellow,
                "4399": TrafficLightState.red,
                "3399": TrafficLightState.red
            }
    
def test_to_json(traffic_light_state_machine: TrafficLightStateMachine):
    traffic_light_state_machine.to_json()
