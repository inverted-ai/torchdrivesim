import pytest
import json
import os
from torchdrivesim.traffic_lights import TrafficLightController, TrafficLightStateMachine, TrafficLightState

@pytest.fixture
def traffic_light_controller():
    folder_path = os.path.join(os.path.dirname(__file__), "resources",  "traffic_lights")
    state_machine_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
    state_machine_files.sort()
    light_machines = [TrafficLightStateMachine.from_json(f) for f in state_machine_files]
    return TrafficLightController(light_machines)


def test_reset(traffic_light_controller: TrafficLightController):
    traffic_light_controller.reset()
    assert all([t > 0 for t in traffic_light_controller.time_remaining])

def test_set_to(traffic_light_controller: TrafficLightController):
    traffic_light_controller.set_to([[2, 4],[0, 1]])
    assert traffic_light_controller.time_remaining == [4, 1]
    assert traffic_light_controller.current_state == {
                "4411": TrafficLightState.green,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.yellow,
                "3399": TrafficLightState.yellow,
                "3404": TrafficLightState.red,
                "4404": TrafficLightState.red,
                "3406": TrafficLightState.red,
                "3405": TrafficLightState.red,
                "3403": TrafficLightState.red,
                "4403": TrafficLightState.red
    }
    traffic_light_controller.set_to([[2, 3],[1, 0]])
    assert traffic_light_controller.time_remaining == [3, 0]
    assert traffic_light_controller.current_state == {
                "4411": TrafficLightState.green,
                "3411": TrafficLightState.red,
                "4399": TrafficLightState.yellow,
                "3399": TrafficLightState.yellow,
                "3404": TrafficLightState.red,
                "4404": TrafficLightState.red,
                "3406": TrafficLightState.green,
                "3405": TrafficLightState.green,
                "3403": TrafficLightState.red,
                "4403": TrafficLightState.red,
    }

def test_tick(traffic_light_controller: TrafficLightController):
    traffic_light_controller.set_to([[0, 10], [0, 2]])
    traffic_light_controller.tick(1)
    assert traffic_light_controller.current_state == {
        "4411": TrafficLightState.red,
        "3411": TrafficLightState.red,
        "4399": TrafficLightState.red,
        "3399": TrafficLightState.red,
        "3404": TrafficLightState.red,
        "4404": TrafficLightState.red,
        "3406": TrafficLightState.red,
        "3405": TrafficLightState.red,
        "3403": TrafficLightState.red,
        "4403": TrafficLightState.red
    }
    traffic_light_controller.tick(1)
    assert traffic_light_controller.current_state == {
        "4411": TrafficLightState.red,
        "3411": TrafficLightState.red,
        "4399": TrafficLightState.red,
        "3399": TrafficLightState.red,
        "3404": TrafficLightState.red,
        "4404": TrafficLightState.red,
        "3406": TrafficLightState.green,
        "3405": TrafficLightState.green,
        "3403": TrafficLightState.red,
        "4403": TrafficLightState.red
    }

def test_load_controller_from_json():
    TrafficLightController.from_json(os.path.join(os.path.dirname(__file__), "resources", "traffic_lights_controller", "intersection_controller.json"))

def test_to_json(traffic_light_controller: TrafficLightController):
    traffic_light_controller.to_json()
