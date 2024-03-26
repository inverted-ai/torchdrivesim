import pytest
import os
from torchdrivesim.traffic_lights import TrafficLightController, TrafficLightStateMachine

@pytest.fixture
def traffic_light_controller():
    folder_path = os.path.join(os.path.dirname(__file__), "resources",  "traffic_lights")
    state_machine_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]
    state_machine_files.sort()
    light_machines = [TrafficLightStateMachine(f) for f in state_machine_files]
    return TrafficLightController(light_machines)


def test_reset(traffic_light_controller: TrafficLightController):
    traffic_light_controller.reset()
    assert all([t > 0 for t in traffic_light_controller.time_remaining])

def test_set_to(traffic_light_controller: TrafficLightController):
    traffic_light_controller.set_to([[2, 4],[0, 1]])
    assert traffic_light_controller.time_remaining == [4, 1]
    assert traffic_light_controller.current_state == {
                "4411": "green",
                "3411": "red",
                "4399": "yellow",
                "3399": "yellow",
                "3404": "red",
                "4404": "red",
                "3406": "red",
                "3405": "red",
                "3403": "red",
                "4403": "red"
    }
    traffic_light_controller.set_to([[2, 3],[1, 0]])
    assert traffic_light_controller.time_remaining == [3, 0]
    assert traffic_light_controller.current_state == {
                "4411": "green",
                "3411": "red",
                "4399": "yellow",
                "3399": "yellow",
                "3404": "red",
                "4404": "red",
                "3406": "green",
                "3405": "green",
                "3403": "red",
                "4403": "red"
    }

def test_tick(traffic_light_controller: TrafficLightController):
    traffic_light_controller.set_to([[0, 10], [0, 2]])
    traffic_light_controller.tick(10)
    assert traffic_light_controller.current_state == {
        "4411": "red",
        "3411": "red",
        "4399": "red",
        "3399": "red",
        "3404": "red",
        "4404": "red",
        "3406": "red",
        "3405": "red",
        "3403": "red",
        "4403": "red"
    }
    traffic_light_controller.tick(10)
    assert traffic_light_controller.current_state == {
        "4411": "red",
        "3411": "red",
        "4399": "red",
        "3399": "red",
        "3404": "red",
        "4404": "red",
        "3406": "green",
        "3405": "green",
        "3403": "red",
        "4403": "red"
    }
