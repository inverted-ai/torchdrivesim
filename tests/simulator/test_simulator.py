import os
import pytest
import torch
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.goals import WaypointGoal
from tests import device


@pytest.mark.depends_on_lanelet2
class TestBaseSimulator:
    dataset_config = None
    mock_batch = None
    dataset = None
    config = None
    data_batch_size = 2
    mock_agent_count = 2
    mock_exposed_agent_count = 2
    mock_agent_attributes = None
    mock_agent_state = None
    mock_future_state = None
    mock_action = None
    lanelet_map_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../resources/testing_lanelet2map.osm")
    states_shape = None
    agent_size_shape = None
    present_mask_shape = None
    agents_absolute_shape = None
    agents_relative_shape = None
    action_shape = None
    offroad_shape = None
    collision_shape = None
    wrong_way_shape = None

    @classmethod
    def get_shapes(cls, batch_size):
        cls.states_shape = torch.Size([batch_size, cls.mock_exposed_agent_count, 4])
        cls.agent_size_shape = torch.Size([batch_size, cls.mock_exposed_agent_count, 2])
        cls.present_mask_shape = torch.Size([batch_size, cls.mock_exposed_agent_count])
        cls.agents_absolute_shape = torch.Size([batch_size, cls.mock_agent_count, 6])
        cls.agents_relative_shape = torch.Size([batch_size, cls.mock_exposed_agent_count, 1, 6])
        cls.action_shape = torch.Size([batch_size, cls.mock_exposed_agent_count, 2])
        cls.offroad_shape = torch.Size([batch_size, cls.mock_exposed_agent_count])
        cls.collision_shape = torch.Size([batch_size, cls.mock_exposed_agent_count])
        cls.wrong_way_shape = torch.Size([batch_size, cls.mock_exposed_agent_count])
        cls.birdview_shape = torch.Size([cls.data_batch_size, cls.mock_exposed_agent_count, 3,
                                         64, 64])

    @classmethod
    def setup_class(cls):
        cls.get_shapes(cls.data_batch_size)
        cls.mock_agent_attributes = torch.ones(cls.data_batch_size, cls.mock_agent_count, 3)
        cls.mock_agent_state = torch.Tensor([[[0, 0, 0, 0], [1, 1, 0, 0]]]).expand(cls.data_batch_size, -1, -1).to(device)
        cls.mock_future_state = torch.Tensor([[[1, 1, 1, 0], [2, 2, 1, 0]]]).expand(cls.data_batch_size, -1, -1).to(device)
        cls.mock_action = torch.Tensor([[[0, 0], [1, 1]]]).expand(cls.data_batch_size, -1, -1).to(device)

    @classmethod
    def setup_method(cls) -> None:
        cls.config = TorchDriveConfig(left_handed_coordinates=False)
        cls.simulator = cls.get_simulator()

    @classmethod
    def get_simulator(cls):
        import lanelet2
        road_mesh = BirdviewMesh.empty(batch_size=cls.data_batch_size)
        kinematic_model = KinematicBicycle()
        kinematic_model.set_params(lr=cls.mock_agent_attributes[..., 2])
        kinematic_model.set_state(cls.mock_agent_state)
        kinematic_model = dict(vehicle=kinematic_model)
        agent_size = dict(vehicle=cls.mock_agent_attributes[..., :2])
        initial_present_mask = dict(vehicle=torch.ones_like(cls.mock_agent_state[..., 0], dtype=torch.bool))
        origin = (0, 0)
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(*origin))
        lanelet_map = lanelet2.io.load(cls.lanelet_map_path, projector)
        lanelet_map = [lanelet_map for _ in range(cls.data_batch_size)]
        waypoint_goals = WaypointGoal(dict(vehicle=torch.zeros_like(cls.mock_agent_state[..., :2])[:, :, None, None, :]))
        return Simulator(road_mesh, kinematic_model, agent_size,
                         initial_present_mask, cls.config, lanelet_map=lanelet_map, waypoint_goals=waypoint_goals).to(device)

    @staticmethod
    def get_tensor(item):
        return item['vehicle']

    @staticmethod
    def tensor_collection(tensor):
        return {'vehicle': tensor}

    def test_move_to_device(self):
        self.simulator.to('cpu').to(device)

    def test_copy(self):
        old_state = self.simulator.get_state()
        simulator_copy = self.simulator.copy()
        new_state = self.simulator.across_agent_types(lambda st: st + 1, old_state)
        simulator_copy.set_state(new_state)
        assert (self.get_tensor(self.simulator.get_state()) == self.get_tensor(old_state)).all()

    def test_extend(self):
        n = 2
        extended = self.simulator.extend(n, in_place=False)
        assert extended.road_mesh.verts.shape[0] == self.simulator.road_mesh.verts.shape[0] * n
        assert extended.batch_size == self.simulator.batch_size * n
        assert len(extended.lanelet_map) == len(self.simulator.lanelet_map) * n
        assert self.get_tensor(extended.agent_size).shape[0] == self.get_tensor(self.simulator.agent_size).shape[0] * n

    def test_select_batch_elements(self):
        idx = [1, 1, 0]
        s = self.simulator[idx]
        assert s.get_innermost_simulator().batch_size == 3
        state = self.get_tensor(s.get_state())
        assert (state[0] == state[1]).all()
        s.render_egocentric()

    def test_get_state(self):
        assert self.get_tensor(self.simulator.get_state()).shape == self.states_shape

    def test_get_agent_size(self):
        assert self.get_tensor(self.simulator.get_agent_size()).shape == self.agent_size_shape

    def test_get_present_mask(self):
        assert self.get_tensor(self.simulator.get_present_mask()).shape == self.present_mask_shape

    def test_get_all_agents_absolute(self):
        self.simulator.set_state(self.tensor_collection(self.mock_agent_state))
        all_agents_absolute = self.get_tensor(self.simulator.get_all_agents_absolute())
        assert all_agents_absolute[0, 1, 0] == 1 and all_agents_absolute[0, 0, 0] == 0
        assert all_agents_absolute.shape == self.agents_absolute_shape

    def test_get_all_agents_relative(self):
        self.simulator.set_state(self.tensor_collection(self.mock_agent_state))
        all_agents_relative = self.get_tensor(self.simulator.get_all_agents_relative())
        assert all_agents_relative[0, 0, 0, 0] == 1
        assert all_agents_relative.shape == self.agents_relative_shape

    def test_get_innermost_simulator(self):
        assert self.simulator is self.simulator.get_innermost_simulator()

    def test_step(self):
        pre_state = self.get_tensor(self.simulator.get_state())
        self.simulator.step(self.tensor_collection(self.mock_action))
        curr_state = self.get_tensor(self.simulator.get_state())
        assert not curr_state[0, 1].eq(pre_state[0, 1]).all() and curr_state[0, 0].eq(pre_state[0, 0]).all()

    def test_fit_action(self):
        future_state = self.tensor_collection(self.mock_future_state)
        current_state = self.tensor_collection(self.mock_agent_state)
        actions = self.get_tensor(self.simulator.fit_action(future_state, current_state))
        assert actions[0, 0, 0] > 0
        assert actions.shape == self.action_shape

    def test_render_egocentric(self):
        res = self.simulator.renderer.res
        assert self.get_tensor(self.simulator.render_egocentric()).shape == self.birdview_shape

    def test_compute_collision(self, collision_metric_type=None):
        self.simulator.set_state(dict(vehicle=torch.zeros_like(self.mock_agent_state)))
        collision_metrics = self.get_tensor(self.simulator.compute_collision())
        assert len(collision_metrics.shape) == 2 and torch.all(collision_metrics)

    def test_compute_offroad(self):
        assert self.get_tensor(self.simulator.compute_offroad()).shape == self.offroad_shape

    def test_compute_wrong_way(self):
        assert self.get_tensor(self.simulator.compute_wrong_way()).shape == self.wrong_way_shape
