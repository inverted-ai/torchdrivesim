import copy

import pytest
import torch

from tests.simulator.test_simulator import TestBaseSimulator
from torchdrivesim.simulator import CollisionMetric, SimulatorWrapper, RecordingWrapper, \
    BirdviewRecordingWrapper, \
    SelectiveWrapper, BoundedRegionWrapper, HomogeneousWrapper
from torchdrivesim.utils import Resolution
from tests import device


@pytest.mark.depends_on_lanelet2
class TestBaseWrappedSimulator(TestBaseSimulator):

    @classmethod
    def get_simulator(cls):
        return SimulatorWrapper(super().get_simulator())

    def test_extend(self):
        n = 2
        assert self.simulator.extend(n, in_place=False)

    def test_get_innermost_simulator(self):
        assert self.simulator.get_innermost_simulator()

    def test_render_egocentric(self):
        res = self.simulator.get_innermost_simulator().renderer.res
        assert self.get_tensor(self.simulator.render_egocentric()).shape == self.birdview_shape


class TestHomogenousSimulator(TestBaseWrappedSimulator):

    @classmethod
    def setup_method(cls):
        super().setup_method()

    @staticmethod
    def get_tensor(item):
        return item

    @staticmethod
    def tensor_collection(tensor):
        return tensor

    @classmethod
    def get_simulator(cls):
        return HomogeneousWrapper(super().get_simulator())

    @pytest.mark.parametrize('collision_metric_type', [cm for cm in (CollisionMetric.nograd,
                                                                     CollisionMetric.iou, CollisionMetric.discs)])
    def test_compute_collision(self, collision_metric_type):

        self.simulator.cfg.collision_metric = collision_metric_type
        collision_metrics = self.simulator.compute_collision()
        assert len(collision_metrics.shape) == 2 and not torch.all(collision_metrics)

    @pytest.mark.depends_on_pytorch3d
    def test_compute_collision_pytorch3d(self):
        self.simulator.cfg.collision_metric = CollisionMetric.nograd_pytorch3d
        collision_metrics = self.simulator.compute_collision()
        assert len(collision_metrics.shape) == 2 and not torch.all(collision_metrics)


class TestRecordingSimulator(TestBaseWrappedSimulator):
    @classmethod
    def get_simulator(cls):
        recording_function = {'recording': lambda _: 'result'}
        return RecordingWrapper(super().get_simulator(),
                                recording_function, initial_recording=False)

    def test_record(self):
        self.simulator.record()
        assert self.simulator.records['recording'] == ['result']


class TestBirdviewRecordingSimulator(TestBaseWrappedSimulator):
    @classmethod
    def get_simulator(cls):
        return BirdviewRecordingWrapper(super().get_simulator(),
                                        res=Resolution(128, 128))

    def test_extend(self):
        super().test_extend()
        assert self.simulator.camera_xy.shape == torch.Size([2, 2]) and \
               self.simulator.camera_psi.shape == torch.Size([2, 1])

    def test_get_birdviews(self):
        assert self.simulator.get_birdviews(stack=True).shape == torch.Size([2, 1, 3, 128, 128])


class TestSelectiveSimulator(TestBaseWrappedSimulator):

    mock_exposed_action = None

    @classmethod
    def setup_class(cls):
        cls.mock_exposed_agent_count = 1
        super().setup_class()
        cls.offroad_shape = torch.Size([cls.data_batch_size, cls.mock_agent_count])
        cls.collision_shape = torch.Size([cls.data_batch_size, cls.mock_agent_count])
        cls.wrong_way_shape = torch.Size([cls.data_batch_size, cls.mock_agent_count])
        cls.mock_exposed_agent_state = torch.Tensor([[[1, 1, 0, 0]], [[1, 1, 0, 0]]]).to(device)
        cls.mock_exposed_future_state = torch.Tensor([[[2, 2, 1, 0]], [[2, 2, 1, 0]]]).to(device)
        cls.mock_exposed_action = torch.Tensor([[[1, 1]], [[1, 1]]]).to(device)
        cls.agents_absolute_shape = torch.Size([2, 1, 6])

    @classmethod
    def get_simulator(cls):
        exposed_agent_limit = cls.tensor_collection(1)
        # default action takes A agents instead of E
        default_action = cls.tensor_collection(cls.mock_exposed_action.expand(-1, cls.mock_agent_count, -1))
        return SelectiveWrapper(super().get_simulator(),
                                exposed_agent_limit, default_action)

    def test_get_all_agents_absolute(self):
        self.simulator.set_state(self.tensor_collection(self.mock_exposed_agent_state))
        all_agents_absolute = self.get_tensor(self.simulator.get_all_agents_absolute())
        assert all_agents_absolute[0, 0, 0] == 1
        assert all_agents_absolute.shape == self.agents_absolute_shape

    def test_fit_action(self):
        future_state = self.tensor_collection(self.mock_exposed_future_state)
        current_state = self.tensor_collection(self.mock_exposed_agent_state)
        actions = self.get_tensor(self.simulator.fit_action(future_state, current_state))
        assert actions[0, 0, 0] > 0
        assert actions.shape == self.action_shape

    def test_render_egocentric(self):
        assert self.get_tensor(self.simulator.render_egocentric()).shape == self.birdview_shape

    def test_step(self):
        pre_state = self.get_tensor(self.simulator.get_state())
        self.simulator.step(self.tensor_collection(self.mock_exposed_action))
        curr_state = self.get_tensor(self.simulator.get_state())
        assert pre_state[0, 0, 0] != curr_state[0, 0, 0]

    def test_compute_collision(self, collision_metric_type=None):
        self.simulator.set_state(dict(vehicle=torch.zeros_like(self.mock_exposed_agent_state)))
        collision_metrics = self.get_tensor(self.simulator.compute_collision())
        assert len(collision_metrics.shape) == 2 and not torch.all(collision_metrics)

    def test_get_all_agents_relative(self):
        self.simulator.set_state(self.tensor_collection(self.mock_exposed_agent_state))
        all_agents_relative = self.get_tensor(self.simulator.get_all_agents_relative())
        assert all_agents_relative[0, 0, 0, 0] == 0
        assert all_agents_relative.shape == self.agents_relative_shape


class TestProximitySimulator(TestSelectiveSimulator):

    @classmethod
    def get_simulator(cls):
        exposed_agent_limit = cls.tensor_collection(1)
        # default action takes A agents instead of E
        default_action = cls.tensor_collection(cls.mock_exposed_action.expand(-1, cls.mock_agent_count, -1))
        config = copy.copy(cls.config)
        config.remove_exiting_vehicles = False
        cutoff_polygon_verts = cls.tensor_collection(torch.Tensor([[-100, 100], [100, 100], [100, -100], [-100, -100]])
                                                     .unsqueeze(0).expand(cls.data_batch_size, -1, -1).to(device))
        return BoundedRegionWrapper(super().get_simulator().inner_simulator,
                                    exposed_agent_limit, default_action,
                                    warmup_timesteps=1,
                                    cutoff_polygon_verts=cutoff_polygon_verts)

    def test_update_exposed_agents(self):
        self.simulator.update_exposed_agents()
        # When both agents can be exposed and there is 1 slot, agent 0 is exposed
        assert torch.all(self.get_tensor(self.simulator.exposed_agents) == 0)

        self.simulator.set_state\
            (self.tensor_collection(torch.ones(self.data_batch_size, self.mock_exposed_agent_count, 4).to(device) * 999))
        self.simulator.update_exposed_agents()
        # When agent 0 is exposed but out of region, agent 1 will be exposed
        assert torch.all(self.get_tensor(self.simulator.exposed_agents) == 1)

        self.simulator.cutoff_polygon_verts = self.tensor_collection(torch.Tensor([[-100, 100], [100, 100], [100, -100], [-100, -100]])
                                                     .unsqueeze(0).expand(self.data_batch_size, -1, -1).to(device))
        self.simulator.update_exposed_agents()
        # When agent 1 is exposed and inside the region, agent 1 remains exposed
        assert torch.all(self.get_tensor(self.simulator.exposed_agents) == 1)

        # Reset exposed agents
        self.simulator.exposed_agents = None
        self.simulator.update_exposed_agents()
        # Move agent 0 outside a region
        self.simulator.set_state \
            (self.tensor_collection(torch.ones(self.data_batch_size, self.mock_exposed_agent_count, 4).to(device) * 200))
        self.simulator.update_exposed_agents()
        # When agent 0 is exposed and outside the region, agent 1 will be exposed
        assert torch.all(self.get_tensor(self.simulator.exposed_agents) == 1)
