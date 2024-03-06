from torchdrivesim.rendering import Pytorch3DRendererConfig, RendererConfig, CV2RendererConfig, \
    renderer_from_config, DummyRendererConfig, NvdiffrastRendererConfig
import torch
import pytest


device = "cuda" if torch.cuda.is_available() else "cpu"


class TestRenderer:
    cfg: RendererConfig = None

    @classmethod
    def setup_class(cls):
        cls.cfg = RendererConfig(highlight_ego_vehicle=True)

    def test_render_agents(self):
        batch_size = 2
        renderer = renderer_from_config(self.cfg, batch_size=batch_size).to(device)
        agents_states = {"vehicle": torch.zeros(batch_size, 3, 4).to(device)}
        agents_attributes = {"vehicle": torch.ones(batch_size, 3, 3).to(device)}
        waypoints = torch.zeros(batch_size, 3, 2, 2).to(device)
        waypoints_mask = torch.ones(waypoints.shape[:-1], device=waypoints.device, dtype=torch.bool)
        renderer.render_frame(
            agents_states, agents_attributes,
            waypoints=waypoints, waypoints_rendering_mask=waypoints_mask
        )


class TestDummyRenderer(TestRenderer):

    @classmethod
    def setup_class(cls):
        cls.cfg = DummyRendererConfig()


class TestCV2Renderer(TestRenderer):

    @classmethod
    def setup_class(cls):
        cls.cfg = CV2RendererConfig()


@pytest.mark.depends_on_pytorch3d
class TestPytorch3DRenderer(TestRenderer):

    @classmethod
    def setup_class(cls):
        cls.cfg = Pytorch3DRendererConfig()


@pytest.mark.depends_on_nvdiffrast
class TestNvdiffrastRenderer(TestRenderer):

    @classmethod
    def setup_class(cls):
        cls.cfg = NvdiffrastRendererConfig()
