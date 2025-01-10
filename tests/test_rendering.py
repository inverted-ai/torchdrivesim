from torchdrivesim.rendering import Pytorch3DRendererConfig, RendererConfig, CV2RendererConfig, \
    renderer_from_config, DummyRendererConfig, NvdiffrastRendererConfig
from torchdrivesim.mesh import BirdviewRGBMeshGenerator, BirdviewMesh
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
        num_agents = 3
        n_cameras = num_agents
        num_waypoints = 2
        renderer = renderer_from_config(self.cfg)
        agents_states = torch.zeros(batch_size, num_agents, 4).to(device)
        agents_attributes = torch.zeros(batch_size, num_agents, 2).to(device)
        agent_type_names = ['vehicle', 'pedestrian']
        agents_types = torch.zeros(batch_size, num_agents).to(device)
        waypoints = torch.zeros(batch_size, n_cameras, num_waypoints, 2).to(device)
        waypoints_mask = torch.ones(waypoints.shape[:-1], device=waypoints.device, dtype=torch.bool)
        camera_xy = agents_states[..., :2]
        camera_psi = torch.ones_like(camera_xy[..., :1]) * torch.pi / 2
        camera_sc = torch.cat([torch.sin(camera_psi), torch.cos(camera_psi)], dim=-1)
        birdview_mesh_generator = BirdviewRGBMeshGenerator(background_mesh=BirdviewMesh.empty().expand(batch_size),
                                                           agent_attributes=agents_attributes,
                                                           agent_types=agents_types,
                                                            agent_type_names=agent_type_names,
                                                                    color_map=renderer.color_map,
                                                                    rendering_levels=renderer.rendering_levels).to(device)        
        rbg_mesh = birdview_mesh_generator.generate(n_cameras,
            agent_state=agents_states.unsqueeze(1).expand(-1, n_cameras, -1, -1),
            waypoints=waypoints, waypoints_rendering_mask=waypoints_mask)

        renderer.render_frame(rbg_mesh, camera_xy, camera_sc)


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
