from torchdrivesim.rendering import NvdiffrastRendererConfig, Pytorch3DRendererConfig, RendererConfig, renderer_from_config
import torch
import pytest


device = "cuda" if torch.cuda.is_available() else "cpu"
@pytest.mark.parametrize("cfg", [Pytorch3DRendererConfig(highlight_ego_vehicle=True),Pytorch3DRendererConfig()])
def test_render_agents(cfg: RendererConfig):
    batch_size = 2
    renderer = renderer_from_config(cfg, batch_size=batch_size).to(device)
    agents_states = {"vehicle": torch.zeros(batch_size, 3, 4).to(device)}
    agents_attributes = {"vehicle": torch.ones(batch_size, 3, 3).to(device)}
    waypoints = torch.zeros(batch_size, 3, 2, 3).to(device)
    waypoints_mask = torch.ones(waypoints.shape[:-1], device=waypoints.device, dtype=torch.bool)
    renderer.render_frame(agents_states, agents_attributes, waypoints=waypoints, waypoints_rendering_mask=waypoints_mask)
