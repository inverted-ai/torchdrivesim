"""
Simple demonstration for how to generate an initial simulator state and visualize it.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import imageio
import torch
from omegaconf import OmegaConf

from torchdrivesim.behavior.iai import iai_initialize
from torchdrivesim.behavior.heuristic import heuristic_initialize
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.map import find_map_config
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.utils import Resolution


@dataclass
class InitializationVisualizationConfig:
    map_name: str = "carla_Town03"
    res: int = 1024
    fov: float = 200
    center: Optional[Tuple[float, float]] = None
    orientation: float = np.pi / 2
    save_path: str = './initialization.png'
    method: str = 'iai'
    left_handed: bool = True
    agent_count: int = 5


def visualize_map(cfg: InitializationVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    map_cfg = find_map_config(cfg.map_name)
    driving_surface_mesh = map_cfg.road_mesh.to(device)
    simulator_cfg = TorchDriveConfig(left_handed_coordinates=map_cfg.left_handed_coordinates,
                                     renderer=RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates))

    if cfg.method == 'iai':
        location = map_cfg.iai_location_name
        agent_attributes, agent_states, _ = iai_initialize(location=location, agent_count=cfg.agent_count,
                                                           center=tuple(cfg.center) if cfg.center is not None else None)
    elif cfg.method == 'heuristic':
        agent_attributes, agent_states = heuristic_initialize(map_cfg.lanelet_map, agent_num=cfg.agent_count)
    else:
        raise ValueError(f'Unrecognized initialization method: {cfg.method}')
    agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(torch.float32)
    agent_attributes, agent_states = agent_attributes.unsqueeze(0), agent_states.unsqueeze(0)
    kinematic_model = KinematicBicycle()
    kinematic_model.set_params(lr=agent_attributes[..., 2])
    kinematic_model.set_state(agent_states)
    renderer = renderer_from_config(simulator_cfg.renderer, static_mesh=driving_surface_mesh)

    simulator = Simulator(
        cfg=simulator_cfg, road_mesh=driving_surface_mesh,
        kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
        initial_present_mask=dict(vehicle=torch.ones_like(agent_states[..., 0], dtype=torch.bool)),
        renderer=renderer,
    )
    if cfg.center is None:
        camera_xy = simulator.renderer.world_center.to(device)
    else:
        camera_xy = torch.tensor(cfg.center).unsqueeze(0).to(torch.float32).to(device)
    camera_psi = torch.ones_like(camera_xy[..., :1]) * cfg.orientation
    image = simulator.render(camera_xy=camera_xy, camera_psi=camera_psi, res=res, fov=cfg.fov)

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    imageio.imsave(
        cfg.save_path, image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    )


if __name__ == '__main__':
    cli_cfg: InitializationVisualizationConfig = OmegaConf.structured(
        InitializationVisualizationConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    visualize_map(cli_cfg)  # type: ignore
