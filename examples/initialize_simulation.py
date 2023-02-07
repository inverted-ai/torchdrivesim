"""
Simple demonstration for how to grab a map and visualize it.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import imageio
import lanelet2
import torch
from omegaconf import OmegaConf

from torchdrive.behavior.iai import iai_initialize
from torchdrive.behavior.heuristic import heuristic_initialize
from torchdrive.kinematic import KinematicBicycle
from torchdrive.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrive.mesh import BirdviewMesh
from torchdrive.rendering import renderer_from_config
from torchdrive.simulator import TorchDriveConfig, Simulator
from torchdrive.utils import Resolution


@dataclass
class InitializationVisualizationConfig:
    maps_path: str
    map_name: str
    res: int = 1024
    fov: float = 200
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = np.pi / 2
    save_path: str = './initialization.png'
    method: str = 'iai'
    left_handed: bool = False
    agent_count: int = 5


def visualize_map(cfg: InitializationVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    map_path = os.path.join(cfg.maps_path, f'{cfg.map_name}.osm')
    lanelet_map = load_lanelet_map(map_path, origin=cfg.map_origin)

    road_mesh = BirdviewMesh.set_properties(road_mesh_from_lanelet_map(lanelet_map), category='road').to(device)
    lane_mesh = lanelet_map_to_lane_mesh(lanelet_map).to(device)
    simulator_cfg = TorchDriveConfig(left_handed_coordinates=cfg.left_handed)

    if cfg.method == 'iai':
        if cfg.map_name.startswith('Town'):
            location = f'carla:{":".join(cfg.map_name.split("_"))}'
        else:
            location = f'canada:vancouver:{cfg.map_name}'
        agent_attributes, agent_states, _ = iai_initialize(location=location, agent_count=cfg.agent_count, center=tuple(cfg.center))
    elif cfg.method == 'heuristic':
        agent_attributes, agent_states = heuristic_initialize(lanelet_map, agent_num=cfg.agent_count)
    else:
        raise ValueError(f'Unrecognized initialization method: {cfg.method}')
    agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(torch.float32)
    kinematic_model = KinematicBicycle()
    kinematic_model.set_params(lr=agent_attributes[..., 2])
    kinematic_model.set_state(agent_states)
    renderer = renderer_from_config(simulator_cfg.renderer, static_mesh=BirdviewMesh.concat([road_mesh, lane_mesh]))

    simulator = Simulator(
        cfg=simulator_cfg, road_mesh=road_mesh,
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
