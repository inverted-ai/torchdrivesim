"""
Simple demonstration for how to replay some agent behaviors from the INTERACTION dataset in TorchDriveSim.
The dataset needs to be downloaded separately in order to run this script.
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

from torchdrivesim.behavior.replay import interaction_replay, ReplayWrapper
from torchdrivesim.kinematic import KinematicBicycle, TeleportingKinematicModel
from torchdrivesim.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.simulator import TorchDriveConfig, Simulator, HomogeneousWrapper
from torchdrivesim.utils import Resolution


@dataclass
class InitializationVisualizationConfig:
    interaction_path: str
    map_name: str
    res: int = 1024
    fov: float = 200
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = np.pi / 2
    save_path: str = './replay.gif'
    method: str = 'iai'
    left_handed: bool = False
    agent_count: int = 5
    steps: int = 20


def visualize_map(cfg: InitializationVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    map_path = os.path.join(cfg.interaction_path, 'maps', f'{cfg.map_name}.osm')
    lanelet_map = load_lanelet_map(map_path, origin=cfg.map_origin)

    road_mesh = BirdviewMesh.set_properties(road_mesh_from_lanelet_map(lanelet_map), category='road').to(device)
    lane_mesh = lanelet_map_to_lane_mesh(lanelet_map).to(device)
    simulator_cfg = TorchDriveConfig(left_handed_coordinates=cfg.left_handed)

    agent_attributes, agent_states, present_mask = interaction_replay(cfg.map_name, cfg.interaction_path)
    agent_attributes = agent_attributes.to(device).to(torch.float32)
    agent_states = agent_states.to(device).to(torch.float32)
    present_mask = present_mask.to(device).to(torch.bool)
    replay_mask = torch.ones_like(present_mask[0, :, 0])

    kinematic_model = TeleportingKinematicModel()
    kinematic_model.set_state(agent_states[..., 0, :])
    renderer = renderer_from_config(simulator_cfg.renderer, static_mesh=BirdviewMesh.concat([road_mesh, lane_mesh]))

    simulator = Simulator(
        cfg=simulator_cfg, road_mesh=road_mesh,
        kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
        initial_present_mask=dict(vehicle=present_mask[..., 0]), renderer=renderer,
    )
    simulator = ReplayWrapper(
        simulator, npc_mask=dict(vehicle=replay_mask),
        agent_states=dict(vehicle=agent_states), present_masks=dict(vehicle=present_mask),
    )

    images = []
    for _ in range(cfg.steps):
        if cfg.center is None:
            camera_xy = simulator.get_innermost_simulator().renderer.world_center.to(device)
        else:
            camera_xy = torch.tensor(cfg.center).unsqueeze(0).to(torch.float32).to(device)
        camera_psi = torch.ones_like(camera_xy[..., :1]) * cfg.orientation
        image = simulator.render(camera_xy=camera_xy, camera_psi=camera_psi, res=res, fov=cfg.fov)
        images.append(image)

        agent_states = simulator.get_state()['vehicle']
        simulator.step(dict(vehicle=agent_states[..., 1:0, :]))

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    imageio.mimsave(
        cfg.save_path, [image[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0) for image in images],
        format="GIF-PIL", fps=10
    )
    try:
        from pygifsicle import optimize
        optimize(cfg.save_path, options=['--no-warnings'])
    except ImportError:
        print("You can install pygifsicle for gif compression and optimization.")


if __name__ == '__main__':
    cli_cfg: InitializationVisualizationConfig = OmegaConf.structured(
        InitializationVisualizationConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    visualize_map(cli_cfg)  # type: ignore
