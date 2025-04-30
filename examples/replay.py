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

from torchdrivesim.behavior.replay import interaction_replay, ReplayController
from torchdrivesim.kinematic import KinematicBicycle, TeleportingKinematicModel
from torchdrivesim.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.simulator import TorchDriveConfig, Simulator
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

    # Create NPC controller for replay
    npc_controller = ReplayController(
        npc_size=agent_attributes[..., :2],
        npc_states=agent_states,
        npc_present_masks=present_mask,
        agent_type_names=['vehicle']  # All agents are vehicles in this example
    )

    # Initialize kinematic model with initial state
    kinematic_model = TeleportingKinematicModel()
    kinematic_model.set_state(agent_states[..., 0, :])

    # Initialize simulator with just the NPC controller
    simulator = Simulator(
        cfg=simulator_cfg,
        road_mesh=BirdviewMesh.concat([road_mesh, lane_mesh]),
        kinematic_model=kinematic_model,
        agent_size=agent_attributes[..., :2],
        initial_present_mask=present_mask[..., 0],
        renderer=renderer_from_config(simulator_cfg.renderer),
        npc_controller=npc_controller
    )

    images = []
    for _ in range(cfg.steps):
        if cfg.center is None:
            camera_xy = simulator.get_world_center().to(device)
        else:
            camera_xy = torch.tensor(cfg.center).unsqueeze(0).to(torch.float32).to(device)
        camera_psi = torch.ones_like(camera_xy[..., :1]) * cfg.orientation
        image = simulator.render(camera_xy=camera_xy, camera_psi=camera_psi, res=res, fov=cfg.fov)
        images.append(image)

        # Step with zero actions since we're just replaying
        action = torch.zeros((simulator.batch_size, simulator.agent_count, simulator.action_size), device=device)
        simulator.step(action)

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
