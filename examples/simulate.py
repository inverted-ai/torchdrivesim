"""
Simple demonstration for how to initialize the simulator and use behavioral models to move the cars.
The behavioral models are provided by the IAI API and a key is required to use it and run this script.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import imageio
import torch
from omegaconf import OmegaConf

from torchdrivesim.behavior.iai import iai_initialize, IAIWrapper
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator, HomogeneousWrapper
from torchdrivesim.utils import Resolution


@dataclass
class SimulationConfig:
    driving_surface_mesh_path: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../resources/maps/carla/meshes/Town03_driving_surface_mesh.pkl"
    )
    map_name: str = "Town03"
    res: int = 1024
    fov: float = 200
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = np.pi / 2
    save_path: str = './simulation.gif'
    method: str = 'iai'
    left_handed: bool = True
    agent_count: int = 5
    steps: int = 20


def simulate(cfg: SimulationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    driving_surface_mesh = BirdviewMesh.unpickle(cfg.driving_surface_mesh_path).to(device)
    simulator_cfg = TorchDriveConfig(left_handed_coordinates=cfg.left_handed,
                                     renderer=RendererConfig(left_handed_coordinates=cfg.left_handed))

    location = f'carla:{":".join(cfg.map_name.split("_"))}'
    agent_attributes, agent_states, recurrent_states =\
        iai_initialize(location=location, agent_count=cfg.agent_count, center=tuple(cfg.center) if cfg.center is not None else None)
    agent_attributes, agent_states = agent_attributes.unsqueeze(0), agent_states.unsqueeze(0)
    agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(torch.float32)
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
    simulator = HomogeneousWrapper(simulator)
    npc_mask = torch.ones(agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
    npc_mask[0] = False
    simulator = IAIWrapper(
        simulator=simulator, npc_mask=npc_mask, recurrent_states=[recurrent_states],
        rear_axis_offset=agent_attributes[..., 2:3], locations=[location]
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

        action = torch.zeros((simulator.batch_size, simulator.agent_count, simulator.action_size))
        action = action.to(agent_states.device).to(agent_states.dtype)
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
    cli_cfg: SimulationConfig = OmegaConf.structured(
        SimulationConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    simulate(cli_cfg)  # type: ignore
