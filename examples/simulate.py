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

from torchdrivesim.behavior.iai import iai_initialize, IAINPCController
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.map import find_map_config, traffic_controls_from_map_config
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator
from torchdrivesim.traffic_lights import current_light_state_tensor_from_controller
from torchdrivesim.utils import Resolution


@dataclass
class SimulationConfig:
    map_name: str = "carla_Town03"
    res: int = 1024
    fov: float = 200
    center: Optional[Tuple[float, float]] = None
    orientation: float = np.pi / 2
    save_path: str = './simulation.gif'
    method: str = 'iai'
    agent_count: int = 5
    steps: int = 20


def simulate(cfg: SimulationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    map_cfg = find_map_config(cfg.map_name)
    traffic_light_controller = map_cfg.traffic_light_controller
    initial_light_state_name = traffic_light_controller.current_state_with_name
    traffic_light_ids = [stopline.actor_id for stopline in map_cfg.stoplines if stopline.agent_type == 'traffic_light']
    if cfg.center is None:
        cfg.center = map_cfg.center
    driving_surface_mesh = map_cfg.road_mesh.to(device)
    simulator_cfg = TorchDriveConfig(left_handed_coordinates=map_cfg.left_handed_coordinates,
                                     renderer=RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates))

    # Pre-compute traffic light states for all timesteps
    light_states_all_timesteps = [initial_light_state_name] * (cfg.steps + 1)  # +1 for initialization

    location = map_cfg.iai_location_name
    agent_attributes, agent_states, recurrent_states =\
        iai_initialize(location=location, agent_count=cfg.agent_count, center=tuple(cfg.center) if cfg.center is not None else None,
                       traffic_light_state_history=[initial_light_state_name])
    agent_attributes, agent_states = agent_attributes.unsqueeze(0), agent_states.unsqueeze(0)
    agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(torch.float32)

    # Separate ego agent from NPCs
    ego_attributes = agent_attributes[..., :1, :]
    ego_states = agent_states[..., :1, :]
    npc_attributes = agent_attributes[..., 1:, :]
    npc_states = agent_states[..., 1:, :]

    # Setup ego agent
    kinematic_model = KinematicBicycle()
    kinematic_model.set_params(lr=ego_attributes[..., 2])
    kinematic_model.set_state(ego_states)
    renderer = renderer_from_config(simulator_cfg.renderer)
    traffic_controls = traffic_controls_from_map_config(map_cfg)
    traffic_controls = {k: v.to(device) for k, v in traffic_controls.items()}

    # Create NPC controller
    npc_present_mask = torch.ones_like(npc_states[..., 0], dtype=torch.bool)
    npc_controller = IAINPCController(
        npc_size=npc_attributes[..., :2],
        npc_state=npc_states,
        npc_lr=npc_attributes[..., 2],
        location=location,
        npc_present_mask=npc_present_mask,
        traffic_light_controller=traffic_light_controller,
        light_states_all_timesteps=light_states_all_timesteps,
        agent_type_names=['vehicle']  # All agents are vehicles in this example
    )

    simulator = Simulator(
        cfg=simulator_cfg, road_mesh=driving_surface_mesh,
        kinematic_model=kinematic_model, agent_size=ego_attributes[..., :2],
        initial_present_mask=torch.ones_like(ego_states[..., 0], dtype=torch.bool),
        renderer=renderer, traffic_controls=traffic_controls,
        npc_controller=npc_controller
    ).to(device=device)

    traffic_controls['traffic_light'].set_state(
        current_light_state_tensor_from_controller(traffic_light_controller, traffic_light_ids).unsqueeze(0).to(device)
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
        action = action.to(ego_states.device).to(ego_states.dtype)
        simulator.step(action)

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    imageio.mimsave(
        cfg.save_path, [image[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0) for image in images],
        format="GIF", fps=10
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
