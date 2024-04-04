"""
Example showing how to extend agent states and attributes with additional information.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
import imageio
import lanelet2
import pandas
import torch
from omegaconf import OmegaConf

from torchdrivesim.behavior.iai import iai_initialize, iai_drive, IAIWrapper
from torchdrivesim.behavior.heuristic import heuristic_initialize
from torchdrivesim.kinematic import KinematicBicycle, TeleportingKinematicModel, KinematicModel
from torchdrivesim.lanelet2 import load_lanelet_map, road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.map import find_map_config
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import renderer_from_config, RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, Simulator, HomogeneousWrapper
from torchdrivesim.utils import Resolution


@dataclass
class InitializationVisualizationConfig:
    map_name: str = "carla_Town03"
    res: int = 1024
    fov: float = 200
    center: Optional[Tuple[float, float]] = None
    orientation: float = np.pi / 2
    save_path: str = './simulation.gif'
    method: str = 'iai'
    left_handed: bool = True
    agent_count: int = 5
    steps: int = 50


class DynamicBicycleModelWithSkidding(KinematicBicycle):
    """
    Similar to the bicycle model, but vehicles also have mass, and the action space is force and steering.
    The state also contains a component for lateral velocity, which determines skidding.
    """
    def __init__(self, max_force=1, dt=0.1, left_handed=False):
        super().__init__(dt=dt, left_handed=left_handed)
        self.max_force = max_force
        self.mass = None

    @staticmethod
    def pack_state(x, y, psi, speed, lateral=None):
        bicycle_state = KinematicBicycle.pack_state(x, y, psi, speed)
        if lateral is None:
            lateral = torch.zeros_like(speed)
        state = torch.cat([bicycle_state, lateral.unsqueeze(-1)], dim=-1)
        return state

    def copy(self, other=None):
        if other is None:
            other = self.__class__(max_force=self.max_force, dt=self.dt, left_handed=self.left_handed)
        return super().copy(other)

    def get_params(self):
        params = super().get_params()
        params['mass'] = self.mass
        return params

    def set_params(self, *args, **kwargs):
        """
        :param mass: Mass of the vehicle in kilograms.
        """
        super().set_params(*args, **kwargs)
        assert 'mass' in kwargs
        self.mass = kwargs['mass']

    def flattening(self, batch_shape):
        """
        Flatten all the batch dimensions of the lr parameter
        """
        super().flattening(batch_shape)
        assert self.mass is not None
        self.mass = self.mass.reshape((int(np.prod(batch_shape)),))

    def unflattening(self, batch_shape):
        """
        Unlatten all the batch dimensions of the lr parameter
        """
        super().unflattening(batch_shape)
        assert self.mass is not None
        self.mass = self.mass.reshape(batch_shape)

    def map_param(self, f):
        super().map_param(f)
        assert self.mass is not None
        self.mass = f(self.mass)

    def normalize_action(self, action, *args, **kwargs):
        return torch.stack([action[..., 0] / self.max_force,
                            action[..., 1] / self.max_steering], dim=-1)

    def denormalize_action(self, action, *args, **kwargs):
        return torch.stack([action[..., 0] * self.max_force,
                            action[..., 1] * self.max_steering], dim=-1)

    def step(self, action, dt=None):
        assert action.shape[-1] == 2, "The bicycle model takes as input only a and beta"
        action = self.denormalize_action(action, by_displacement=False)
        a, beta = action[..., 0] / self.mass, action[..., 1]
        if self.left_handed:
            beta = - beta # Flip steering angle when using left-hand coordinate system
        if dt is None:
            dt = self.dt
        x, y, psi, v = self.unpack_state(self.get_state())
        lateral = self.get_state()[..., 4]
        v = v + a * dt
        x = x + v * torch.cos(psi + beta) * dt - lateral * torch.sin(psi) * dt
        y = y + v * torch.sin(psi + beta) * dt + lateral * torch.cos(psi) * dt
        psi = psi + (v / self.lr) * torch.sin(beta) * dt
        lateral = 0.5 * lateral
        # psi = (np.pi + psi) % (2 * np.pi) - np.pi # Normalize angle between -pi and pi

        return self.pack_state(x, y, psi, v, lateral=lateral)

    def fit_action(self, future_state, current_state=None):
        bicycle_action = super().fit_action(future_state, current_state=current_state)
        acceleration, steering = bicycle_action[..., 0], bicycle_action[..., 1]
        force = acceleration * self.mass
        action = torch.stack([force, steering], dim=-1)

        return action

    def fit_param(self, df: pandas.DataFrame, precision: float = 0.01) -> Any:
        raise NotImplementedError


def visualize_map(cfg: InitializationVisualizationConfig):
    device = 'cuda'
    res = Resolution(cfg.res, cfg.res)
    map_cfg = find_map_config(cfg.map_name)
    driving_surface_mesh = map_cfg.road_mesh.to(device)
    simulator_cfg = TorchDriveConfig(left_handed_coordinates=map_cfg.left_handed_coordinates,
                                     renderer=RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates))

    location = map_cfg.iai_location_name
    agent_attributes, agent_states, recurrent_states =\
        iai_initialize(location=location, agent_count=cfg.agent_count, center=tuple(cfg.center) if cfg.center is not None else None)
    agent_states = torch.cat([agent_states, 10*torch.ones_like(agent_states[..., :1])], dim=-1)
    agent_attributes, agent_states = agent_attributes.unsqueeze(0), agent_states.unsqueeze(0)
    agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(torch.float32)
    kinematic_model = DynamicBicycleModelWithSkidding()
    kinematic_model.set_params(mass=agent_attributes[..., 2], lr=agent_attributes[..., 2])
    kinematic_model.set_state(agent_states)
    action_size = kinematic_model.action_size
    renderer = renderer_from_config(RendererConfig(left_handed_coordinates=cfg.left_handed), static_mesh=driving_surface_mesh)
    simulator = Simulator(
        cfg=simulator_cfg, road_mesh=driving_surface_mesh,
        kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
        initial_present_mask=dict(vehicle=torch.ones_like(agent_states[..., 0], dtype=torch.bool)),
        renderer=renderer,
    )
    simulator = HomogeneousWrapper(simulator)
    npc_mask = torch.ones(agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
    simulator = IAIWrapper(
        simulator=simulator, npc_mask=npc_mask, recurrent_states=[recurrent_states],
        rear_axis_offset=agent_attributes[..., 2:3], locations=[location]
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
        agent_states = agent_states.to(device).to(torch.float32)
        simulator.step(torch.zeros(agent_states.shape[0], 0, action_size, dtype=agent_states.dtype, device=agent_states.device))

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
