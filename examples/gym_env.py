"""
An example showing how to define an OpenAI gym environment based on TorchDriveSim.
It uses the IAI API to provide behaviors for other vehicles and requires an access key to run.
See https://github.com/inverted-ai/torchdriveenv for a production-quality RL environment.
"""
import contextlib
import os
import signal
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import gym
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import Tensor
import imageio

from torchdrivesim.behavior.iai import iai_initialize, IAINPCController
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.map import find_map_config
from torchdrivesim.rendering import RendererConfig, renderer_from_config
from torchdrivesim.utils import Resolution
from torchdrivesim.simulator import TorchDriveConfig, Simulator

logger = logging.getLogger(__name__)


@dataclass
class TorchDriveGymEnvConfig:
    simulator: TorchDriveConfig = TorchDriveConfig()
    visualize_to: Optional[str] = None
    map_name: str = 'carla_Town03'
    res: int = 1024
    fov: float = 200
    agent_count: int = 5
    steps: int = 20


class GymEnv(gym.Env):
    def __init__(self, config: TorchDriveGymEnvConfig, simulator: Simulator):
        dtype = np.float32
        acceleration_range = (-1.0, 1.0)
        steering_range = (-1.0, 1.0)
        action_range = np.ndarray(shape=(2, 2), dtype=dtype)
        action_range[:, 0] = acceleration_range
        action_range[:, 1] = steering_range
        self.max_environment_steps = 1000
        self.environment_steps = 0
        self.action_space = gym.spaces.Box(
            low=action_range[0],
            high=action_range[1],
            dtype=dtype
        )
        self.observation_space = gym.spaces.Dict({
            'speed': gym.spaces.Box(low=np.array([0.0], dtype=dtype), high=np.array([200.0], dtype=dtype), dtype=dtype),
            'birdview_image': gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=dtype),
            'prev_action': self.action_space
        })
        self.reward_range = (- float('inf'), float('inf'))
        self.collision_threshold = 0.0
        self.offroad_threshold = 0.0

        self.config = config
        self.simulator = simulator
        self.start_sim = self.simulator.copy()
        self.prev_action = None
        self.frames = []
        self.video_filename = None
        self.recording = False
        self.res = None
        self.fov = None

    def reset(self):
        self.simulator = self.start_sim.copy()
        self.environment_steps = 0
        if self.recording:
            self.frames = []
        return self.get_obs()

    def step(self, action: Tensor):
        self.environment_steps += 1
        # Move action to the same device as simulator tensors
        action = action.to(self.simulator.get_state().device)
        self.simulator.step(action)
        self.prev_action = action
        
        # Capture frame if recording
        if self.recording:
            frame = self.simulator.render_egocentric(res=self.res, fov=self.fov)
            self.frames.append(frame[0].cpu().numpy().transpose(1, 2, 0))
            
        return self.get_obs(), self.get_reward(), self.is_done(), self.get_info()

    def get_obs(self):
        state = self.simulator.get_state()
        speed = state[..., 3]
        birdview = self.simulator.render_egocentric()
        obs = dict(
            speed=speed,
            birdview_image=birdview,
            prev_action=self.prev_action,
        )
        return obs

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        r = torch.zeros_like(x)
        return r

    def is_done(self):
        x = self.simulator.get_state()[..., 0]
        done = torch.zeros_like(x, dtype=torch.bool)
        done += self.environment_steps >= self.max_environment_steps
        return done

    def get_info(self):
        x = self.simulator.get_state()[..., 0]
        info = dict(
            offroad=self.simulator.compute_offroad() > self.offroad_threshold,
            collision=self.simulator.compute_collision() > self.collision_threshold,
            outcome=None
        )
        return info

    def seed(self, seed=None):
        pass

    def render(self, mode='human', res=Resolution(1024, 1024), fov=100, filename='video.gif'):
        if mode == 'human':
            raise NotImplementedError
        elif mode == 'video':
            self.video_filename = filename
            self.recording = True
            self.res = res
            self.fov = fov
            self.frames = []

    def close(self):
        if self.recording and self.frames:
            imageio.mimsave(self.video_filename, self.frames, format="GIF-PIL", fps=10)
            try:
                from pygifsicle import optimize
                optimize(self.video_filename, options=['--no-warnings'])
            except ImportError:
                print("You can install pygifsicle for gif compression and optimization.")
            self.recording = False
            self.frames = []


class IAIGymEnv(GymEnv):
    """
    A gym environment for driving with background traffic animated by the IAI API.
    In the current version, reset will use the same initial conditions, but different behaviors.
    """
    def __init__(self, cfg: TorchDriveGymEnvConfig):
        device = torch.device('cuda')
        map_cfg = find_map_config(cfg.map_name)
        if map_cfg is None:
            raise RuntimeError(f'Map {cfg.map_name} not found')
        driving_surface_mesh = map_cfg.road_mesh.to(device)
        simulator_cfg = TorchDriveConfig(
            left_handed_coordinates=map_cfg.left_handed_coordinates,
            renderer=RendererConfig(left_handed_coordinates=map_cfg.left_handed_coordinates)
        )
        iai_location = map_cfg.iai_location_name

        agent_attributes, agent_states, recurrent_states = \
            iai_initialize(location=iai_location, agent_count=cfg.agent_count, center=tuple(map_cfg.center))
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

        # Create NPC controller
        npc_present_mask = torch.ones_like(npc_states[..., 0], dtype=torch.bool)
        npc_controller = IAINPCController(
            npc_size=npc_attributes[..., :2],
            npc_state=npc_states,
            npc_lr=npc_attributes[..., 2],
            location=iai_location,
            npc_present_mask=npc_present_mask,
            agent_type_names=['vehicle']  # All agents are vehicles in this example
        )

        simulator = Simulator(
            cfg=simulator_cfg, road_mesh=driving_surface_mesh,
            kinematic_model=kinematic_model, agent_size=ego_attributes[..., :2],
            initial_present_mask=torch.ones_like(ego_states[..., 0], dtype=torch.bool),
            renderer=renderer,
            npc_controller=npc_controller
        )

        super().__init__(config=cfg, simulator=simulator)
        self.max_environment_steps = 100

    def get_reward(self):
        offroad_penalty = -self.simulator.compute_offroad()
        collision = -self.simulator.compute_collision()
        economy_penalty = -self.prev_action.norm(2)
        speed_bonus = self.simulator.get_state()[..., 3]
        x = self.simulator.get_state()[..., 0]
        r = torch.zeros_like(x)
        r += offroad_penalty + collision + economy_penalty + speed_bonus
        r = torch.clamp(r,min=-10.,max=10.)
        return r


class SingleAgentWrapper(gym.Wrapper):
    """
    Removes batch and agent dimensions from the environment interface.
    Only safe if those dimensions are both singletons.
    """
    def reset(self):
        obs = super().reset()
        return self.transform_out(obs)

    def step(self, action: Tensor):
        action = action.unsqueeze(0).unsqueeze(0)
        obs, reward, is_done, info = super().step(action)
        obs = self.transform_out(obs)
        reward = self.transform_out(reward)
        is_done = self.transform_out(is_done)
        info = self.transform_out(info)

        return obs, reward, is_done, info

    def transform_out(self, x):
        if torch.is_tensor(x):
            t = x.squeeze(0).squeeze(0)
        elif isinstance(x, dict):
            t = {k: self.transform_out(v) for (k, v) in x.items()}
        else:
            t = x
        return t

    def transform_in(self, x):
        if torch.is_tensor(x):
            t = x.unsqueeze(0).unsqueeze(0)
        elif isinstance(x, dict):
            t = {k: self.transform_in(v) for (k, v) in x.items()}
        else:
            t = x
        return t

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()


gym.register('torchdrivesim/IAI-v0', entry_point=lambda args: SingleAgentWrapper(IAIGymEnv(cfg=args)))


def main(cfg: TorchDriveGymEnvConfig):
    # Enable graceful termination
    def sigterm_handler(signum, frame):
        raise InterruptedError("SIGTERM received")

    signal.signal(signal.SIGTERM, sigterm_handler)

    with contextlib.closing(gym.make('torchdrivesim/IAI-v0', args=cfg)) as env:
        # will produce a video showing what's going on
        if cfg.visualize_to is not None:
            env.render(mode='video', res=Resolution(cfg.res, cfg.res), fov=cfg.fov, filename=cfg.visualize_to)

        np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

        for j in range(2):
            env.reset()
            action = torch.tensor([1, 0], dtype=torch.float32, device='cuda')  # accelerate hard without steering
            for i in range(cfg.steps):
                obs, reward, done, info = env.step(action)
                # action = info['expert_action']
                if info['collision']:
                    print("collision")
                if info['offroad']:
                    print("offroad")
                if done:
                    break


if __name__ == "__main__":

    cli_cfg: TorchDriveGymEnvConfig = OmegaConf.structured(
        TorchDriveGymEnvConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )

    main(cli_cfg)
