"""
An example showing how to define an OpenAI gym environment based on TorchDriveSim.
It uses the IAI API to provide behaviors for other vehicles and requires an access key to run.
"""
import os
import logging
import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List

import gymnasium as gym
import torch
import numpy as np
from torch import Tensor
from invertedai.common import TrafficLightState

from torchdrivesim.behavior.iai import iai_location_info, get_static_actors, IAIWrapper, \
    iai_area_initialize, iai_initialize
from torchdrivesim.kinematic import BicycleByDisplacement, BicycleNoReversing
from torchdrivesim.mesh import BirdviewMesh, point_to_mesh
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.utils import Resolution, save_video
from torchdrivesim.lanelet2 import find_lanelet_directions, load_lanelet_map
from torchdrivesim.traffic_controls import TrafficLightControl
from torchdrivesim.simulator import TorchDriveConfig, SimulatorInterface, \
    BirdviewRecordingWrapper, Simulator, HomogeneousWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class IAIGymEnvConfig:
    simulator: TorchDriveConfig = TorchDriveConfig()
    driving_surface_mesh_path: str = os.path.join(
        os.path.dirname(os.path.realpath(
            __file__)), "../resources/maps/carla/meshes/Town03_driving_surface_mesh.pkl"
    )
    render_mode: Optional[str] = None
    video_filename: str = "rendered_video.mp4"
    location: str = 'Town03'
    res: int = 1024
    fov: float = 200
    center: Tuple[float, float] = (0, 0)
    left_handed: bool = True
    agent_count: int = 5
    # x, y, orientation, speed
    ego_state: Optional[Tuple[float, float, float, float]] = None
    use_mock_lights: bool = True
    use_area_initialize: bool = False
    max_environment_steps: int = 200


@dataclass
class TaskGymEnvConfig:
    iai_gym: IAIGymEnvConfig = IAIGymEnvConfig()
    start_speed: Optional[float] = 1.0
    start_orientation: Optional[float] = None
    start_point: Tuple[float, float] = (0, 0)
    goal_point: Tuple[float, float] = (0, 0)


@dataclass
class WaypointEnvConfig:
    iai_gym: IAIGymEnvConfig = IAIGymEnvConfig()
    waypoints: List[List[float]] = None


class GymEnv(gym.Env):

    metadata = {
        "render_modes": ["video", "rgb_array"],
        "render_fps": 10
    }

    def __init__(self, cfg: IAIGymEnvConfig, simulator: SimulatorInterface):
        if cfg.render_mode is not None and cfg.render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError
        self.render_mode = cfg.render_mode

        acceleration_range = (-1.0, 1.0)
        steering_range = (-1.0, 1.0)
        action_range = np.ndarray(shape=(2, 2), dtype=np.float32)
        action_range[:, 0] = acceleration_range
        action_range[:, 1] = steering_range
        self.max_environment_steps = cfg.max_environment_steps
        self.environment_steps = 0
        self.action_space = gym.spaces.Box(
            low=action_range[0],
            high=action_range[1],
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

        self.reward_range = (- float('inf'), float('inf'))
        self.collision_threshold = 0.0
        self.offroad_threshold = 0.0

        self.config = cfg
        self.simulator = simulator
        self.current_action = None

        if self.render_mode == "video":
            self.simulator = BirdviewRecordingWrapper(
                self.simulator, res=Resolution(cfg.res, cfg.res), fov=cfg.fov, to_cpu=True)
            self.video_filename = cfg.video_filename

        self.last_birdview = None

        self.start_sim = self.simulator.copy()

    # TODO: use the seed
    # TODO: return the reset info
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.simulator = self.start_sim.copy()
        self.environment_steps = 0
        self.last_birdview = None
        return self.get_obs(), {}

    def step(self, action: Tensor):
        self.environment_steps += 1
        self.simulator.step(action)
        self.last_action = self.current_action if self.current_action is not None else action
        self.current_action = action
        return self.get_obs(), self.get_reward(), self.is_terminated(), self.is_truncated(), self.get_info()

    def get_obs(self):
        birdview = self.simulator.render_egocentric().cpu().numpy()
        return birdview

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        r = torch.zeros_like(x)
        return r

    def is_done(self):
        x = self.simulator.get_state()[..., 0]
        done = torch.zeros_like(x, dtype=torch.bool)
        done += self.environment_steps >= self.max_environment_steps
        return done

    def is_truncated(self):
        return self.environment_steps >= self.max_environment_steps

    def is_terminated(self):
        return False

    def get_info(self):
        info = dict(
            offroad=self.simulator.compute_offroad() > self.offroad_threshold,
            collision=self.simulator.compute_collision() > self.collision_threshold,
            outcome=None
        )
        return info

    def seed(self, seed=None):
        pass

    def render(self):
        if self.render_mode == 'rgb_array':
            birdview = self.simulator.render_egocentric().cpu().numpy()
            return np.transpose(birdview.squeeze(), axes=(1, 2, 0))
        else:
            raise NotImplementedError

    def close(self):
        if isinstance(self.simulator, BirdviewRecordingWrapper):
            bvs = self.simulator.get_birdviews()
            save_video(bvs, self.video_filename)


class IAIGymEnv(GymEnv):
    """
    A gym environment for driving with background traffic animated by the IAI API.
    In the current version, reset will use the same initial conditions, but different behaviors.
    """

    def __init__(self, cfg: IAIGymEnvConfig):
        device = torch.device('cuda')
        driving_surface_mesh = BirdviewMesh.unpickle(
            cfg.driving_surface_mesh_path).to(device)
        simulator_cfg = cfg.simulator
        iai_location = f'carla:{":".join(cfg.location.split("_"))}'

        if cfg.use_mock_lights:
            static_actors = get_static_actors(iai_location_info(iai_location))
            traffic_light_control = TrafficLightControl(pos=torch.stack(tuple(static_actors.values())).unsqueeze(0), use_mock_lights=True, ids=list(static_actors.keys()))
            traffic_light_states = dict(zip(static_actors.keys(), traffic_light_control.compute_state(0).squeeze()))
            traffic_light_state_history = [{k:TrafficLightState(traffic_light_control.allowed_states[int(traffic_light_states[k])]) for k in traffic_light_states}]
        else:
            traffic_light_control = None
            traffic_light_state_history = None

        if cfg.use_area_initialize:
            agent_attributes, agent_states, recurrent_states = \
                iai_area_initialize(location=iai_location,
                                    agent_density=cfg.agent_count, center=tuple(cfg.center), traffic_light_state_history=traffic_light_state_history)
        else:
            agent_attributes, agent_states, recurrent_states = \
                iai_initialize(location=iai_location,
                                    agent_count=cfg.agent_count, center=tuple(cfg.center), traffic_light_state_history=traffic_light_state_history)

        if cfg.ego_state is not None:
            agent_states[0, :] = torch.Tensor(cfg.ego_state)

        agent_attributes, agent_states = agent_attributes.unsqueeze(
            0), agent_states.unsqueeze(0)
        agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(
            torch.float32)
        kinematic_model = BicycleNoReversing()
#        kinematic_model = BicycleByDisplacement()
        kinematic_model.set_params(lr=agent_attributes[..., 2])
        kinematic_model.set_state(agent_states)
        renderer = renderer_from_config(
            simulator_cfg.renderer, static_mesh=driving_surface_mesh)

        simulator = Simulator(
            cfg=simulator_cfg, road_mesh=driving_surface_mesh,
            kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
            initial_present_mask=dict(vehicle=torch.ones_like(
                agent_states[..., 0], dtype=torch.bool)),
            renderer=renderer,
            traffic_controls={"traffic-light": traffic_light_control}
        )
        simulator = HomogeneousWrapper(simulator)
        npc_mask = torch.ones(
            agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
        npc_mask[0] = False
        simulator = IAIWrapper(
            simulator=simulator, npc_mask=npc_mask, recurrent_states=[
                recurrent_states],
            rear_axis_offset=agent_attributes[..., 2:3], locations=[
                iai_location]
        )
        super().__init__(cfg=cfg, simulator=simulator)

    def get_reward(self):
        offroad_penalty = -self.simulator.compute_offroad()
        collision = -self.simulator.compute_collision()
        economy_penalty = -self.current_action.norm(2)
        speed_bonus = self.simulator.get_state()[..., 3]
        x = self.simulator.get_state()[..., 0]
        r = torch.zeros_like(x)
        r += offroad_penalty + collision + economy_penalty + speed_bonus
        r = torch.clamp(r, min=-10., max=10.)
        return r


class TaskGymEnv(IAIGymEnv):
    def __init__(self, cfg: TaskGymEnvConfig):
        self.start_point = cfg.start_point
        self.start_speed = cfg.start_speed
        lanelet_map_path = os.path.join(
                os.path.dirname(os.path.realpath(
                    __file__)), f"../resources/maps/carla/maps/{cfg.iai_gym.location}.osm"
            )
        self.lanelet_map = load_lanelet_map(lanelet_map_path)
        if cfg.start_orientation is None:
            self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map, x=cfg.start_point[0], y=cfg.start_point[1])[0])
        else:
            self.start_orientation = cfg.start_orientation
        self.goal_point = cfg.goal_point
        cfg.iai_gym.ego_state = (self.start_point[0], self.start_point[1], self.start_orientation, self.start_speed)
        cfg.iai_gym.center = self.start_point
        annotated_mesh = [point_to_mesh(self.start_point, "start_point"), point_to_mesh(self.goal_point, "goal_point")]
        super().__init__(cfg.iai_gym)
        innermost_simulator = self.simulator.get_innermost_simulator()
        innermost_simulator.renderer.add_static_meshes(annotated_mesh)
        self.start_sim = self.simulator.copy()
        logger.info(inspect.getsource(TaskGymEnv.get_reward))

    def get_reward(self):
        offroad_penalty = -self.simulator.compute_offroad()
        collision_penalty = -self.simulator.compute_collision()
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        orientation = self.simulator.get_state()[..., 2]
        speed = self.simulator.get_state()[..., 3]
        lanelet_orientations = torch.Tensor(find_lanelet_directions(lanelet_map=self.lanelet_map, x=x, y=y)).cuda()
        if len(lanelet_orientations) > 0:
            lanelet_orientation = float(lanelet_orientations[torch.argmin(abs(lanelet_orientations - orientation)).item()])
            orientation_reward = math.cos(orientation - lanelet_orientation)
            orientation_penalty = -math.sin(abs(orientation - lanelet_orientation))
        else:
            orientation_reward = 0
            orientation_penalty = 0
        reach_goal = 100 if math.dist((x, y), self.goal_point) < 1 else 0 #0.5 and abs(orientation - lanelet_orientation) < math.pi / 16) else 0
        r = torch.zeros_like(x)
        r += reach_goal + orientation_reward * speed + orientation_penalty * speed # + offroad_penalty + collision_penalty
        return r

    def is_terminated(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        p = torch.tensor((x, y))
        t = torch.tensor(self.goal_point)
        pass_goal = math.cos(math.pi / 2 - math.atan2((t - p)[1], (t - p)[0])) < 0
        reach_goal = math.dist((x, y), self.goal_point) < 0.5
        return pass_goal or reach_goal


class WaypointEnv(IAIGymEnv):
    def __init__(self, cfg: WaypointEnvConfig):
        self.waypoints = cfg.waypoints
        assert(self.waypoints is not None)
        assert(len(self.waypoints) >= 2)
        self.current_target_idx = 1
        self.current_target = self.waypoints[self.current_target_idx]

        lanelet_map_path = os.path.join(
                os.path.dirname(os.path.realpath(
                    __file__)), f"../resources/maps/carla/maps/{cfg.iai_gym.location}.osm"
            )
        self.lanelet_map = load_lanelet_map(lanelet_map_path)

        self.set_start_pos()

        cfg.iai_gym.ego_state = (self.start_point[0], self.start_point[1], self.start_orientation, self.start_speed)
        cfg.iai_gym.center = self.start_point

        super().__init__(cfg.iai_gym)
        innermost_simulator = self.simulator.get_innermost_simulator()
        annotated_mesh = []
        for waypoint in self.waypoints:
            annotated_mesh.append(point_to_mesh(waypoint, "waypoint"))
        innermost_simulator.renderer.add_static_meshes(annotated_mesh)
        self.start_sim = self.simulator.copy()
        logger.info(inspect.getsource(WaypointEnv.get_reward))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.set_start_pos()
        self.current_target_idx = 1
        self.current_target = self.waypoints[self.current_target_idx]
        obs, info = super().reset()
        return obs, info

    def set_start_pos(self):
        self.start_point = self.waypoints[0]
        self.start_speed = 3
        self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map, x=self.start_point[0], y=self.start_point[1])[0])

    def step(self, action: Tensor):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.check_reach_target():
            self.current_target_idx += 1
            if self.current_target_idx < len(self.waypoints):
                self.current_target = self.waypoints[self.current_target_idx]
        return obs, reward, terminated, truncated, info

    def check_reach_target(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        return (self.current_target is not None) and (math.dist((x, y), self.current_target) < 1)

    def get_reward(self):
        offroad_penalty = -self.simulator.compute_offroad()
        collision_penalty = -self.simulator.compute_collision()
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        orientation = self.simulator.get_state()[..., 2]
        speed = self.simulator.get_state()[..., 3]
        lanelet_orientations = torch.Tensor(find_lanelet_directions(lanelet_map=self.lanelet_map, x=x, y=y)).cuda()
        if len(lanelet_orientations) > 0:
            lanelet_orientation = float(lanelet_orientations[torch.argmin(abs(lanelet_orientations - orientation)).item()])
            orientation_reward = math.cos(orientation - lanelet_orientation)
        else:
            orientation_reward = 0
        reach_target_reward = 100 if self.check_reach_target() else 0
        r = torch.zeros_like(x)
        r += reach_target_reward + orientation_reward * speed
        return r


class SingleAgentWrapper(gym.Wrapper):
    """
    Removes batch and agent dimensions from the environment interface.
    Only safe if those dimensions are both singletons.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        return self.transform_out(obs), _

    def step(self, action: Tensor):
        action = torch.Tensor(action).unsqueeze(0).unsqueeze(0).to("cuda")
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.transform_out(obs)
        reward = self.transform_out(reward)
        terminated = self.transform_out(terminated)
        info = self.transform_out(info)

        return obs, reward, terminated, truncated, info

    def transform_out(self, x):
        if torch.is_tensor(x):
            t = x.squeeze(0).squeeze(0).cpu()
        elif isinstance(x, dict):
            t = {k: self.transform_out(v) for (k, v) in x.items()}
        elif isinstance(x, np.ndarray):
            t = self.transform_out(torch.tensor(x)).cpu().numpy()
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
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
