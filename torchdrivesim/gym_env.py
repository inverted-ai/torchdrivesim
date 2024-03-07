"""
An example showing how to define an OpenAI gym environment based on TorchDriveSim.
It uses the IAI API to provide behaviors for other vehicles and requires an access key to run.
"""
import os
import logging
import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import gymnasium as gym
import torch
import pickle
import random
import numpy as np
from torch import Tensor
from invertedai.common import TrafficLightState, AgentState, Point, AgentAttributes, RecurrentState

from torchdrivesim.behavior.iai import iai_location_info, get_static_actors, IAIWrapper, \
    iai_initialize, cache_iai_location_info
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.mesh import BirdviewMesh, point_to_mesh
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.utils import Resolution, save_video
from torchdrivesim.utils import set_seeds
from torchdrivesim.lanelet2 import find_lanelet_directions, load_lanelet_map
from torchdrivesim.traffic_controls import TrafficLightControl, StopSignControl, YieldControl
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
    use_background_traffic: bool = True
    # True for training, and False for evaluation
    terminated_at_infraction: bool = False
    ego_only: bool = False
    seed: Optional[int] = None


@dataclass
class Scenario:
    agent_states: List[List[float]] = None
    agent_attributes: List[List[float]] = None
    recurrent_states: List[List[float]] = None


@dataclass
class WaypointSuiteEnvConfig:
    iai_gym: IAIGymEnvConfig = IAIGymEnvConfig()
    locations: List[str] = None
    waypointsuite: List[List[List[float]]] = None
    car_sequence_suite: List[Optional[Dict[int, List[List[float]]]]] = None
    traffic_light_state_suite: List[Optional[Dict[int, List[str]]]] = None
    stop_sign_suite: List[Optional[List[int]]] = None
    scenarios: List[Optional[Scenario]] = None


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
#        steering_range = (-1.0, 1.0)
        steering_range = (-0.3, 0.3)
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

#        if self.render_mode == "video":
#            self.simulator = BirdviewRecordingWrapper(
#                self.simulator, res=Resolution(cfg.res, cfg.res), fov=cfg.fov, to_cpu=True)
#            self.video_filename = cfg.video_filename

        self.last_birdview = None

#        self.start_sim = self.simulator.copy()

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
#        x = self.simulator.get_state()[..., 0]
#        done = torch.zeros_like(x, dtype=torch.bool)
#        done += self.environment_steps >= self.max_environment_steps
#        return done
         return self.is_truncated() or self.is_terminated()

    def is_truncated(self):
        return self.environment_steps >= self.max_environment_steps

    def is_terminated(self):
        return False

    def get_info(self):
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            is_success=(self.environment_steps >= self.max_environment_steps),
            occur_exception=False
        )
        return self.info

    def seed(self, seed=None):
        pass

    def render(self):
        if self.render_mode == 'rgb_array':
            birdview = self.simulator.render_egocentric().cpu().numpy()
            return np.transpose(birdview.squeeze(), axes=(1, 2, 0))
        else:
            raise NotImplementedError

    def mock_step(self):
        obs = np.zeros((1, 3, 64, 64)) # self.last_obs
        reward = 0
        terminated = False
        truncated = True
        info = dict(
            offroad=torch.Tensor([[0]]),
            collision=torch.Tensor([[0]]),
            traffic_light_violation=torch.Tensor([[0]]),
            is_success=False,
            occur_exception=True
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        if isinstance(self.simulator, BirdviewRecordingWrapper):
            bvs = self.simulator.get_birdviews()
            save_video(bvs, self.config.video_filename)


def build_iai_simulator(cfg: IAIGymEnvConfig, scenario=None, car_sequences=None, preset_traffic_light_states=None, stop_sign_ids=None):
    with torch.no_grad():
        device = torch.device("cuda")
        driving_surface_mesh_path = os.path.join(
            os.path.dirname(os.path.realpath(
                __file__)), f"../resources/maps/carla/meshes/{cfg.location}_driving_surface_mesh.pkl"
        )
        driving_surface_mesh = BirdviewMesh.unpickle(
            driving_surface_mesh_path).to(device)
        simulator_cfg = cfg.simulator
        iai_location = f'carla:{":".join(cfg.location.split("_"))}'

        if cfg.use_mock_lights:
            static_actors = get_static_actors(iai_location_info(iai_location))
            traffic_light_ids = []
            traffic_light_poses = []
            stop_sign_ids = []
            stop_sign_poses = []
            yield_sign_ids = []
            yield_sign_poses = []
            for id in static_actors:
                if static_actors[id]['agent_type'] == "traffic-light":
                    traffic_light_ids.append(id)
                    traffic_light_poses.append(static_actors[id]['pos'])
                if static_actors[id]['agent_type'] == "stop-sign":
                    stop_sign_ids.append(id)
                    stop_sign_poses.append(static_actors[id]['pos'])
                if static_actors[id]['agent_type'] == "yield":
                    yield_sign_ids.append(id)
                    yield_sign_poses.append(static_actors[id]['pos'])

            if len(traffic_light_ids) > 0:
                traffic_light_control = TrafficLightControl(location=cfg.location, pos=torch.stack(traffic_light_poses).unsqueeze(0), use_mock_lights=True, ids=traffic_light_ids, preset_states=preset_traffic_light_states)
                traffic_light_states = dict(zip(static_actors.keys(), traffic_light_control.compute_state(0).squeeze()))
                traffic_light_state_history = [{k:TrafficLightState(traffic_light_control.allowed_states[int(traffic_light_states[k])]) for k in traffic_light_states}]
            else:
                traffic_light_control = None
                traffic_light_state_history = None

            if len(stop_sign_ids) > 0:
                stop_sign_control = StopSignControl(pos=torch.stack(stop_sign_poses).unsqueeze(0), ids=stop_sign_ids)
            else:
                stop_sign_control = None

            if len(yield_sign_ids) > 0:
                yield_control = YieldControl(pos=torch.stack(yield_sign_poses).unsqueeze(0), ids=yield_sign_ids)
            else:
                yield_control = None

        else:
            traffic_light_control = None
            traffic_light_state_history = None
            stop_sign_control = None
            yield_control = None

    #    if scenario is not None:
    #        agent_states = torch.Tensor(scenario.agent_states)
    #        agent_attributes = torch.Tensor(scenario.agent_attributes)
    #        recurrent_states = [RecurrentState(packed=recurrent_state) for recurrent_state in scenario.recurrent_states]
    #        if cfg.ego_state is not None:
    #            agent_states = torch.cat([torch.Tensor(cfg.ego_state).unsqueeze(0), agent_states])
    #            agent_attributes = torch.cat([agent_attributes[0, :].unsqueeze(0), agent_attributes])
    #            recurrent_states =  [recurrent_states[0]] + recurrent_states
    #            if car_sequences is not None:
    #                new_car_sequences = {}
    #                for agent_idx in car_sequences:
    #                    new_car_sequences[agent_idx + 1] = car_sequences[agent_idx].copy()
    #                car_sequences = new_car_sequences
    #    else:
    #        if cfg.use_area_initialize:
    #            agent_attributes, agent_states, recurrent_states = \
    #                iai_area_initialize(location=iai_location,
    #                                    agent_density=cfg.agent_count, center=tuple(cfg.center), traffic_light_state_history=traffic_light_state_history)
    #        else:
    #            agent_attributes, agent_states, recurrent_states = \
    #                iai_initialize(location=iai_location,
    #                               agent_count=cfg.agent_count, center=tuple(cfg.center), traffic_light_state_history=traffic_light_state_history)
    #
    #        if cfg.ego_state is not None:
    #            agent_states[0, :] = torch.Tensor(cfg.ego_state)

        if cfg.ego_only:
#            ego_state = AgentState(center=Point(x=cfg.ego_state[0], y=cfg.ego_state[1]), orientation=cfg.ego_state[2], speed=cfg.ego_state[3])
            agent_states = torch.Tensor([cfg.ego_state[0], cfg.ego_state[1], cfg.ego_state[2], cfg.ego_state[3]]).unsqueeze(0)
            length = np.random.random() * (5.5 - 4.8) + 4.8
            width = np.random.random() * (2.2 - 1.8) + 1.8
            rear_axis_offset = np.random.random() * (0.97 - 0.82) + 0.82
            agent_attributes = torch.Tensor([length, width, rear_axis_offset]).unsqueeze(0)
            recurrent_states = torch.Tensor([0] * 132).unsqueeze(0)


        if cfg.use_background_traffic:
            background_traffic_dir = "background_traffic"
            while True:
                background_traffic_file = os.path.join(background_traffic_dir, random.choice(list(filter(lambda x: x.split("_")[0]==cfg.location, os.listdir(background_traffic_dir)))))
                with open(background_traffic_file, "rb") as f:
                    background_traffic = pickle.load(f)
                if len(background_traffic["agent_states"]) + background_traffic["agent_density"] < 100:
                    break

            ego_state = AgentState(center=Point(x=cfg.ego_state[0], y=cfg.ego_state[1]), orientation=cfg.ego_state[2], speed=cfg.ego_state[3])
            remain_agent_states = [ego_state]
            remain_agent_attributes = [background_traffic["agent_attributes"][0]]
            remain_recurrent_states = [background_traffic["recurrent_states"][0]]
            if scenario is not None:
                for agent_state in scenario.agent_states:
                    remain_agent_states.append(AgentState(center=Point(x=agent_state[0], y=agent_state[1]), orientation=agent_state[2], speed=agent_state[3]))
                for agent_attribute in scenario.agent_attributes:
                    remain_agent_attributes.append(AgentAttributes(length=agent_attribute[0], width=agent_attribute[1], rear_axis_offset=agent_attribute[2]))
                for recurrent_state in scenario.recurrent_states:
                    remain_recurrent_states.append(RecurrentState(packed=recurrent_state))

            for i in range(len(background_traffic["agent_states"])):
                agent_state = background_traffic["agent_states"][i]
                if math.dist(cfg.ego_state[:2], (agent_state.center.x, agent_state.center.y)) > 100:
                    remain_agent_states.append(agent_state)
                    remain_agent_attributes.append(background_traffic["agent_attributes"][i])
                    remain_recurrent_states.append(background_traffic["recurrent_states"][i])
#            agent_attributes, agent_states, recurrent_states = iai_initialize(location=iai_location,
#                   agent_count=background_traffic["agent_density"], agent_attributes=remain_agent_attributes, agent_states=remain_agent_states, recurrent_states=remain_recurrent_states,
#                   center=tuple(cfg.ego_state[:2]), traffic_light_state_history=traffic_light_state_history)
            agent_attributes, agent_states, recurrent_states = iai_initialize(location=iai_location,
                   agent_count=max(95 - len(remain_agent_states), background_traffic["agent_density"]), agent_attributes=remain_agent_attributes, agent_states=remain_agent_states, recurrent_states=remain_recurrent_states,
                   center=tuple(cfg.ego_state[:2]), traffic_light_state_history=traffic_light_state_history)

#            agent_attributes, agent_states, recurrent_states = iai_initialize(location=iai_location,
#                   agent_count=1, agent_attributes=remain_agent_attributes, agent_states=remain_agent_states, recurrent_states=remain_recurrent_states,
#                   center=tuple(cfg.ego_state[:2]), traffic_light_state_history=traffic_light_state_history)


        agent_attributes, agent_states = agent_attributes.unsqueeze(
            0), agent_states.unsqueeze(0)
        agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(
            torch.float32)
        kinematic_model = KinematicBicycle()
        kinematic_model.set_params(lr=agent_attributes[..., 2])
        kinematic_model.set_state(agent_states)
        renderer = renderer_from_config(
            simulator_cfg.renderer, static_mesh=driving_surface_mesh)

        traffic_controls = {}
        if traffic_light_control is not None:
            traffic_controls["traffic_light"] = traffic_light_control
        if stop_sign_control is not None:
            traffic_controls["stop_sign"] = stop_sign_control
        if yield_control is not None:
            traffic_controls["yield_sign"] = yield_control

        simulator = Simulator(
            cfg=simulator_cfg, road_mesh=driving_surface_mesh,
            kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
            initial_present_mask=dict(vehicle=torch.ones_like(
                agent_states[..., 0], dtype=torch.bool)),
            renderer=renderer,
            traffic_controls=traffic_controls
        )
        simulator = HomogeneousWrapper(simulator)
        npc_mask = torch.ones(
            agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
        npc_mask[0] = False
        if not cfg.ego_only:
            simulator = IAIWrapper(
                simulator=simulator, npc_mask=npc_mask, recurrent_states=[
                    recurrent_states],
                rear_axis_offset=agent_attributes[..., 2:3], locations=[
                    iai_location],
                car_sequences=car_sequences
            )
        if cfg.render_mode == "video":
            simulator = BirdviewRecordingWrapper(
                simulator, res=Resolution(cfg.res, cfg.res), fov=cfg.fov, to_cpu=True)
        return simulator


class WaypointSuiteEnv(GymEnv):
    def __init__(self, cfg: WaypointSuiteEnvConfig):
        self.iai_cfg = cfg.iai_gym
        set_seeds(self.iai_cfg.seed, logger)
        self.locations = cfg.locations
        self.iai_locations = [f'carla:{":".join(location.split("_"))}' for location in self.locations]

        self.waypointsuite = cfg.waypointsuite
        self.car_sequence_suite = cfg.car_sequence_suite
        self.scenarios = cfg.scenarios
        self.lanelet_maps = {}
        self.traffic_light_state_suite = cfg.traffic_light_state_suite
        self.stop_sign_suite = cfg.stop_sign_suite
        for location in self.locations:
            if location not in self.lanelet_maps:
                lanelet_map_path = os.path.join(
                        os.path.dirname(os.path.realpath(
                            __file__)), f"../resources/maps/carla/maps/{location}.osm"
                    )
                self.lanelet_maps[location] = load_lanelet_map(lanelet_map_path)
#        cache_iai_location_info(self.iai_locations)
        super().__init__(cfg=cfg.iai_gym, simulator=None)
#        self.reset()

        logger.info(inspect.getsource(WaypointSuiteEnv.get_reward))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_waypoint_suite_idx = np.random.randint(len(self.waypointsuite))
        location = self.locations[self.current_waypoint_suite_idx]
        while location not in ["Town01", "Town02",  "Town03", "Town07", "Town10HD"]:
#        while location != "Town06":
            self.current_waypoint_suite_idx = np.random.randint(len(self.waypointsuite))
            location = self.locations[self.current_waypoint_suite_idx]
        self.lanelet_map = self.lanelet_maps[location]

        self.set_start_pos()
        self.current_target_idx = 1
        self.current_target = self.waypointsuite[self.current_waypoint_suite_idx][self.current_target_idx]

        self.iai_cfg.ego_state = (self.start_point[0], self.start_point[1], self.start_orientation, self.start_speed)
        self.iai_cfg.center = self.start_point
        self.iai_cfg.location = location

        self.last_x = None
        self.last_y = None
        self.last_psi = None

        self.last_obs = None
        self.last_reward = None
        self.last_info = None

        self.reached_waypoint_num = 0

        self.simulator = build_iai_simulator(self.iai_cfg,
                                             self.scenarios[self.current_waypoint_suite_idx],
                                             self.car_sequence_suite[self.current_waypoint_suite_idx],
                                             self.traffic_light_state_suite[self.current_waypoint_suite_idx],
                                             self.stop_sign_suite[self.current_waypoint_suite_idx])

        self.innermost_simulator = self.simulator.get_innermost_simulator()
        self.innermost_simulator.renderer.set_waypoint_mesh(point_to_mesh(self.current_target, "waypoint"))

        self.environment_steps = 0
        self.occur_exception = False
        return self.get_obs(), {}

    def set_start_pos(self):
        self.waypoints = self.waypointsuite[self.current_waypoint_suite_idx]
        p0 = np.array(self.waypoints[0])
        p1 = np.array(self.waypoints[1])
        # in case the start_point is offroad
        try:
            self.start_point = p0 + np.random.rand() * (p1 - p0)
            self.start_speed = np.random.rand() * 10
            self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map,
                                                                   x=self.start_point[0], y=self.start_point[1])[0]) \
                                     + np.random.normal(0, 0.1)
        except Exception as e:
            logger.warning("Cannot randomly sample a valid start position. Use the first waypoint directly.")
            logger.warning(e)
            self.start_point = p0
            self.start_speed = np.random.rand() * 10
            self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map,
                                                                   x=self.start_point[0], y=self.start_point[1])[0]) \
                                     + np.random.normal(0, 0.1)

    def step(self, action: Tensor):
        state = self.simulator.get_state()
        self.last_x = state[..., 0]
        self.last_y = state[..., 1]
        self.last_psi = state[..., 2]
        self.last_speed = state[..., 3]

        obs, reward, terminated, truncated, info = super().step(action)
        if self.check_reach_target():
            self.current_target_idx += 1
            if self.current_target_idx < len(self.waypoints):
                self.current_target = self.waypoints[self.current_target_idx]
                innermost_simulator = self.simulator.get_innermost_simulator()
                innermost_simulator.renderer.set_waypoint_mesh(point_to_mesh(self.current_target, "waypoint"))
            else:
                self.current_target = None
                innermost_simulator = self.simulator.get_innermost_simulator()
                innermost_simulator.renderer.set_waypoint_mesh(None)
        self.last_obs = obs
        self.last_reward = reward
        self.last_info = info
        return obs, reward, terminated, truncated, info


    def check_reach_target(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        return (self.current_target is not None) and (math.dist((x, y), self.current_target) < 3)

    def get_reward(self):
        offroad_penalty = -10.0 if self.simulator.compute_offroad() > 0 else 0
        collision_penalty = -10.0 if self.simulator.compute_collision() > 0 else 0
        traffic_light_violation_penalty = -10.0 if self.simulator.compute_traffic_lights_violations() > 0 else 0
#        stop_sign_violation_penalty = -10.0 if self.simulator.compute_stop_sign_violations(self.environment_steps) > 0 else 0

        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        psi = self.simulator.get_state()[..., 2]

        d = math.dist((x, y), (self.last_x, self.last_y)) if (self.last_x is not None) and (self.last_y is not None) else 0
        distance_reward = 1 if d > 0.5 else 0
        psi_reward = (1 - math.cos(psi - self.last_psi)) * (-0.05) if (self.last_psi is not None) else 0
#        self.last_x = x
#        self.last_y = y
#        self.last_psi = psi
#        orientation = self.simulator.get_state()[..., 2]
#        speed = self.simulator.get_state()[..., 3]
#        lanelet_orientations = torch.Tensor(find_lanelet_directions(lanelet_map=self.lanelet_map, x=x, y=y)).cuda()
#        if len(lanelet_orientations) > 0:
#            lanelet_orientation = float(lanelet_orientations[torch.argmin(abs(lanelet_orientations - orientation)).item()])
#            orientation_reward = math.cos(orientation - lanelet_orientation)
#        else:
#            orientation_reward = 0
        if self.check_reach_target():
            reach_target_reward = 10
            self.reached_waypoint_num += 1
        else:
            reach_target_reward = 0
        r = torch.zeros_like(x)
#        r += reach_target_reward + offroad_penalty + collision_penalty + traffic_light_violation_penalty + d # stop_sign_violation_penalty + d
        r += reach_target_reward + distance_reward + psi_reward # stop_sign_violation_penalty + d
        return r

    def is_terminated(self):
        if self.iai_cfg.terminated_at_infraction:
            return (self.simulator.compute_offroad() > 0) or (self.simulator.compute_collision() > 0) or ((self.simulator.compute_traffic_lights_violations()) > 0)
        else:
            return False

    def get_info(self):
        psi = self.simulator.get_state()[..., 2]
        speed = self.simulator.get_state()[..., 3]
        reached_waypoint_num = self.reached_waypoint_num
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            is_success=(self.environment_steps >= self.max_environment_steps),
            occur_exception=False,
            reached_waypoint_num=reached_waypoint_num,
            psi_smoothness=((self.last_psi - psi) / 0.1).norm(p=2).item(),
            speed_smoothness=((self.last_speed - speed) / 0.1).norm(p=2).item()
        )
        return self.info


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
