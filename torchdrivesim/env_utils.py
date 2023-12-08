from omegaconf import OmegaConf

from torchdrivesim.rendering import RendererConfig
from torchdrivesim.simulator import TorchDriveConfig
from torchdrivesim.simulator import TorchDriveConfig, CollisionMetric
from torchdrivesim.gym_env import IAIGymEnvConfig, TaskGymEnvConfig, WaypointEnvConfig


def construct_iai_env_config(config):
    iai_gym = IAIGymEnvConfig(**config)
    iai_gym.simulator = TorchDriveConfig(**iai_gym.simulator)
    iai_gym.simulator.renderer = RendererConfig(**iai_gym.simulator.renderer)
    iai_gym.simulator.collision_metric = CollisionMetric.discs
    return iai_gym


def load_task_env_config(yaml_path):
    config_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    task_env_config = TaskGymEnvConfig(**config_from_yaml)
    task_env_config.iai_gym = construct_iai_env_config(task_env_config.iai_gym)
    return task_env_config

def load_waypoint_env_config(yaml_path):
    config_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    waypoint_env_config = WaypointEnvConfig(**config_from_yaml)
    waypoint_env_config.iai_gym = construct_iai_env_config(waypoint_env_config.iai_gym)
    return waypoint_env_config
