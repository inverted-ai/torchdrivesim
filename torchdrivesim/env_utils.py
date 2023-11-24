from omegaconf import OmegaConf

from torchdrivesim.rendering import RendererConfig
from torchdrivesim.simulator import TorchDriveConfig
from torchdrivesim.simulator import TorchDriveConfig, CollisionMetric
from torchdrivesim.gym_env import IAIGymEnvConfig, TaskGymEnvConfig


def load_task_env_config(yaml_path):
    config_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    task_env_config = TaskGymEnvConfig(**config_from_yaml)
    task_env_config.iai_gym = IAIGymEnvConfig(**task_env_config.iai_gym)
    task_env_config.iai_gym.simulator = TorchDriveConfig(**task_env_config.iai_gym.simulator)
    task_env_config.iai_gym.simulator.renderer = RendererConfig(**task_env_config.iai_gym.simulator.renderer)
    task_env_config.iai_gym.simulator.collision_metric = CollisionMetric.discs
    return task_env_config
