from omegaconf import OmegaConf

from torchdrivesim.rendering import RendererConfig
from torchdrivesim.simulator import TorchDriveConfig
from torchdrivesim.simulator import TorchDriveConfig, CollisionMetric
from torchdrivesim.gym_env import IAIGymEnvConfig, WaypointSuiteEnvConfig, Scenario


def construct_iai_env_config(config):
    iai_gym = IAIGymEnvConfig(**config)
    iai_gym.simulator = TorchDriveConfig(**iai_gym.simulator)
    iai_gym.simulator.renderer = RendererConfig(**iai_gym.simulator.renderer)
    iai_gym.simulator.collision_metric = CollisionMetric.discs
    return iai_gym


def load_waypoint_suite_env_config(yaml_path):
    config_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    waypoint_suite_env_config = WaypointSuiteEnvConfig(**config_from_yaml)
    waypoint_suite_env_config.iai_gym = construct_iai_env_config(waypoint_suite_env_config.iai_gym)
    if waypoint_suite_env_config.scenarios is not None:
        waypoint_suite_env_config.scenarios = [Scenario(agent_states=scenario["agent_states"],
                                                        agent_attributes=scenario["agent_attributes"],
                                                        recurrent_states=scenario["recurrent_states"])
                                               if scenario is not None else None for scenario in waypoint_suite_env_config.scenarios]
    return waypoint_suite_env_config
