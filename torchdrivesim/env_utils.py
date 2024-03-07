from omegaconf import OmegaConf
from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

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


class EvalNTimestepsCallback(BaseCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param eval_n_episodes: How many episodes to evaluate each time
    """
    def __init__(self, eval_env, n_steps: int, eval_n_episodes: int, deterministic=False, log_tab="eval"):
        super().__init__()
        self.log_tab=log_tab
        self.n_steps = n_steps
        self.eval_n_episodes = eval_n_episodes
        self.deterministic = deterministic
        self.last_time_trigger = 0
        self.eval_env = eval_env

    def _calc_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        Called after each step
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if "psi_smoothness" not in info:
            return
        self.psi_smoothness_for_single_episode.append(info["psi_smoothness"])
        self.speed_smoothness_for_single_episode.append(info["speed_smoothness"])
        if (info["offroad"] > 0) or (info["collision"] > 0) or (info["traffic_light_violation"] > 0) \
                                 or (info["occur_exception"]) or (info["is_success"]):
            self.episode_num += 1

            if info["offroad"] > 0:
                self.offroad_num += 1
            if info["collision"] > 0:
                self.collision_num += 1
            if info["traffic_light_violation"] > 0:
                self.traffic_light_violation_num += 1
            if info["occur_exception"]:
                self.exception_num += 1
            if info["is_success"]:
                self.success_num += 1
            self.reached_waypoint_nums.append(info["reached_waypoint_num"])
            if len(self.psi_smoothness_for_single_episode) > 0:
                self.psi_smoothness.append(sum(self.psi_smoothness_for_single_episode) / len(self.psi_smoothness_for_single_episode))
            if len(self.speed_smoothness_for_single_episode) > 0:
                self.speed_smoothness.append(sum(self.speed_smoothness_for_single_episode) / len(self.speed_smoothness_for_single_episode))


    def _evaluate(self) -> bool:
        self.episode_num = 0
        self.offroad_num = 0
        self.collision_num = 0
        self.traffic_light_violation_num = 0
        self.exception_num = 0
        self.success_num = 0
        self.reached_waypoint_nums = []
        self.psi_smoothness = []
        self.speed_smoothness = []

        mean_episode_reward = 0
        mean_episode_length = 0
        for i in range(self.eval_n_episodes):
            self.psi_smoothness_for_single_episode = []
            self.speed_smoothness_for_single_episode = []
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=1,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                callback=self._calc_metrics,
            )
            mean_episode_reward += sum(episode_rewards) / len(episode_rewards)
            mean_episode_length += sum(episode_lengths) / len(episode_lengths)

        mean_episode_reward /= self.eval_n_episodes
        mean_episode_length /= self.eval_n_episodes

        self.logger.record(f"{self.log_tab}/mean_episode_reward", mean_episode_reward)
        self.logger.record(f"{self.log_tab}/mean_episode_length", mean_episode_length)

        self.logger.record(f"{self.log_tab}/offroad_rate", self.offroad_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/collision_rate", self.collision_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/traffic_light_violation_rate", self.traffic_light_violation_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/exception_rate", self.exception_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/success_percentage", self.success_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/reached_waypoint_num", sum(self.reached_waypoint_nums) / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/psi_smoothness", sum(self.psi_smoothness) / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/speed_smoothness", sum(self.speed_smoothness) / self.eval_n_episodes)


    def _on_training_start(self) -> None:
        self._evaluate()


    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            self._evaluate()
        return True
