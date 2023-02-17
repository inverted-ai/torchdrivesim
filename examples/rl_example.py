"""
A simple version of Proximal Policy Optimization for driving in a minimal environment.
Note that the behavior of other agents are provided by the IAI API, which requires an access key.
However, it is easy to modify the environment to provide alternative means of controlling other agents.
"""
import torch.nn as nn
from gym_env import *
import warnings
from tqdm import trange
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DictDataset(torch.utils.data.TensorDataset):

    def __init__(self, data_dict):
        self.data = data_dict
        self.data_keys = data_dict.keys()

    def __getitem__(self, index):
        return {key:self.data[key][index] for key in self.data_keys}

    def __len__(self):
        return [len(self.data[key]) for key in self.data_keys][0]


class RolloutStorage:

    def __init__(self, num_steps, num_processes, obs_space, action_space):
        if obs_space.shape is None:
            spaces_names = list(obs_space.spaces.keys())
            spaces_shapes = [obs_space.spaces[key].shape for key in spaces_names]
            spaces_dict = {key:value for (key, value) in zip(spaces_names, spaces_shapes)}
        else:
            spaces_names = ['obs']
            spaces_shapes = [obs_space.shape]
            spaces_dict = {key:value for (key, value) in zip(spaces_names,spaces_shapes)}
        self.spaces_dict = spaces_dict
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.obs_keys = spaces_names
        self.obs = {key:torch.zeros(num_steps + 1, num_processes, *obs_shape) for (key, obs_shape) in zip(spaces_names,spaces_shapes)}
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for key in self.obs_keys:
            self.obs[key] = self.obs[key].to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, next_obs, actions, action_log_probs,
               value_preds, rewards, next_masks, next_bad_masks, k=None):
        for key in self.obs_keys:
            self.obs[key][self.step + 1].copy_(next_obs[key])
        self.actions[self.step].copy_(actions.view(self.actions[self.step].size()))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(next_masks)
        self.bad_masks[self.step + 1].copy_(next_bad_masks)
        if k is not None:
            assert self.step == k
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for key in self.obs_keys:
            self.obs[key][0].copy_(self.obs[key][-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, gamma=0.975, gae_lambda=0.995):
        self.value_preds[-1], gae = next_value, 0.
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step +1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step +  1] * gae
            self.returns[step] = gae + self.value_preds[step]


class FixedNormal(torch.distributions.Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

    def std(self):
        return self.std


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs, bounds=None, param_sd=None, zero_mean=True):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = torch.nn.Parameter(- 0.25 * torch.ones(num_outputs))

    def get_mean(self, x):
        return self.fc_mean(x)

    def get_std(self, sd_input, min_std=1e-12):
        return self.logstd.exp() + min_std

    def forward(self, x, sd_input=None):
        return FixedNormal(self.get_mean(x),  self.get_std(sd_input))


class NNBase(nn.Module):

    def __init__(self, obs_shape, num_layers=2, num_filters=32,
                     feature_dim=50, stride=(2,2),  kernel_size=(3,3), output_logits=False):
        super().__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.output_logits = output_logits
        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, kernel_size, stride=stride)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride))
        fake_input = torch.zeros(obs_shape).unsqueeze(0)
        conv = torch.relu(self.convs[0](fake_input))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        out_dim = conv.view(conv.size(0), -1).size()[1]
        self.fc = nn.Linear(out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs):
        if len(obs.size()) > len(self.obs_shape)+1:
            obs = obs.squeeze(0)
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = conv.view(conv.size(0), -1)

        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs['birdview_image'])
        h = h.reshape(h.shape[0],-1)
        if detach:
            h = h.detach()
        out = self.fc(h)
        out = self.ln(out)
        out = torch.tanh(out)
        return out


class ActorCritic(nn.Module):

    def __init__(self, obs_shape, action_space, base_kwargs):
        super(ActorCritic, self).__init__()
        self.action_space = action_space
        self.observation_space = obs_shape
        self.feature_encoder = NNBase(obs_shape,feature_dim=50)
        self.critic = torch.nn.Linear(self.feature_encoder.feature_dim,1)
        self.dist = DiagGaussian(self.feature_encoder.feature_dim, action_space, bounds=None, param_sd=None)

    def act(self, inputs, deterministic=False):
        state_value, features = self.forward(inputs)
        action_dist = self.dist(features)
        action = action_dist.mode() if deterministic else action_dist.sample()
        action_log_probs = action_dist.log_probs(action)
        return state_value, action, action_log_probs

    def get_value(self, inputs):
        state_value, features = self.forward(inputs)
        return state_value

    def evaluate_actions(self, inputs, actions, detach_encoder=False):
        state_value, features = self.forward(inputs)
        action_dist = self.dist(features)
        action_log_probs = action_dist.log_probs(actions)
        dist_entropy = action_dist.entropy().mean()
        return state_value, action_log_probs, dist_entropy

    def forward(self,inputs):
        features = self.feature_encoder(inputs)
        state_value = self.critic(features)
        return state_value, features


class PPOTrainer:

    def __init__(self, envs_gen):
        super(PPOTrainer, self).__init__()
        self.env_gen, self.envs = envs_gen, envs_gen()
        self.gamma = 0.975
        self.num_processes = 1
        self.num_steps = 512
        self.device = "cuda:0"
        self.actor_critic = ActorCritic([3, 64, 64], 2, base_kwargs={})
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.max_grad_norm = 10.
        self.ppo_clip = 0.2
        self.entropy_coeff = 0.01
        self.policy_updates = 1
        self.critic_updates = 1
        self.batch_size = 128
        self.steps, self.time_steps = 0, 0
        self.rollouts = RolloutStorage(self.num_steps, 1, self.envs.observation_space, self.envs.action_space)
        self.obs = self.envs.reset()
        for key in list(self.rollouts.obs.keys()):
            if self.obs[key] is not None:
                self.rollouts.obs[key][0].copy_(self.obs[key])
        self.actor_critic.to(self.device)
        self.rollouts.to(self.device)

    def evaluate_policy(self, traces=3):
        with torch.no_grad():
            eval_env = self.env_gen()
            obs = eval_env.reset()
            avg_reward, done = 0., 0
            for eval_run in range(traces):
                while not done:
                    obs['birdview_image'] = obs['birdview_image'].unsqueeze(0)
                    value, action, action_log_prob = self.actor_critic.act(obs, deterministic=True)
                    obs, reward, done, infos = eval_env.step(action.squeeze(0))
                    avg_reward += reward
                obs = eval_env.reset()
            return avg_reward / traces

    def sample(self):
        with torch.no_grad():
            for step in range(self.num_steps):
                self.obs['birdview_image'] = self.obs['birdview_image'].unsqueeze(0)
                value, action, action_log_prob = self.actor_critic.act(self.obs)
                self.obs, reward, done, infos = self.envs.step(action.squeeze(0))
                bad_done = 1 * torch.tensor(self.envs.environment_steps >= self.envs.max_environment_steps)
                if done:
                    self.obs = self.envs.reset()
                next_masks = 1 * done.unsqueeze(0)
                next_bad_masks = 1 * bad_done.unsqueeze(0)
                self.rollouts.insert(self.obs, action, action_log_prob, value, reward, next_masks, next_bad_masks, k=step)
            self.time_steps += self.num_processes * self.num_steps
            with torch.no_grad():
                next_value = self.actor_critic.get_value({key:self.rollouts.obs[key][-1] for key in self.obs.keys()}).detach()
                next_value = next_value*self.rollouts.masks[-1]
                self.rollouts.compute_returns(next_value, self.gamma)
            return self.rollouts

    def policy_update(self, optim, model, rollouts, reshaped_obs, actor_epochs, clip_eps, detach_encoder=True):
        action_shape, action_loss_avg = rollouts.actions.size()[-1], 0.
        num_steps, num_processes, _ = rollouts.rewards.size()
        values, action_log_probs, dist_entropy = model.evaluate_actions(
            reshaped_obs, rollouts.actions.view(-1, action_shape), detach_encoder=False)
        values = values.view(num_steps, num_processes, 1)
        actions = rollouts.actions.view(-1, action_shape)
        advantages = rollouts.returns[:-1].detach() - values
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)
        reshaped_obs.update({'advantages': advantages.view(-1,1).detach(),
                             'actions': actions.detach(),
                             'action_log_probs':action_log_probs.detach()})
        data_loader = torch.utils.data.DataLoader(DictDataset(reshaped_obs), batch_size=self.batch_size, shuffle=True)
        for _ in range(actor_epochs):
            for r, batch in enumerate(data_loader):
                value, new_log_probs, dist_entropy = model.evaluate_actions(batch, batch['actions'])
                ratio = torch.exp(new_log_probs - batch['action_log_probs'] )
                clipped_ratio = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
                loss = -torch.min(ratio * batch['advantages'], clipped_ratio * batch['advantages'] ).mean() - self.entropy_coeff * dist_entropy
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optim.step()
                action_loss_avg += loss.detach()
        action_loss = action_loss_avg.item() / (r+1)
        return action_loss, dist_entropy.item()

    def critic_update(self, optim, model, rollouts, reshaped_obs,  critic_epochs, detach_encoder=False):
        value_target, value_loss_avg = rollouts.returns[:-1].detach(), 0.
        reshaped_obs.update({'value_target': value_target.reshape(-1,1).detach()})
        data_loader = torch.utils.data.DataLoader(DictDataset(reshaped_obs), batch_size=self.batch_size, shuffle=True)
        for _ in range(critic_epochs):
            for r, batch in enumerate(data_loader):
                value_loss = (batch['value_target'] - model.get_value(batch)).pow(2).mean()
                optim.zero_grad()
                value_loss.backward()
                optim.step()
            value_loss_avg += value_loss
        return value_loss_avg.item() / (r+1)

    def train_step(self):
        rollouts = self.sample()
        spaces_names = [key for key in rollouts.obs.keys()  if rollouts.obs[key] is not None]
        spaces_shapes = [rollouts.obs[key].size()[2:] for key in spaces_names]
        spaces_dict = {key:value for (key, value) in zip(spaces_names,spaces_shapes)}
        reshaped_obs = {key:rollouts.obs[key][:-1].view(-1, *spaces_dict[key]) for key in spaces_names}
        value_loss = self.critic_update(self.optimizer, self.actor_critic,
                        rollouts, reshaped_obs, critic_epochs=self.critic_updates,
                        detach_encoder=False)
        action_loss, dist_entropy = self.policy_update(
                    self.optimizer, self.actor_critic, rollouts, reshaped_obs,
                    actor_epochs=self.policy_updates, clip_eps=self.ppo_clip,
                    detach_encoder=True)
        with torch.no_grad():
            self.steps += 1
            self.rollouts.after_update()


def rl_trainer(cfg: TorchDriveGymEnvConfig):
    env_gen = lambda: gym.make('torchdrivesim/IAI-v0', args=cfg)
    policy_trainer = PPOTrainer(env_gen)
    print('Training Reinforcement Learning Agent:')
    progress_bar = trange(100, leave=True)
    for i in progress_bar:
        policy_trainer.train_step()
        progress_bar.set_description(f"Average Return: {policy_trainer.evaluate_policy().item():.2f}")
    return policy_trainer


if __name__ == "__main__":
    cli_cfg: TorchDriveGymEnvConfig = OmegaConf.structured(
        TorchDriveGymEnvConfig(**OmegaConf.from_dotlist(sys.argv[1:]))
    )
    rl_trainer(cli_cfg)
