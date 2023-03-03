
import os
import time
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim
from tqdm.auto import tqdm

from utils import ReplayBuffer, Agent

import torch
import torch.nn as nn


class QAttentionNetwork(nn.Module):
    # TODO
    def __init__(self, n_observations, n_actions, hidden_dims=[256, 256], device="cpu"):
        super(QNetwork, self).__init__()
        self.device = device
        network_1 = []
        network_2 = []

        input_channels = n_observations + n_actions
        for n_channels in hidden_dims:
            network_1.append(nn.Linear(input_channels, n_channels))
            # network_1.append(nn.InstanceNorm1d(n_channels))
            network_1.append(nn.ReLU())

            network_2.append(nn.Linear(input_channels, n_channels))
            # network_2.append(nn.InstanceNorm1d(n_channels))
            network_2.append(nn.ReLU())

            input_channels = n_channels

        network_1.append(nn.Linear(input_channels, 1))
        network_2.append(nn.Linear(input_channels, 1))

        self.network_1 = nn.Sequential(*network_1)
        self.network_2 = nn.Sequential(*network_2)

        self.to(self.device)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], 1)

        x1 = self.network_1(x)
        x2 = self.network_2(x)

        return x1, x2

    
class GaussianAttentionPolicy(nn.Module):
    # TODO
    def __init__(
        self,
        n_observations,
        n_actions,
        hidden_dims=[256, 256],
        device="cpu",
        noise=1e-6,
    ):
        super(GaussianPolicy, self).__init__()
        self.device = device
        self.noise = noise

        layers = []
        input_channels = n_observations
        for n_channels in hidden_dims:
            layers.append(nn.Linear(input_channels, n_channels))
            layers.append(nn.ReLU())
            input_channels = n_channels

        self.network = nn.Sequential(*layers)
        self.mu = nn.Linear(input_channels, n_actions)
        self.log_std = nn.Linear(input_channels, n_actions)

        self.to(self.device)

    def forward(self, observations):
        x = self.network(observations)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, observations, greedy=False):
        mu, log_std = self.forward(observations)
        sigma = log_std.exp()

        probs = Normal(mu, sigma)
        sample = probs.rsample() if not greedy else mu
        actions = torch.tanh(sample)

        log_probs = probs.log_prob(sample)
        log_probs -= torch.log(1 - actions.pow(2) + self.noise)
        log_probs = log_probs.sum(dim=1, keepdim=True)

        return actions, log_probs


class AttentionSoftActorCritic(Agent):

    def __init__(
            self,
            env: gym.Env,
            lr=3e-4,
            weight_decay=1e-5,
            hidden_dims=[256, 256],
            buffer=None,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,  # update factor
            gamma=0.99,  # discount factor
            temperature=0.2,  # initial entropy coefficient
            target_update_interval=1,
            device="cpu",
            entropy_tuning=True,  # activate automatic entropy tuning
            gradient_steps=1,
            learning_starts=1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.alpha = torch.FloatTensor([temperature]).to(self.device)
        self.target_update_interval = target_update_interval
        self.entropy_tuning = entropy_tuning
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts

        self.env = env

        self.policy = GaussianAttentionPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.policy_optim = optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic = QAttentionNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.target_critic = deepcopy(self.critic)

        if buffer is None:
            self.buffer = ReplayBuffer(
                buffer_size,
                (self.env.observation_space.shape[0],),
                self.env.action_space.shape[0],
            )
        else:
            self.buffer = buffer

        self.target_entropy = -torch.prod(
            torch.tensor(self.env.action_space.shape).to(self.device)
        ).item()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam(
            [self.log_alpha], lr=lr, weight_decay=weight_decay
        )
        self.total_num_steps = 0
        self.state_trajectory = []
        
    def get_context(self, to_torch=True):
        state_trajectory = np.array(self.state_trajectory)
        state_trajectory = state_trajectory.reshape(1,state_trajectory.shape[0],state_trajectory.shape[1])
        if to_torch:
            state_trajectory = torch.from_numpy(state_trajectory).to(self.device)
        return state_trajectory
    
    def add_context(self, state):
        self.state_trajectory.append(state)
    
    def clear_context(self):
        self.state_trajectory = []

    def learn(self, episodes: int = 1, episode_steps: int = 100, writer=None, two_pbars=True, tracker=None):
        self.policy.train()
        self.critic.train()
        self.target_critic.train()
        self._fill_buffer()
        pbar = tqdm(range(episodes), leave=True)
        for episode in pbar:
            
            if tracker is not None:
                tracker.new_episode()
            
            self.clear_context()
            state, info = self.env.reset()  # is_training=True
            
            self.add_context(state)
            trajectory_idx = 0
            return_ = 0
            iterator = range(episode_steps)
            if two_pbars:
                iterator = tqdm(iterator, leave=False)
            for i in iterator:
                action = self._choose_action(self.get_context()).detach().cpu().numpy()[0]
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.add_context(next_state)
                return_ += reward
                if tracker is not None:
                    tracker.add(reward)
                self.buffer.store_transition(state, action, reward, next_state, terminated, trajectory_idx)
                trajectory_idx += 1
                if self.buffer.buffer_total > self.learning_starts:
                    for j in range(self.gradient_steps):
                        update_target_value = True if self.buffer.buffer_total % self.target_update_interval == 0 else False
                        self._learn_step(update_target_value=update_target_value, writer=writer)
                state = next_state
                self.total_num_steps += 1
                
                if terminated or truncated:
                    if two_pbars:
                        iterator.disp(close=True)
                    break
            pbar.set_description(f"total steps: {self.total_num_steps}, episode: {episode}, return: {return_:.4f}")

    def predict(self):
        # add context after step manually!
        self.policy.eval()
        
        state = torch.tensor(self.get_context()).float().to(self.device)
            
        if len(state.shape) != 3:
            state = state.unsqueeze(0)

        action, log_probs = self.policy.sample(state, greedy=True)
        
        action = action.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()
            
        return action, log_probs

    def _fill_buffer(self):
        while len(self.buffer) < self.batch_size:
            state, info = self.env.reset()
            trajectory_idx = 0
            while not done and not truncated:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.buffer.store_transition(state, action, reward, next_state, terminated, trajectory_idx)
                trajectory_idx += 1
                state = next_state

    def _learn_step(self, update_target_value, writer=None):
        # TODO
        state, action, reward, next_state, terminal = self.buffer.sample_buffer(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(self.device).view(-1, 1)

        # calc next q values for TD update
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state)
            next_critic1_target, next_critic2_target = self.target_critic(next_state, next_actions)
            next_min_critic_target = torch.min(next_critic1_target, next_critic2_target) - self.alpha * next_log_probs
            next_q_value = reward + torch.logical_not(terminal) * self.gamma * next_min_critic_target

        # train critic/values with mse loss
        critic_1, critic_2 = self.critic(state, action)
        critic_1_loss = F.mse_loss(critic_1, next_q_value)
        critic_2_loss = F.mse_loss(critic_2, next_q_value)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # train actor/policy with TD error
        actions, log_probs = self.policy.sample(state)
        next_critic_1, next_critic_2 = self.critic(state, actions)
        next_min_q = torch.min(next_critic_1, next_critic_2)
        policy_loss = ((self.alpha * log_probs) - next_min_q).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 0.5)
        self.policy_optim.step()

        if self.entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        if update_target_value:
            self._update_target_value()

        if writer is not None:
            writer.add_scalar("loss/policy", policy_loss.item(), self.total_num_steps)
            writer.add_scalar(
                "loss/critic_1", critic_1_loss.item(), self.total_num_steps
            )
            writer.add_scalar(
                "loss/critic_2", critic_2_loss.item(), self.total_num_steps
            )
            if self.entropy_tuning:
                writer.add_scalar(
                    "loss/entropy", alpha_loss.item(), self.total_num_steps
                )
                writer.add_scalar(
                    "alpha",
                    self.alpha.clone().detach().cpu().numpy(),
                    self.total_num_steps,
                )

    def _choose_action(self, context):
        context_tensor = torch.tensor(context).float().to(self.device)
        action, _ = self.policy.sample(context_tensor)
        return action

    def _update_target_value(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_critic_param, critic_param in zip(
                self.target_critic.parameters(), self.critic.parameters()
        ):
            target_critic_param.data.copy_(
                target_critic_param.data * (1.0 - tau) + critic_param.data * tau
            )

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(
            self.target_critic.state_dict(), os.path.join(path, "target_critic.pt")
        )

    @classmethod
    def load(
            cls,
            env,
            path,
            policy: str = "GaussianPolicy",
            critic: str = "QNetwork",
            device="cpu",
    ):
        """
        Loads only the policy weights for inference.
        """
        sac = cls(env, policy, critic)
        sac.policy.load_state_dict(
            torch.load(os.path.join(path, "policy.pt"), map_location=device)
        )
        return sac

    def train(self):
        self.policy.train()
        self.critic.train()

    def eval(self):
        self.policy.eval()
        self.critic.eval()