
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


class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_dims=[256, 256], device="cpu"):
        super(QNetwork, self).__init__()
        self.device = device
        network = []

        input_channels = n_observations + n_actions
        for n_channels in hidden_dims:
            network.append(nn.Linear(input_channels, n_channels))
            network.append(nn.ReLU())

            input_channels = n_channels

        network.append(nn.Linear(input_channels, 1))

        self.network = nn.Sequential(*network)

        self.to(self.device)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], 1)

        x1 = self.network(x)

        return x1

    
class GaussianPolicy(nn.Module):

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
            # layers.append(nn.InstanceNorm1d(n_channels))
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


class ActorCriticOffPolicy(Agent):
    POLICIES = {"GaussianPolicy": GaussianPolicy}
    CRITICS = {"QNetwork": QNetwork}

    def __init__(
            self,
            env: gym.Env,
            policy: str = "GaussianPolicy",
            critic: str = "QNetwork",
            lr=3e-4,
            weight_decay=1e-5,
            hidden_dims=[256, 256],
            buffer=None,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,  # discount factor
            device="cpu",
            gradient_steps=1,
            learning_starts=1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts

        self.env = env

        policy_builder = self.POLICIES.get(policy, GaussianPolicy)
        critic_builder = self.CRITICS.get(critic, QNetwork)

        self.policy = policy_builder(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.policy_optim = optim.Adam(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic = critic_builder(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_dims=hidden_dims,
            device=self.device,
        )
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        if buffer is None:
            self.buffer = ReplayBuffer(
                buffer_size,
                (self.env.observation_space.shape[0],),
                self.env.action_space.shape[0],
            )
        else:
            self.buffer = buffer

        self.total_num_steps = 0

    def learn(self, episodes: int = 1, episode_steps: int = 100, writer=None, two_pbars=True, tracker=None):
        self.policy.train()
        self.critic.train()
        self._fill_buffer()
        pbar = tqdm(range(episodes), leave=True)
        for episode in pbar:
            
            if tracker is not None:
                tracker.new_episode()
            
            state, info = self.env.reset()
            trajectory_idx = 0
            return_ = 0
            iterator = range(episode_steps)
            if two_pbars:
                iterator = tqdm(iterator, leave=False)
            for i in iterator:
                action = self._choose_action(state).cpu().detach().numpy()[0]
                next_state, reward, terminated, truncated, info = self.env.step(action)
                return_ += reward
                if tracker is not None:
                    tracker.add(reward)
                self.buffer.store_transition(state, action, reward, next_state, terminated, trajectory_idx)
                trajectory_idx += 1
                if self.buffer.buffer_total > self.learning_starts:
                    for j in range(self.gradient_steps):
                        self._learn_step(writer=writer)
                state = next_state
                self.total_num_steps += 1
                
                if terminated or truncated:
                    if two_pbars:
                        iterator.disp(close=True)
                    break
            pbar.set_description(f"total steps: {self.total_num_steps}, episode: {episode}, return: {return_:.4f}")

    def predict(self, state):
        self.policy.eval()
        
        state = torch.tensor(state).float().to(self.device)
            
        if len(state.shape) != 2:
            state = state.unsqueeze(0)

        action, log_probs = self.policy.sample(state, greedy=True)
        
        action = action.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()
            
        return action, log_probs

    def _fill_buffer(self):
        state, info = self.env.reset()
        while len(self.buffer) < self.batch_size:
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.buffer.store_transition(state, action, reward, next_state, terminated)
            state = next_state

    def _learn_step(self, writer=None):
        state, action, reward, next_state, terminal = self.buffer.sample_buffer(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal, dtype=torch.bool).to(self.device).view(-1, 1)

        # calc next q values for TD update
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_state)
            next_q_value = reward + torch.logical_not(terminal) * self.gamma * self.critic(next_state, next_actions)

        # train critic/values with mse loss
        critic = self.critic(state, action)
        critic_loss = F.mse_loss(critic, next_q_value)
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # train actor/policy with TD error
        actions, log_probs = self.policy.sample(state)
        next_critic = self.critic(state, actions)
        policy_loss = - next_critic.mean()
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 0.5)
        self.policy_optim.step()

    def _choose_action(self, state):
        state_tensor = torch.tensor(np.array([state])).float().to(self.device)
        action, _ = self.policy.sample(state_tensor)
        return action

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))

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