import gym
import numpy as np
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tqdm.auto import tqdm

from utils import Agent


class PolicyNet(nn.Module):

    def __init__(self, n_observations, n_actions, nodes=256, noise=1e-6, device='cpu'):
        super(PolicyNet, self).__init__()
        self.nodes = nodes
        self.fc1 = nn.Linear(n_observations, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.mu = nn.Linear(nodes, n_actions)
        self.log_std = nn.Linear(nodes, n_actions)
        self.noise = noise
        
        self.device = device
        self.to(self.device)

    def forward(self, observations):
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-10, max=2)
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
    

class ValueNet(nn.Module):

    def __init__(self, n_observations, nodes=256, device='cpu'):
        super(ValueNet, self).__init__()
        self.nodes = nodes
        self.fc1 = nn.Linear(n_observations, nodes)
        self.fc2 = nn.Linear(nodes, nodes)
        self.fc3 = nn.Linear(nodes, 1)
        
        self.device = device
        self.to(self.device)

    def forward(self, observations):
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    
class ActorCritic(Agent):
    
    def __init__(self, env, lr_policy=2e-5, lr_critic=1e-3, gamma=.99, device='cpu'):
        self.env = env
        self.lr_policy = lr_policy
        self.lr_critic = lr_critic
        self.gamma = torch.FloatTensor([gamma]).to(device)
        
        self.policy = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0], device=device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        
        self.critic = ValueNet(env.observation_space.shape[0], device=device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.total_num_steps = 0
        self.device = device
    
    def update(self, state, action, log_prob, reward, new_state, steps):
        
        # train critic
        value = self.critic(state)
        new_value = self.critic(new_state)
        critic_loss = F.mse_loss(reward + self.gamma*new_value.detach(), value)
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        
        # train policy
        td_error = reward + self.gamma * new_value.detach() - value.detach()
        policy_loss = - td_error * log_prob  # * self.gamma ** steps 
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 0.5)
        self.policy_optim.step()
        
    
    def learn(self, episodes, max_steps=None, tracker=None, verb=True):
        
        self.train()
        
        iterator = range(episodes)
        if verb:
            iterator = tqdm(iterator, leave=True)
        
        for episode in iterator:

            if tracker is not None:
                tracker.new_episode()
            
            state, info = self.env.reset()
            state = torch.tensor(state).float().reshape(1,-1).to(self.device)
            
            returns = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                
                # pdb.set_trace()
                action, log_prob = self.policy.sample(state)
                
                new_state, reward, terminated, truncated, info = self.env.step(action.flatten().detach().cpu().numpy())
                new_state = torch.tensor(new_state).float().reshape(1,-1).to(self.device)

                #update
                self.update(state, action, log_prob, reward, new_state, steps)
                
                state = new_state
                self.total_num_steps += 1
                steps += 1
                returns += reward
                
                if tracker is not None:
                    tracker.add(reward)
                
                if max_steps is not None:
                    if steps > max_steps:
                        break
            
            if verb:
                iterator.set_description(f"total steps: {self.total_num_steps}, episode: {episode}, return: {returns:.4f}")
    
    def predict(self, state):
        self.policy.eval()
        
        state = torch.tensor(state).float().to(self.device)
            
        if len(state.shape) != 2:
            state = state.unsqueeze(0)

        action, log_probs = self.policy.sample(state, greedy=True)
        
        action = action.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()
            
        return action, log_probs
    
    def train(self):
        self.policy.train()
        self.critic.train()

    def eval(self):
        self.policy.eval()
        self.critic.eval()