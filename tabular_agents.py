import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

    
class TDZeroPrediction(Agent):
    # random policy
    
    def __init__(self, env, lr=1e-1, gamma=.9, policy=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.state_values = np.zeros(env.observation_space.n)
        self.total_num_steps = 0
        self.policy = policy
        
    def update(self, state, action, reward, new_state):
        td_error = reward + self.gamma*self.state_values[new_state] - self.state_values[state]
        self.state_values[state] += self.lr*td_error
    
    def learn(self, episodes, max_steps=None, tracker=None, verb=True):
        
        iterator = range(episodes)
        if verb:
            iterator = tqdm(iterator, leave=True)
        
        for episode in iterator:
            
            if tracker is not None:
                tracker.new_episode()
                
            state, info = self.env.reset()
            
            steps = 0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                
                if self.policy is None:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy(state)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                #update
                self.update(state, action, reward, new_state)

                self.total_num_steps += 1
                steps += 1
                
                if tracker is not None:
                    tracker.add(reward)
                
                if max_steps is not None:
                    if steps > max_steps:
                        break
    
    def predict(self, state):
        if self.policy is None:
            action = self.env.action_space.sample()
        else:
            action = self.policy(state)
        return action


class TDZeroControl(Agent):
    
    def __init__(self, env, lr=1e-2, gamma=.9, epsilon=.1, off_policy=False):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.total_num_steps = 0
        self.off_policy = off_policy    
    
    def _greedy(self):
        return np.random.choice(a=[False, True], p=[self.epsilon, 1-self.epsilon])
    
    def _choose_action(self, state, greedy=False):
        if greedy:
            action = np.argmax(self.Q_values[state])
        else:
            action = self.env.action_space.sample()
        return action
    
    def update(self, state, action, reward, new_state):
        new_action = self._choose_action(new_state, greedy=True if self.off_policy else self._greedy()) 
        td_error = reward + self.gamma*self.Q_values[new_state, new_action] - self.Q_values[state, action] 
        self.Q_values[state, action] += self.lr*td_error 
    
    def learn(self, episodes, max_steps=None, exploration_scheme=None, tracker=None, verb=True):
        
        iterator = range(episodes)
        if verb:
            iterator = tqdm(iterator, leave=True)
        
        for episode in iterator:
            
            # update epsilon
            if exploration_scheme is not None:
                max_epsilon = exploration_scheme['max_epsilon']
                min_epsilon = exploration_scheme['min_epsilon']
                decay_rate = exploration_scheme['decay_rate']
                self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
            
            if tracker is not None:
                tracker.new_episode()
            
            state, info = self.env.reset()
            
            steps = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                
                action = self._choose_action(state, self._greedy()) 
                new_state, reward, terminated, truncated, info = self.env.step(action) 

                #update
                self.update(state, action, reward, new_state)
                
                state = new_state
                self.total_num_steps += 1
                steps += 1
                
                if tracker is not None:
                    tracker.add(reward)
                
                if max_steps is not None:
                    if steps > max_steps:
                        break
    
    def predict(self, state):
        return self._choose_action(state, greedy=True)
