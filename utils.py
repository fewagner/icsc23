import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class ReturnTracker():
    
    def __init__(self):
        self.returns = []
        self.steps = []
        self.collected_rewards = 0
        self.step = 0
    
    def new_episode(self):
        if self.step > 0:
            self.returns.append(self.collected_rewards)
            self.steps.append(self.step)
            self.collected_rewards = 0
            self.step = 0
    
    def add(self, reward):
        self.collected_rewards += reward
        self.step += 1
    
    def plot(self, title=None, smooth=1):
        
        returns = np.array(self.returns)
        steps = np.array(self.steps)
        returns = returns[np.array(self.steps) > 0]
        steps = steps[np.array(self.steps) > 0]
        x_axis = np.arange(len(steps))
        
        if smooth > 1:
            cut = len(steps) - len(steps) % smooth
            x_axis = x_axis[:cut]
            steps = steps[:cut]
            returns = returns[:cut]
            x_axis = np.floor(np.mean(x_axis.reshape(-1, smooth), axis=1))
            steps = np.mean(steps.reshape(-1, smooth), axis=1)
            returns = np.mean(returns.reshape(-1, smooth), axis=1)
        
        plt.plot(x_axis, returns/steps)
        plt.ylabel('Average Return')
        plt.xlabel('Episodes')
        plt.title(title)
        plt.show()
    
    def average(self):
        return np.mean(np.array(self.returns)[np.array(self.steps) > 0]/np.array(self.steps)[np.array(self.steps) > 0])
    
    def get_data(self):
        return np.array(self.steps)[np.array(self.steps) > 0], np.array(np.array(self.returns)[np.array(self.steps) > 0])
    
class Agent():
    
    def __init__(self):
        raise NotImplemente('Agent class requires that you implement __init__!')
    
    def learn(self):
        raise NotImplemente('Agent class requires that you implement learn!')
    
    def predict(self, state):
        raise NotImplemente('Agent class requires that you implement predict!')

    


class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.buffer_total = 0

        self.state_memory = np.zeros((self.buffer_size, *input_shape))
        self.next_state_memory = np.zeros((self.buffer_size, *input_shape))
        self.action_memory = np.zeros((self.buffer_size, n_actions))
        self.reward_memory = np.zeros(self.buffer_size)
        self.terminal_memory = np.zeros(self.buffer_size, dtype=np.bool)
        self.trajectory_idx = np.zeros(self.buffer_size, dtype=np.int)

    def store_transition(self, state, action, reward, next_state, terminal, trajectory_idx=None):
        idx = self.buffer_counter % self.buffer_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = terminal
        if trajectory_idx is not None:
            self.trajectory_idx[idx] = trajectory_idx

        self.buffer_counter += 1
        self.buffer_total += 1

    def sample_buffer(self, batch_size):
        if self.buffer_counter < self.buffer_size:
            batch_idxs = np.random.choice(self.buffer_counter, batch_size)
        else:
            batch_idxs = np.random.choice(self.buffer_size, batch_size)

        states = self.state_memory[batch_idxs]
        next_states = self.next_state_memory[batch_idxs]
        actions = self.action_memory[batch_idxs]
        rewards = self.reward_memory[batch_idxs]
        terminals = self.terminal_memory[batch_idxs]

        return states, actions, rewards, next_states, terminals

    def sample_trajectories(self, batch_size, steps, burn_in=0):
        if self.buffer_counter < self.buffer_size:
            start_idxs = np.random.choice(self.buffer_counter - steps, size=batch_size)
            idxs = start_idxs.reshape(-1, 1) * np.ones((batch_size, steps), dtype=int) + np.arange(steps,
                                                                                                   dtype=int).reshape(1,
                                                                                                                      -1)
        else:
            start_idxs = np.random.choice(self.buffer_size, batch_size)
            idxs = start_idxs.reshape(-1, 1) * np.ones((batch_size, steps), dtype=int) + np.arange(steps,
                                                                                                   dtype=int).reshape(1,
                                                                                                                      -1)
            idxs %= self.buffer_size

        states = self.state_memory[idxs]
        next_states = self.next_state_memory[idxs]
        actions = self.action_memory[idxs]
        rewards = self.reward_memory[idxs]
        terminals = self.terminal_memory[idxs]
        trajectory_idx = self.trajectory_idx[idxs]

        mask = np.ones((batch_size, steps), dtype=bool)

        for i, diffs in enumerate(np.diff(trajectory_idx)):
            nonzeros = np.nonzero(diffs - 1)[0]
            if len(nonzeros) > 0:
                mask[i, nonzeros[0] + 1:] = False
            else:
                pass

        mask[:, :burn_in] = False

        return states, actions, rewards, next_states, terminals, trajectory_idx, mask

    def __len__(self):
        return min(self.buffer_size, self.buffer_counter)

    
