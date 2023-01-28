import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class MDPEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    NMBR_STATES = 3
    NMBR_ACTIONS = 2

    def __init__(self):
        super(MDPEnv, self).__init__()
        self.action_space = spaces.Discrete(self.NMBR_ACTIONS) 
        self.observation_space = spaces.Discrete(self.NMBR_STATES) 
        
        self.state_transitions = np.array([[ .5, 0., .7, 0.0, .4, .3],
                                           [ 0., 0., .1, .95, 0., .3],
                                           [ .5, 1., .2, .05, .6, .4]])
        self.state = np.random.randint(3)
        
    def dynamics_function(self, state, action):
        state_one_hot = np.eye(self.NMBR_STATES*self.NMBR_ACTIONS,1,-state*(self.NMBR_STATES-1)-action*(self.NMBR_ACTIONS-1))
        probs = np.dot(self.state_transitions,state_one_hot)
        new_state = np.random.choice(np.arange(3), p=probs.flatten())
        return new_state
    
    def reward_function(self, state, action, next_state):
        reward = 0
        if (state == 1) and (action == 0) and (next_state == 0):
            reward = 5
        elif (state == 2) and (action == 1) and (next_state == 0):
            reward = -1
        return reward
        
    def step(self, action):
        
        info = {}
        terminated = False
        truncated = False
        
        new_state = self.dynamics_function(self.state, action)
        reward = self.reward_function(self.state, action, new_state)
        
        self.state = new_state
        
        return new_state, reward, terminated, truncated, info
    
    def reset(self, state=None):
        info = {}
        if state is None:
            self.state = np.random.randint(3)
        else:
            self.state = state
        
        return self.state, info
        
    def render(self):
        print('state: {}'.format(self.state))


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
    

class TupleSpaceWrapper:
    
    def __init__(self, tuple_space):
        self.tuple_space = tuple_space
        self.all_n = []
        for space in tuple_space:
            self.all_n.append(space.n)
        self.n = np.prod(self.all_n)
            
    def encode(self, tuple_state):
        return int(np.sum([tuple_state[i]*np.prod(self.all_n[i+1:]) for i in range(len(self.all_n))]))
    
    def decode(self, state):

        tuple_state = np.zeros(len(self.all_n))

        for i, n in zip(reversed(np.arange(len(self.all_n))), reversed(self.all_n)):
            residual = state % n
            state = int(state / n)
            tuple_state[i] = residual

        return tuple_state
            
    def sample(self):
        return np.random.randint(self.n)
    
class TupleEnvWrapper(gym.Env):

    def __init__(self, env):
        super(TupleEnvWrapper, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = TupleSpaceWrapper(env.observation_space)
        
    def step(self, action):
        
        new_state, reward, terminated, truncated, info = self.env.step(action)
        new_state = self.observation_space.encode(new_state)
        
        return new_state, reward, terminated, truncated, info
    
    def reset(self):
        
        state, info = self.env.reset()
        state = self.observation_space.encode(state)
        
        return state, info
        
    def render(self):
        return self.env.render()
    
    
