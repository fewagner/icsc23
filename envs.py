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
    
    
class OptimizationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    NMBR_STATES = 2
    NMBR_ACTIONS = NMBR_STATES

    def __init__(self, reset_params=False, continuous=False):
        super(OptimizationEnv, self).__init__()
        self.action_space = spaces.Box(low=- np.ones(self.NMBR_STATES),
                                       high=np.ones(self.NMBR_STATES),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=- np.ones(self.NMBR_STATES),
                                       high=np.ones(self.NMBR_STATES),
                                       dtype=np.float32)
        self.state = np.random.uniform(-1,1,size=self.NMBR_STATES)
        self.nmbr_minima = 1
        self.params = self.make_params()
        self.reset_params=reset_params
        self.continuous = continuous
        
    def make_params(self):
        return {'x0': np.random.uniform(-1,1,size=(self.nmbr_minima,2)), 
                'k': np.random.normal(loc=1, scale=.2,size=self.nmbr_minima), 
                'noise': np.random.uniform(0,.01)}
        
    def loss(self, x, y):
        loss = 0
        for x0,y0,k in zip(self.params['x0'][:,0], self.params['x0'][:,1], self.params['k']):
            loss += k * ((x - x0)**2 + (y - y0)**2)
            loss += np.random.normal(scale=self.params['noise'],size=loss.shape)
        return loss
        
    def step(self, action):
        
        info = {}
        
        new_state = action
        reward = - self.loss(new_state[0], new_state[1])
        terminated = True if np.linalg.norm(self.params['x0'] - new_state) < 0.1 and not self.continuous else False
        if terminated:
            reward += 1
        truncated = False
        
        self.state = new_state
        
        return new_state, reward, terminated, truncated, info
    
    def reset(self, state=None, new_params=None):
        info = {}
        if state is None:
            self.state = np.random.uniform(-1,1,size=self.NMBR_STATES)
        else:
            self.state = state
        if new_params or (new_params is None and self.reset_params):
            self.params = self.make_params()
        return self.state, info
        
    def render(self):
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)

        X, Y = np.meshgrid(x, y)
        Z = self.loss(X, Y)

        plt.contour(X, Y, Z, levels=10, colors='black')
        plt.scatter(self.state[0], self.state[1], color='red', s=100)
        plt.scatter(self.params['x0'][:,0], self.params['x0'][:,1], color='green', s=100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()