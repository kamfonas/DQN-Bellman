import numpy as np
import gym
from gym.utils import seeding
from gym import spaces
import random

## shares normalization factor
EPISODE_LENGTH = 2000
NBR_FEATURES = 5

class RandEnvR1(gym.Env):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,episode_length=EPISODE_LENGTH):

        self.episode_length = episode_length
        # action_space allows 3 actions in one dimension (we only trade one stock)
        self.action_space = spaces.Discrete(1) # Hold, Buy, Sell

        # Observation space  
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape = (NBR_FEATURES,))

        # reset data, counters and accumulators
        self.reset()

        self.episode = 0
            
    def step(self, action):

        # Set indicator "terminal" if this is the last day of the dataset.
        self.terminal = self.iter >= (self.episode_length)

        #print(actions)
        
        if self.terminal:  # If reached end of episode...                     
            self.episode += 1
           #return self.state, self.reward, self.terminal,{}

        else:
            
            self.iter += 1

            self.state = [random.random() for i in range(NBR_FEATURES)]  
            
            self.reward = np.mean(self.state) 

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.iter = 0
        self.state = [random.random() for i in range(NBR_FEATURES)] 
        self.reward = 0
        self.terminal = False 

        return self.state

    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]