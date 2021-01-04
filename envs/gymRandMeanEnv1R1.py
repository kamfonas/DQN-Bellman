import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib.pyplot as plt
import random

## shares normalization factor
TRANSACTION_FEE_PERCENT = 0.001
EPISODE_LENGTH = 2000
NBR_FEATURES = 5
INVESTED = 0
SEED = None

class RandEnvR1(gym.Env):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,episode_length=EPISODE_LENGTH,runPath='episodes',runType='train'):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.episode_length = episode_length
        self.runPath = runPath
        self.runType = runType
        self.iter = 0
        # action_space allows 3 actions in one dimension (we only trade one stock)
        self.action_space = spaces.Discrete(3) # Hold, Buy, Sell

        # Observation space  
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape = (NBR_FEATURES+1,))

        # reset data, counters and accumulators
        self.reset()

        self._seed(SEED)
        print("This run Seed =",self.seed)
        self.episode = 0
            
        
    def step(self, action):
        runPath = self.runPath
        # Set indicator "terminal" if this is the last day of the dataset.
        self.terminal = self.iter >= (self.episode_length)

        #print(actions)
        
        if self.terminal:  # If reached end of episode...
            if self.episode % 10 == 0: 
                filePath = runPath+"/%s_value_%05.0d.png" % (self.runType,self.episode)
                title = "Episode {} ({})".\
                    format(self.episode,
                           self.runType)
                plt.plot(self.asset_memory,'r',alpha = 0.7)
                plt.title(title)
                #plt.legend()
                #
                plt.savefig(filePath)
                plt.show()
                plt.close()
            
            # Save total value to csv file
            df_total_value = pd.DataFrame(self.asset_memory)
            
            # add column
            df_total_value.columns = ['total_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            #print("Sharpe: ",sharpe)
            df_total_value.to_csv(self.runPath+"/%s_account_value.csv" % self.runType)
     
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(self.runPath+"/%s_account_rewards.csv")
            if self.episode % 10 == 9:
                filePath = self.runPath+"/%s_account_rewards_%05.0d.csv" % (self.runType,self.episode)
                df_rewards.to_csv(filePath)
                filePath = self.runPath+"/%s_account_value%05.0d.csv" % (self.runType,self.episode)
                df_total_value.to_csv(filePath)
                
            self.episode += 1
           #return self.state, self.reward, self.terminal,{}

        else:
            
            # actions: 0-doNothing/Hold; 1- Buy; 2- Sell
            if action == 2: # sell
                self.state[INVESTED] = 0
            if action == 1: # buy
                self.state[INVESTED] = 1
            self.iter += 1
            self.data = [random.random() for i in range(NBR_FEATURES)]

            #self.data = self.df.loc[self.day,:]         

            self.state = [self.state[INVESTED]]+\
                     self.data  
                                  
            
            #print("end_total_asset:{};  end_cash:{}".format(end_total_asset,self.state[ACCT_BAL]))
            #print("end_cash:{}".format(self.state[ACCT_BAL]))
            
            self.reward = self.state[INVESTED]*np.mean(self.state[1:])            
            #print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(self.reward)
            self.action_memory.append(action)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.iter = 0
#Generate 5 random numbers between 10 and 30

        self.data = [random.random() for i in range(NBR_FEATURES)]
        self.state = [0] + self.data 
        self.reward = 0
        self.terminal = False 
        self.rewards_memory = []
        self.action_memory = []
        self.asset_memory = [self.state[INVESTED]]
        self.trxLog=[]

        return self.state

    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]