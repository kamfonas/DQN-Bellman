import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

## shares normalization factor
HMAX_NORMALIZE = 30

## initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=10000

## total number of stocks in our portfolio
STOCK_DIM = 1

## total number of prices held on state
STOCK_PRICES_LEN = 20

## transaction fee: 2/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.002

## turbulence index: 120 reasonable threshold
TURBULENCE_THRESHOLD = 120

### State Positions:
ACCT_BAL   = 0      # first position = account balance
EARLIEST_PRICE=1    # first of array of lagged stock prices
LAST_PRICE = STOCK_PRICES_LEN+1 # Last known price (yesterday's adj close )
POSITION   = STOCK_PRICES_LEN+2 # Position Stock quantity
STATE_BUFFER_LEN = STOCK_PRICES_LEN+3
MAX_LONG=100
MAX_SHORT=-100
SCALING_FACTOR=1

class StockEnvD1(gym.Env):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):

        self.day = day
        self.df = pd.DataFrame(df, dtype=float)
        self.initPrice = df[0,(LAST_PRICE-1)]
        self.df_BuyHold = self.df.iloc[:,(LAST_PRICE-1)]*INITIAL_ACCOUNT_BALANCE/self.initPrice

        # action_space allows 3 actions in one dimension (we only trade one stock)
        self.action_space = spaces.Discrete(101) # Hold, Buy, Sell

        # Observation space denotes the position 
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape = (STATE_BUFFER_LEN,))
        # load data as a numpy matrix
        self.data = list(self.df.iloc[self.day,0:(STOCK_PRICES_LEN+1)])
        # print(self.data)
        self.terminal = False             
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE]+\
                     self.data+\
                     [0]
                     
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = self.state[0]
        self.rewards_memory = []
        self.action_memory =[]
        self.position_memory = []
        #self.reset()
        self._seed()
        self.episode = 0
    def baseline(self):
        return self.df_BuyHold
            
    def _sell_stock(self, action):

        mapped_action = (action - 50) * SCALING_FACTOR

        # perform sell action based on the sign of the action
        if self.state[POSITION] > MAX_SHORT:
            #accountBalance=adjClosingPrice*positionShares(1-transactionFee)
            self.state[ACCT_BAL] += \
                self.state[LAST_PRICE]*\
                (-1) * mapped_action * (1- TRANSACTION_FEE_PERCENT)
                
            self.state[POSITION] += mapped_action
            self.cost += self.state[LAST_PRICE] * mapped_action * TRANSACTION_FEE_PERCENT
    
    def _buy_stock(self, action):

        mapped_action = (action - 50) * SCALING_FACTOR

        # perform buy action 
        if (self.state[POSITION] < MAX_LONG) and (self.state[ACCT_BAL] > self.state[LAST_PRICE] * mapped_action):
            #buy_qty = self.state[ACCT_BAL] // self.state[LAST_PRICE]
            # print('buy quantity:{}'.format(buy_qty))
            
            #update balance
            self.state[ACCT_BAL] -= \
                self.state[LAST_PRICE]  * \
                mapped_action * (1+ TRANSACTION_FEE_PERCENT)

            self.state[POSITION] += mapped_action
            self.cost += self.state[LAST_PRICE] * mapped_action * TRANSACTION_FEE_PERCENT
        
    def step(self, actions):

        # Set indicator "terminal" if this is the last day of the dataset.
        self.terminal = self.day >= (self.df.shape[0]-1)

        #print(actions)
        
        if self.terminal:  # If reached end of episode...
            L = len(self.asset_memory)
            if self.episode % 10 == 0: 
                title = "Episode {} vs Buy and Hold".format(self.episode)
                filePath = "episodes/account_value_%05.0d.png" % self.episode
                buyIx = np.array([i for i, x in enumerate(self.action_memory) if x==1 if i < L ])
                buyIx = buyIx[buyIx - shift(buyIx, 1, cval=999999) > 1].tolist()
                sellIx = np.array([i for i, x in enumerate(self.action_memory) if x==2 if i < L])
                sellIx = sellIx[sellIx - shift(sellIx, 1, cval=999999) > 1].tolist()
                #print(buyIx)
                #print(sellIx)
                buyY  = [self.asset_memory[i]/self.asset_memory[0] for i in buyIx]
                sellY  = [self.asset_memory[i]/self.df_BuyHold[0] for i in sellIx]
                plt.plot(pd.Series(self.asset_memory)/self.asset_memory[0],'r',alpha = 0.7)
                plt.plot(self.df_BuyHold/self.df_BuyHold[0],'gray',alpha = 0.7)
                plt.plot(buyIx,buyY,'>',color = 'g',alpha = 0.7)
                plt.plot(sellIx,sellY,'<',color='b',alpha = 0.7)
                plt.title(title)
                #plt.legend()
                #
                plt.savefig(filePath)
                plt.show()
                plt.close()
            
            # Save total value to csv file
            df_total_value = pd.DataFrame(self.asset_memory)
            
            # add column
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            with np.errstate(divide='ignore'):
                try:
                    sharpe = (252**0.5)*df_total_value['daily_return'].mean()/df_total_value['daily_return'].std()
                except:
                    sharpe = 0.0
            #print("Sharpe: ",sharpe)
            df_total_value.to_csv('account_value.csv')
     
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('account_rewards.csv')
            df_actions = pd.DataFrame(self.action_memory)
            df_positions = pd.DataFrame(self.position_memory)
            if self.episode % 10 == 0:
                filePath = "episodes/account_rewards_%05.0d.csv" % self.episode
                df_rewards.to_csv(filePath)
                filePath = "episodes/account_value%05.0d.csv" % self.episode
                df_total_value.to_csv(filePath)
                filePath = "episodes/actions_%05.0d.csv" % self.episode
                df_actions.to_csv(filePath)
                filePath = "episodes/positions_%05.0d.csv" % self.episode
                df_positions.to_csv(filePath)
                with open("episodes/buyIx%05.0d.txt" % self.episode,'w') as F:
                   F.write(str(buyIx))
                with open("episodes/sellIx%05.0d.txt" % self.episode,'w') as F:
                   F.write(str(sellIx))
                
            self.episode += 1
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
        
            return self.state, self.reward, self.terminal,{'sharpe':sharpe}

        else:
            # print(np.array(self.state[1:29]))
            # action = (action.astype(int))
            begin_total_asset = self.state[ACCT_BAL]+ \
                     (self.state[POSITION] * self.state[LAST_PRICE])
            #
            # print("begin_total_asset:{}".format(begin_total_asset))
            
            # actions: 0-doNothing/Hold; 1- Buy; 2- Sell
            if 0 <= actions <= 49:
                self._sell_stock(actions)
            if 51 <= actions <= 100:
                self._buy_stock(actions)
            self.day += 1
            self.data = list(self.df.iloc[self.day,0:(STOCK_PRICES_LEN+1)])
            #self.data = self.df.loc[self.day,:]         

            #load next state
            #self.state[EARLIEST_PRICE:LAST_PRICE]= self.data
            self.state = [self.state[ACCT_BAL]]+\
                     self.data + [self.state[POSITION]] 
                                  
            end_total_asset = self.state[ACCT_BAL]+ \
                (self.state[LAST_PRICE]*self.state[POSITION])
            
            #print("end_total_asset:{};  end_cash:{}".format(end_total_asset,self.state[ACCT_BAL]))
            #print("end_cash:{}".format(self.state[ACCT_BAL]))
            
            self.reward = (end_total_asset - begin_total_asset) / begin_total_asset            
            #print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)
            self.action_memory.append((actions - 50) * SCALING_FACTOR)
            self.position_memory.append(self.state[POSITION])

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = list(self.df.iloc[self.day,0:(STOCK_PRICES_LEN+1)])
        self.cost = 0
        self.terminal = False 
        self.rewards_memory = []
        self.action_memory = []
        self.position_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE]+\
                     self.data +\
                     [0]
        # iteration += 1 
        #self.episode = 0
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]