import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
## shares normalization factor
#HMAX_NORMALIZE = 30
## initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000.0
## total number of stocks in our portfolio
STOCK_DIM = 1
## total number of prices held on state
#STOCK_PRICES_LEN = 20
## transaction fee: 2/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.002

## turbulence index: 120 reasonable threshold
TURBULENCE_THRESHOLD = 120
## Normalization factor: (not used)
##MAX_ACCOUNT_BALANCE = 2147483647
##MAX_NUM_SHARES = 2147483647
##MAX_CLOSE_PRICE = 5000
### State Positions:
ACCT_BAL   = 0      # first position = account balance
HIGH       = 1
LOW        = 2
OPEN	       = 3
CLOSE      = 4
VOLUME     = 5	
ADJCLOSE	   = 6
RSI	       = 7
MACD       = 8
MACDS      = 9
MACDH      = 10
CCI	       = 11
PDI	       = 12
MDI	       = 13
DX	       = 14
ADX        = 15
POSITION   = 16 # Position Stock quantity
STARTDATA  = 1
ENDDATA    = 16
STATE_BUFFER_LEN = 17
DATA_BUFFER_LEN = 15

class StockEnvB1(gym.Env):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.initDay = day
        self.df = pd.DataFrame(df, dtype=float)
        self.initPrice = df[0,ADJCLOSE-1]
        self.df_BuyHold = self.df.iloc[:,ADJCLOSE-1]*1000000/self.initPrice

        # action_space allows 3 actions in one dimension (we only trade one stock)
        self.action_space = spaces.Discrete(3) # Hold, Buy, Sell
        # Observation space denotes the position 
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape = (STATE_BUFFER_LEN,))

        # reset data, counters and accumulators
        self.reset()

        self._seed()
        self.episode = 0
    def baseline(self):
        return self.df_BuyHold
            
    def _sell_stock(self):
        # perform sell action based on the sign of the action
        if self.state[POSITION] > 0:
            #accountBalance=adjClosingPrice*positionShares(1-transactionFee)
            self.state[ACCT_BAL] += \
                self.state[ADJCLOSE]*\
                self.state[POSITION] * (1- TRANSACTION_FEE_PERCENT)
                
            self.state[POSITION] -= self.state[POSITION]
            self.cost += self.state[ADJCLOSE]*self.state[POSITION]*TRANSACTION_FEE_PERCENT
        else:
            pass
    
    def _buy_stock(self):
        # perform buy action 
        if self.state[ACCT_BAL] > 0:
            buy_qty = self.state[ACCT_BAL] // self.state[ADJCLOSE] 
            # print('buy quantity:{}'.format(buy_qty))
            
            #update balance
            self.state[ACCT_BAL] -= \
                self.state[ADJCLOSE]  * \
                buy_qty * (1+ TRANSACTION_FEE_PERCENT)

            self.state[POSITION] += buy_qty
            
            self.cost += self.state[ADJCLOSE] * buy_qty * TRANSACTION_FEE_PERCENT
        
    def step(self, actions):

        # Set indicator "terminal" if this is the last day of the dataset.
        self.terminal = self.day >= (self.df.shape[0]-1)

        #print(actions)
        
        if self.terminal:  # If reached end of episode...
            L = len(self.asset_memory)
            if self.episode % 10 == 9: 
                title = "Episode {} vs Buy and Hold".format(self.episode)
                filePath = "episodes/account_value_%05.0d.png" % self.episode
                buyIx = np.array([i for i, x in enumerate(self.action_memory) if x==1 if i < L ])
                buyIx = buyIx[buyIx - shift(buyIx, 1, cval=999999) > 1].tolist()
                sellIx = np.array([i for i, x in enumerate(self.action_memory) if x==2 if i < L])
                sellIx = sellIx[sellIx - shift(sellIx, 1, cval=999999) > 1].tolist()
                #print(buyIx)
                #print(sellIx)
                buyY  = [self.asset_memory[i] for i in buyIx ]
                sellY  = [self.asset_memory[i] for i in sellIx]
                plt.plot(self.asset_memory,'r',alpha = 0.7)
                plt.plot(self.df_BuyHold,'gray',alpha = 0.7)
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
            if self.episode % 10 == 9:
                filePath = "episodes/account_rewards_%05.0d.csv" % self.episode
                df_rewards.to_csv(filePath)
                filePath = "episodes/account_value%05.0d.csv" % self.episode
                df_total_value.to_csv(filePath)
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
                     (self.state[POSITION] * self.state[ADJCLOSE])
            #
            # print("begin_total_asset:{}".format(begin_total_asset))
            
            # actions: 0-doNothing/Hold; 1- Buy; 2- Sell
            if actions == 2:
                self._sell_stock()
            if actions == 1:
                self._buy_stock()
            self.day += 1
            self.data = list(self.df.iloc[self.day,0:DATA_BUFFER_LEN])
            #self.data = self.df.loc[self.day,:]         

            #load next state
            #self.state[EARLIEST_PRICE:LAST_PRICE]= self.data
            self.state = [self.state[ACCT_BAL]]+\
                     self.data + [self.state[POSITION]] 
                                  
            end_total_asset = self.state[ACCT_BAL]+ \
                (self.state[ADJCLOSE]*self.state[POSITION])
            
            #print("end_total_asset:{};  end_cash:{}".format(end_total_asset,self.state[ACCT_BAL]))
            #print("end_cash:{}".format(self.state[ACCT_BAL]))
            
            self.reward = end_total_asset - begin_total_asset            
            #print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)
            self.action_memory.append(actions)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.day = self.initDay
        self.data = list(self.df.iloc[self.day,0:DATA_BUFFER_LEN])
        self.state = [INITIAL_ACCOUNT_BALANCE]+\
                     self.data +\
                     [0]
        self.cost = 0
        self.reward = 0
        self.terminal = False 
        self.rewards_memory = []
        self.action_memory = []
        self.asset_memory = [self.state[ACCT_BAL]]
        return self.state

    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]