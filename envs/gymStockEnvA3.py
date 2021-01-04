import numpy as np
import pandas as pd
import gym
from gym.utils import seeding
from gym import spaces
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

## initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
## total number of assets in our portfolio
STOCK_DIM = 1
## total number of prices held on state
STOCK_PRICES_LEN = 21
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
EARLIEST_PRICE=1    # first of array of lagged stock prices
LAST_PRICE = STOCK_PRICES_LEN # Last known price (yesterday's adj close )
POSITION   = STOCK_PRICES_LEN+1 # Position Stock quantity
STATE_BUFFER_LEN = STOCK_PRICES_LEN+2

class StockEnvA1(gym.Env):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = pd.DataFrame(df, dtype=float)
        self.initPrice = df[0,LAST_PRICE-1]
        self.df_BuyHold = self.df.iloc[:,LAST_PRICE-1]/self.initPrice*\
             INITIAL_ACCOUNT_BALANCE*(1-TRANSACTION_FEE_PERCENT)
        # action_space allows 3 actions in one dimension (we only trade one stock)
        self.action_space = spaces.Discrete(3) # Hold, Buy, Sell
        # Observation space denotes the position 
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape = (STATE_BUFFER_LEN,))
        # load data as a numpy matrix
        self.data = list(self.df.iloc[self.day,0:(STOCK_PRICES_LEN)])
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
        self.trxLog=[]
        #self.reset()
        self._seed()
        self.episode = 0
    def baseline(self):
        return self.df_BuyHold

    def _eval_sell_stock(self):
        # perform buy action 
        cashInit = cashEnd = self.state[ACCT_BAL]
        price    = self.state[LAST_PRICE]
        nextPrice = self.data[STOCK_PRICES_LEN-1]
        posInit  = posEnd  = self.state[POSITION]
        qty   = 0
        trxAmt   = trxCost = 0.0
        trxType  = 'hold'
        valInit  = cashInit + (posInit * price)
        if  posInit > 0:
            qty      = - posInit
            posEnd   = posInit + qty
            trxAmt   = - price * qty
            trxCost  = trxAmt * TRANSACTION_FEE_PERCENT            
            cashEnd  = cashInit + trxAmt - trxCost  
            trxType  = 'sell'
        valEnd   = cashEnd  + (posEnd * nextPrice)
        valGain  = valEnd - valInit
        return (trxType,cashInit,cashEnd,posInit,posEnd,
                    qty,price,trxCost,trxAmt,valInit,valEnd,valGain)

    def _eval_buy_stock(self):
        # perform buy action 
        cashInit = cashEnd = self.state[ACCT_BAL]
        price    = self.state[LAST_PRICE]
        nextPrice = self.data[STOCK_PRICES_LEN-1]
        posInit  = posEnd  = self.state[POSITION]
        qty   = 0
        trxAmt   = trxCost = 0.0
        trxType  = 'hold'
        valInit  = cashInit + (posInit * price)
        valEnd   = cashEnd  + (posEnd * nextPrice)
        valGain  = 0.0
        if cashInit > INITIAL_ACCOUNT_BALANCE*TRANSACTION_FEE_PERCENT:
            qty      = cashInit*(1-TRANSACTION_FEE_PERCENT) // price
            posEnd   = posInit + qty
            trxAmt   = - price * qty
            trxCost  = - trxAmt * TRANSACTION_FEE_PERCENT            
            cashEnd  = cashInit + trxAmt - trxCost  
            trxType  = 'buy'
        valEnd   = cashEnd  + (posEnd * nextPrice)
        valGain  = valEnd - valInit
        return (trxType,cashInit,cashEnd,posInit,posEnd,
                    qty,price,trxCost,trxAmt,valInit,valEnd,valGain)

    def _eval_hold_stock(self):
        # perform buy action 
        cashInit = cashEnd = self.state[ACCT_BAL]
        price    = self.state[LAST_PRICE]
        nextPrice = self.data[STOCK_PRICES_LEN-1]
        posInit  = posEnd  = self.state[POSITION]
        qty   = 0
        trxAmt   = trxCost = 0.0
        trxType  = 'hold'
        valInit  = cashInit + (posInit * price)
        valEnd   = cashEnd  + (posEnd * nextPrice)
        valGain  = valEnd - valInit
        return (trxType,cashInit,cashEnd,posInit,posEnd,
                    qty,price,trxCost,trxAmt,valInit,valEnd,valGain)
            
        
    def step(self, actions):

        # Set indicator "terminal" if this is the last day of the dataset.
        self.terminal = self.day >= (self.df.shape[0]-1)

        #print(actions)
        
        if self.terminal:  # If reached end of episode...
            filePath = "episodes/trxLog_%05.0d.csv" % self.episode
            trxLog = pd.DataFrame(self.trxLog,
                         columns=['trxDay','trxType','cashInit','cashEnd',
                                  'posInit','posEnd','trxQty','trxPrice',
                                           'cost','trxAmt']
                         )
            trxLog.to_csv(filePath)
            #print(self.trxLog)
            L = len(self.asset_memory)
            if self.episode % 1 == 0: 
                title = "Episode {} vs Buy and Hold".format(self.episode)
                filePath = "episodes/account_value_%05.0d.png" % self.episode
                buyIx=trxLog.loc[trxLog['trxType']=='buy','trxDay'].tolist()
                sellIx=trxLog.loc[trxLog['trxType']=='sell','trxDay'].tolist()
                buyY  = [self.asset_memory[i] for i in buyIx ]
                sellY  = [self.asset_memory[i] for i in sellIx]
                plt.plot(self.asset_memory,'r',alpha = 0.6)
                plt.plot(self.df_BuyHold,'gray',alpha = 0.7)
                plt.plot(buyIx,buyY,'>',color = 'g',alpha = 0.5)
                plt.plot(sellIx,sellY,'<',color='b',alpha = 0.3)
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
            if self.episode % 10 == 0:
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
                     (self.state[POSITION] * self.state[LAST_PRICE])
            #
            # print("begin_total_asset:{}".format(begin_total_asset))

            # Get new prices 
            self.day += 1
            self.data = list(self.df.iloc[self.day,0:(STOCK_PRICES_LEN)])
            #self.data = self.df.loc[self.day,:]         
            
            # actions: 0-doNothing/Hold; 1- Buy; 2- Sell
            eval_actions = []
            for a in actions:
                if a == 2:
                    eval_actions.append(self._eval_sell_stock())
                elif a == 1:
                    eval_actions.append(self._eval_buy_stock())
                else:
                    eval_actions.append(self._eval_hold_stock())
            
            choice = np.argmax([x[11] for x in eval_actions])
            trxType,cashInit,cashEnd,posInit,posEnd,qty,price,trxCost,\
            trxAmt,valInit,valEnd,valGain = eval_actions[choice]
            #load next state
            #self.state[EARLIEST_PRICE:LAST_PRICE]= self.data
            self.state = [cashEnd]+\
                     self.data + [posEnd] 
                                  
            end_total_asset = self.state[ACCT_BAL]+ \
                (self.state[LAST_PRICE]*self.state[POSITION])
            
            #print("end_total_asset:{};  end_cash:{}".format(end_total_asset,self.state[ACCT_BAL]))
            #print("end_cash:{}".format(self.state[ACCT_BAL]))
            
            self.reward = end_total_asset - begin_total_asset            
            #print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)
            self.action_memory.append(actions[choice])

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = list(self.df.iloc[self.day,0:(STOCK_PRICES_LEN)])
        self.cost = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE]+\
                     self.data +\
                     [0]
        # iteration += 1 
        #self.episode = 0
        self.trxLog=[]
        # self.trxLog=pd.DataFrame([[self.day,'ini',
        #                            INITIAL_ACCOUNT_BALANCE,
        #                            INITIAL_ACCOUNT_BALANCE,
        #                            0,
        #                            0,
        #                            0,
        #                            self.initPrice,
        #                            0,
        #                            0
        #                            ]],
        #                         columns=['trxDay','trxType','cashInit','cashEnd',
        #                                   'posInit','posEnd',
        #                                   'trxQty','trxPrice',
        #                                   'cost','trxAmt']
        #                          )

        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]