# Overview

This is an implementation of a trivial Deep Q-Learning model on a trivial environment intended to demonstrate how the Bellman equation is approximated by a DQN. The environment returns a state of five random floating point numbers from 0 to 1. There is only one action (action-0) and the reward is always the mean of the five values. Although this is definitely not a very interesting game, it allows us to watch how the target, the prediction and the expected Q-values compare step after step. 

The Bellman equation is a tail recursive function that calculates the Q-value as a function of the state s and the action taken a. The Q-value at step t is the sime of the reward  r plus the discounted highest Q-value of the next step: <img src="https://render.githubusercontent.com/render/math?math=Q(s_t,a_t) \= r_t %2B \gamma \max_{a_{t%2B1}}Q(s_{t%2B1},a_{t%2B1})">. The highest Q-value of the next step is calculated recursively applying function Q on the next state and trying all possible actions to get the maximum Q-value. 

In our trivial environment with only one action and the expected value of the reward always 0.5, we can simplify the Bellman equation to: <img src="https://render.githubusercontent.com/render/math?math=Q_t \= r %2B \gamma Q_{t%2B1}">.  
By unraveling the recursion and viewing it as an iteration, we get:


> <img src="https://render.githubusercontent.com/render/math?math=Q_0 \= r \gamma^0 ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_1 \= r (\gamma^0 %2B \gamma^1) ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_2 \= r (\gamma^0 %2B \gamma^1 %2B \gamma^2 ) ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_k \= r \sum_{i=0}^{k}\gamma^i ">

# The Code

The main program is A1_R1_1. It reads its parameters as variables from a config.ini file which the user can edit for each run. Every run creates a uniquely named subdirectory under the results directory. A copy of the main file and the config file are copied in this directory to ensure that the actual parameters of the run are traceable. In addition, the run creates plots that show accumulated rewards, loss plots and graphs comparing the Q-value progression by step generated from the target DQN, the policy DQN and a theoretically calculated value. 

The configuration parameters of relevance are described below:

<dl>
  <dt>PROCESSING Section</dt>
  <dd><dl>
    <dt>USE_CUDA:</dt> <dd> Value 0 forces the use of CPU as Torch device. Otherwise CUDA will be used if it exists</dd> 
    <dt>NUMBER_OF_TRAIN_EPISODES:</dt> <dd> Takes an integer </dd> 
    <dt> MAX_STEPS:</dt> <dd>An integer indicating the maximum number of steps per episode. This is a limit value that gets used when the environment is registered and it has to be higher than the EPISODE_LENGTH.</dd>
<dt>EPISODE_LENGTH:</dt> <dd>An integer denoting the number of steps per episode. This value muist be less than MAX_STEPS. If the plots are empty this is most likely the problem.</dd>
</dl></dd>
  
<dt>EGREEDY Section</dt>
  <dd><dl>
EPSILON=0.0
EPSILON_FINAL=0.00
DECAY_STEPS=1
EGREEDY_TAIL = 5
    </dl></dd>


[ENVIRONMENT]
  <dd><dl>
ENV_NAME = RandEnvR1-v1
ENV_ENTRY_POINT = envs.gymRandMeanEnv1R2:RandEnvR1
    </dl></dd>

[EXPERIENCE_REPLAY]
  <dd><dl>
REPLAY_BUFFER_SIZE=5000
BATCH_SIZE = 1000
    </dl></dd>

[OPTIMIZATION]
  <dd><dl>
DISCOUNT_FACTOR = 0.8 
UPDATE_TARGET_FREQUENCY = 1
    </dl></dd>

[DQN]
  <dd><dl>
DQN_IMPORT_LIBRARY=dqn_linear3H
HIDDEN_LAYERS_SIZE=5,5,5
DROPOUT = 0.2
OFFSET = 1
    </dl></dd>
</dl>