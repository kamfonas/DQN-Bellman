# Overview

This is an implementation of a simple Deep Q-Learning model with dual DQNs on a trivial environment intended to demonstrate how the Bellman equation is approximated by a DQN. The environment returns a state of five random floating point numbers from 0 to 1. There is only one action (action-0) and the reward is always the mean of the five values.  Although this is definitely not a very interesting game, the regularity of the rewards and the deteministic nature of the single action makes the workings of the DQN more ovious as we watch how the target, the prediction and the expected theoretically derived Q-values compare over steps and episodes of training.

The Bellman equation is a tail recursive function that calculates the Q-value as a function of the state s and the action taken a. The Q-value at step t is the sum of that step's  reward plus the discounted highest Q-value of the next step: <img src="https://render.githubusercontent.com/render/math?math=Q(s_t,a_t) \= r_t %2B \gamma \max_{a_{t%2B1}}Q(s_{t%2B1},a_{t%2B1})">. The highest Q-value of the next step is calculated recursively applying function Q on the next state and trying all possible actions to get the maximum Q-value.

In our trivial environment with only one action and the expected value of the reward consistently at 0.5, we can simplify the Bellman equation to: <img src="https://render.githubusercontent.com/render/math?math=Q_t \= r %2B \gamma Q_{t%2B1}">.  By unraveling the recursion and viewing it as an iteration, we get:

> <img src="https://render.githubusercontent.com/render/math?math=Q_0 \= r \gamma^0 ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_1 \= r (\gamma^0 %2B \gamma^1) ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_2 \= r (\gamma^0 %2B \gamma^1 %2B \gamma^2 ) ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_k \= r \sum_{i=0}^{k}\gamma^i ">

# The Code

## General Architecture
The main program is A1_R1_1. It reads its parameters as variables from a config.ini file which the user can edit for each run. At every execution the system creates a uniquely named subdirectory under the "results" directory. A copy of the main file and the config file are copied in this directory to ensure that the actual parameters of the run are traceable. In addition, the run creates plots that show accumulated rewards, loss plots and graphs comparing the Q-value progression generated three ways: (1) from the target DQN, (2) the policy DQN and (3) a theoretically calculated value. 

The main program contains the class definitions for the experience replay buffer and the DQN agent. The plot code is in-line with the procedures, something that grew organically. Some of these components would probably be cleaner if separated to their own module, however it should serve the purpose for this demonstration.  

The environment is provided to the main program via a configuration parameter, and it is unregistered and reregistered with every execution. This can be adjusted by commenting or changing main program lines 275-285. The DQN is maintained in a separate module and it is also loaded dynamically from its path stored in a configuration parameter. Three such files of are provided with one, two and three dense hidden layers with parametrized drop layers in between. This makes changing the DQN to see its impact simple and makes it easy to add different types of DQNs. Utility functions are kept in a separate module.

## Environment Modules

Subdirectory "envs" contains all environment modules, compliant with the "gym" libraries. Environment gymRandMeanEnv1R2.py is the one tested for the A1_R1_1 main program.

## The Configuration Mechanism

We make simple use of the configparser library with the configuration parameters of relevance described below. The name of the configuration file is config_XXX.ini where XXX is the stem of the name of the meain program. This way, if variants of this program get created, they each will maintain their own configuration file.

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
<dt>EPSILON</dt> <dd>Probability threshold above which exploration is forced in selecting alpha. E.g.  0.0</dd>
<dt>EPSILON_FINAL</dt> <dd> The final value of EPSILON used during exploitation. E.g. 0.00</dd>
<dt>DECAY_STEPS</dt> <dd> Number of steps over which the decay occurs. </dd>
<dt>EGREEDY_TAIL</dt> <dd>The number of training episodes after the decay and into exploitation mode. It is used to report episode loss after the exploration is completed</dd>
    </dl></dd>


<dt>ENVIRONMENT</dt>
  <dd><dl>
<dt>ENV_NAME</dt> <dd>The name in the form RandEnvR1-v1 used by gym to register environments</dd>
<dt>ENV_ENTRY_POINT</dt><dd>The gym-required environment entry point used in the registration to find the actual source file and class, e.g. envs.gymRandMeanEnv1R2:RandEnvR1</dd>
    </dl></dd>

<dt>EXPERIENCE_REPLAY</dt>
  <dd><dl>
<dt>REPLAY_BUFFER_SIZE</dt> <dd>An integer denoting the length of the experience replay buffer.</dd>
<dt>BATCH_SIZE</dt> <dd>The number of experiences in a minibatch sample used for training the DQN</dd>
    </dl></dd>

<dt>DQN</dt>
  <dd><dl>
<dt>DQN_IMPORT_LIBRARY</dt> <dd>The name of the DQN file with the definition of the DQN to be used. e.g. dqn_linear3H</dd>
<dt>HIDDEN_LAYERS_SIZE</dt> <dd>Comma separated integers denoting the number of units in each hidden layer. e.g. 5,5,5</dd>
<dt>DROPOUT</dt> <dd> A value between 0 and 1 that denotes the ratio of dropout units, e.g. 0.2 </dd>
<dt>OFFSET</dt> <dd>By default 1. This is an offset used in plotting the theoretical series of Q-values generated from the value of gamma. </dd>
    </dl></dd>
</dl>
