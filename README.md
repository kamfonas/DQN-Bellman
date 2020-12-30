# Overview

This is an implementation of a trivial Deep Q-Learning model on a trivial environment intended to demonstrate how the Bellman equation is approximated by a DQN. The environment returns a state of five random floating point numbers from 0 to 1. There is only one action (action-0) and the reward is always the mean of the five values. Although this is definitely not a very interesting game, it allows us to watch how the target, the prediction and the expected Q-values compare step after step. 

The Bellman equation is a tail recursive function that calculates the Q-value as a function of the state s and the action taken a. The Q-value at step t is the sime of the reward  r plus the discounted highest Q-value of the next step: <img src="https://render.githubusercontent.com/render/math?math=Q(s_t,a_t) \= r_t %2B \gamma \max_{a_{t%2B1}}Q(s_{t%2B1},a_{t%2B1})">. The highest Q-value of the next step is calculated by recursively applying function Q on the next state and trying all possible actions to get the maximum Q-value. 

In our trivial environment there is only one action, and the expected value of the reward is always 0.5 since it is the mean of the five random features with values between 0 and 1. We can thus simplify the Bellman equation to: <img src="https://render.githubusercontent.com/render/math?math=Q_t \= r %2B \gamma Q_{t%2B1}">.  
By unraveling the recursion and viewing it as an iteration, we get the following sequence:


> <img src="https://render.githubusercontent.com/render/math?math=Q_0 \= r \gamma^0 ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_1 \= r (\gamma^0 %2B \gamma^1) ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_2 \= r (\gamma^0 %2B \gamma^1 %2B \gamma^2 ) ">

> <img src="https://render.githubusercontent.com/render/math?math=Q_k \= r \sum_{i=0}^{k}\gamma^i ">


