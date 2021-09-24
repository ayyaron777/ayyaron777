#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time # to get the time
import math # needed for calculations
env = gym.make("CartPole-v1")
observation = env.reset()

train_episodes = 2000         # Total train episodes
test_episodes = 100           # Total test episodes
max_steps = 2000


# In[2]:


training_rewards = []   # list of rewards
data = []
actions = []

# EXPLORATION / EXPLOITATION PARAMETERS
epsilon = 1                   # Exploration rate
max_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob


epochs = 0
penalties, reward = 0, 0

for episode in range(train_episodes):
    state = env.reset()    # Reset the environment
    cumulative_training_rewards = 0
    
    for step in range(max_steps):
        #env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        cumulative_training_rewards += reward

        epochs += 1
        
        # If we reach the end of the episode
        if done == True:
            #print ("Cumulative reward for episode {}: {}".format(episode, cumulative_training_rewards))
            data.append(cumulative_training_rewards)
            actions.append(episode)
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    # append the episode cumulative reward to the list
    training_rewards.append(cumulative_training_rewards)

print ("Training score over time: " + str(sum(training_rewards)/train_episodes))
env.close()

fig, ax = plt.subplots(figsize=(10, 4), dpi=72)
ax.set(xlabel='Episodes', ylabel='Rewards',
       title='Training reward over time')
ax.grid()
ax.plot(actions, data)


# In[3]:


n_actions = env.action_space.n
n_states = env.observation_space.shape[0]


# In[4]:


# define the number of buckets for each state value (x, x', theta, theta')
buckets = (1, 1, 6, 12)     

# define upper and lower bounds for each state value
upper_bounds = [
        env.observation_space.high[0], 
        0.5, 
        env.observation_space.high[2], 
        math.radians(50)
        ]
lower_bounds = [
        env.observation_space.low[0], 
        -0.5, 
        env.observation_space.low[2], 
        -math.radians(50)]


# In[7]:


# HYPERPARAMETERS
n_episodes = 2000           # Total train episodes
n_steps = 200               # Max steps per episode
min_alpha = 0.1             # learning rate
min_epsilon = 0.1           # exploration rate
gamma = 1                   # discount factor
ada_divisor = 25            # decay rate parameter for alpha and epsilon

# INITIALISE Q MATRIX
Q = np.zeros(buckets + (n_actions,)) 
print(np.shape(Q))

def discretize(obs):
    ''' discretise the continuous state into buckets ''' 
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def epsilon_policy(state, epsilon):
    ''' choose an action using the epsilon policy '''
    exploration_exploitation_tradeoff = np.random.random()
    if exploration_exploitation_tradeoff <= epsilon:
        action = env.action_space.sample()  # exploration
    else:
        action = np.argmax(Q[state])   # exploitation
    return action

def greedy_policy(state):
    ''' choose an action using the greedy policy '''
    return np.argmax(Q[state])

def update_q(current_state, action, reward, new_state, alpha):
    ''' update the Q matrix with the Bellman equation '''
    Q[current_state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[current_state][action])

def get_epsilon(t):
    ''' decrease the exploration rate at each episode '''
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))

def get_alpha(t):
    ''' decrease the learning rate at each episode '''
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))

# TRAINING PHASE
rewards = [] 

for episode in range(n_episodes):
    current_state = env.reset()
    current_state = discretize(current_state)

    alpha = get_alpha(episode)
    epsilon = get_epsilon(episode)

    episode_rewards = 0

    for t in range(n_steps):
        # env.render()
        action = epsilon_policy(current_state, epsilon)
        new_state, reward, done, _ = env.step(action)
        new_state = discretize(new_state)
        update_q(current_state, action, reward, new_state, alpha)
        current_state = new_state

        # increment the cumulative reward
        episode_rewards += reward

        # at the end of the episode
        if done:
            #print('Episode:{}/{} finished with a total reward of: {}'.format(episode, n_episodes, episode_rewards))
            break

    # append the episode cumulative reward to the reward list
    rewards.append(episode_rewards)


# PLOT RESULTS
x = range(n_episodes)

fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
ax.set(xlabel='Episodes', ylabel='Rewards',
       title='Training reward over time')
ax.grid()
ax.plot(x, rewards)


# In[ ]:




