#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import gym
import random


# In[8]:


# CREATE THE ENVIRONMENT
env = gym.make("FrozenLake-v1")
action_size = env.action_space.n
state_size = env.observation_space.n
print("Action space size: ", action_size)
print("State space size: ", state_size)


# In[9]:


# INITIALISE Q TABLE TO ZERO
Q = np.zeros((state_size, action_size))


# In[10]:


# HYPERPARAMETERS
train_episodes = 2000         # Total train episodes
test_episodes = 100           # Total test episodes
max_steps = 2000               # Max steps per episode
alpha = 0.7                   # Learning rate
gamma = 0.618                 # Discounting rate

# EXPLORATION / EXPLOITATION PARAMETERS
epsilon = 1                   # Exploration rate
max_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob


# In[11]:


# TRAINING PHASE
training_rewards = []   # list of rewards

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation
data = []
actions = []

done = False

for episode in range(train_episodes):
    state = env.reset()    # Reset the environment
    cumulative_training_rewards = 0
    
    for step in range(max_steps):
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

#plt.plot(data)
fig, ax = plt.subplots(figsize=(10, 4), dpi=72)
ax.set(xlabel='Episodes', ylabel='Rewards',
       title='Training reward over time')
ax.grid()
ax.plot(actions, data)


# In[ ]:





# In[12]:


# TRAINING PHASE
training_rewards = []   # list of rewards
data = []
actions = []
for episode in range(train_episodes):
    state = env.reset()    # Reset the environment
    cumulative_training_rewards = 0
    
    for step in range(max_steps):
        # Choose an action (a) among the possible states (s)
        exp_exp_tradeoff = random.uniform(0, 1)   # choose a random number
        
        # If this number > epsilon, select the action corresponding to the biggest Q value for this state (Exploitation)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state,:])        
        # Else choose a random action (Exploration)
        else:
            action = env.action_space.sample()
        
        # Perform the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update the Q table using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action]) 
        cumulative_training_rewards += reward  # increment the cumulative reward        
        state = new_state         # Update the state
        
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

fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
ax.set(xlabel='Episodes', ylabel='Rewards',
       title='Training reward over time')
ax.grid()
ax.plot(actions, data)


# In[13]:


def action_epsilon_greedy(q, s, epsilon=0.05):
    if np.random.rand() > epsilon:
        return np.argmax(q[s])
    return np.random.randint(4)

def get_action_epsilon_greedy(epsilon):
    return lambda q,s: action_epsilon_greedy(q, s, epsilon=epsilon)


# In[14]:


training_rewards = []   # list of rewards
data = []
actions = []
for episode in range(train_episodes):
    state = env.reset()    # Reset the environment
    cumulative_training_rewards = 0
    
    for step in range(max_steps):
        # Choose an action (a) among the possible states (s)
        exp_exp_tradeoff = random.uniform(0, 1)   # choose a random number
        
        # If this number > epsilon, select the action corresponding to the biggest Q value for this state (Exploitation)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state,:])        
        # Else choose a random action (Exploration)
        else:
            action = env.action_space.sample()
        
        q = np.ones((16,4))
        # Perform the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        
        new_action = action_epsilon_greedy(Q, new_state, epsilon=epsilon)

        # Update the Q table using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, new_action]) - Q[state, action]) 

        cumulative_training_rewards += reward  # increment the cumulative reward        
        state = new_state         # Update the state
        
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

fig, ax = plt.subplots(figsize=(10, 4), dpi=72)
ax.set(xlabel='Episodes', ylabel='Rewards',
       title='Training reward over time')
ax.grid()
ax.plot(actions, data)


# In[ ]:




