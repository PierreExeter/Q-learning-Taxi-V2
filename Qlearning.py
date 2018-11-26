import numpy as np
import matplotlib.pyplot as plt
import gym
import random

# create the environment
env = gym.make("Taxi-v2")
#env = gym.make("FrozenLake-v0")
#env = gym.make("FrozenLake8x8-v0")
env.render()

action_size = env.action_space.n
print("Action space size: ", action_size)

state_size = env.observation_space.n
print("State space size: ", state_size)

# initialise the Q table
qtable = np.zeros((state_size, action_size))
print(qtable)

# hyperparameters for the taxi
train_episodes = 50000        # Total episodes
test_episodes = 100           # Total test episodes
max_steps = 99                # Max steps per episode
learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# hyperparameters for the frozen lake
#train_episodes = 15000        # Total episodes
#test_episodes = 100           # Total test episodes
#max_steps = 99                # Max steps per episode
#learning_rate = 0.8           # Learning rate
#gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1                   # Exploration rate
max_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# implement the Q learning algorithm: training phase
training_rewards = []   # list of rewards

for episode in range(train_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    cumulative_training_rewards = 0
    
    for step in range(max_steps):
        # Choose an action a in the current world state (s)
        exp_exp_tradeoff = random.uniform(0,1)   # choose a random number
        
        # If this number > epsilon --> exploitation (Select the action corresponding to the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])        
        # Else choose a random action --> exploration
        else:
            action = env.action_space.sample()
        
        # Perform the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update the Q table for that state and that action using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
                                    np.max(qtable[new_state, :]) - qtable[state, action]) 
        cumulative_training_rewards += reward  # increment the cumulative reward        
        state = new_state # Update the state
        
        # If we reach the end of the episode
        if done == True:
            print ("Cumulative reward for episode {}: {}".format(episode, cumulative_training_rewards))
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    # append the episode cumulative reward to the list
    training_rewards.append(cumulative_training_rewards)

print ("Training score over time: " + str(sum(training_rewards)/train_episodes))
print(qtable)

x = range(train_episodes)
plt.plot(x, training_rewards)
plt.xlabel('episode')
plt.ylabel('Training cumulative reward')
plt.show()
    
# Test phase
env.reset()
test_rewards = []

for episode in range(test_episodes):
    state = env.reset()
    step = 0
    done = False
    cumulative_test_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
#        env.render() # COMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])        
        new_state, reward, done, info = env.step(action)
        cumulative_test_rewards += reward
        state = new_state
        
        if done:
            print ("Cumulative reward for episode {}: {}".format(episode, cumulative_test_rewards))
            break
    test_rewards.append(cumulative_test_rewards)
    
env.close()
print ("Test score over time: " + str(sum(test_rewards)/test_episodes))

x = range(test_episodes)
plt.plot(x, test_rewards)
plt.xlabel('episode')
plt.ylabel('Testing cumulative reward')
plt.show()
