import gym
import numpy as np
import random
from IPython.display import clear_output

env = gym.make("Taxi-v3", render_mode = "human")
env.reset()

actions = env.action_space # 6
states = env.observation_space # 500

alpha = 0.1
gamma = 0.6
epsilon = 0.1
all_epochs = []
all_penalties = []
steps = 100001

#print(env.P[328])
#print(f"State : {env.encode(3, 1, 2, 0)}")# 328
# {action: [(probability, nextstate, reward, done)]}

q_table = np.zeros([states.n, actions.n])

"""Training the agent"""

for i in range(1, 5):
    state, a = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, l, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 5 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 3

for _ in range(episodes):
    state, a = env.reset()
    
    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, l, info = env.step(action)
        env.close()
        
        if reward == -10 : 
            penalties += 1
        epochs += 1
        
    total_epochs += epochs
    total_penalties += penalties
    
    
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")