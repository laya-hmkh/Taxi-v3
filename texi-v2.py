import gym
import numpy as np
import random
from IPython.display import clear_output

# Create the Taxi-v3 environment with human rendering mode
env = gym.make("Taxi-v3", render_mode="human")
env.reset()

# Define the action and state spaces
actions = env.action_space  # 6 possible actions
states = env.observation_space  # 500 possible states

# Set the learning parameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate
all_epochs = []
all_penalties = []
steps = 100001

# Initialize the Q-table with zeros
q_table = np.zeros([states.n, actions.n])

"""Training the agent"""

for i in range(1, 10001):  # Train for 10,000 episodes
    state = env.reset()  # Reset the environment to get the initial state

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)  # Take the action and get the next state and reward

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Update the Q-value using the Q-learning update rule
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1  # Count the penalties

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100  # Evaluate for 100 episodes

for _ in range(episodes):
    state = env.reset()  # Reset the environment to get the initial state

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])  # Choose the best action based on the Q-table
        state, reward, done, info = env.step(action)  # Take the action and get the next state and reward

        if reward == -10:
            penalties += 1  # Count the penalties
        epochs += 1

    total_epochs += epochs
    total_penalties += penalties

env.close()  # Close the environment

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
