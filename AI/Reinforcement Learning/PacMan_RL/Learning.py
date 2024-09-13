# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:20:49 2023
@author: Kiran
"""

import argparse
import numpy as np
from Interface import MsPacmanGame
from TDLearning import TDLearningAgent

# Set up argument parser
parser = argparse.ArgumentParser(description="Train a TD Learning Agent on Ms. Pacman")
parser.add_argument("--weights_path", type=str, default=None, help="Path to the weights file to load (optional)")
parser.add_argument("--display", type=bool, default=False, help="Visualize the game (default: True)")
parser.add_argument("--learn", type=bool, default=False, help="Enable learning (default: False)")
parser.add_argument("--action_size", type=int, default=4, help="Number of actions (default: 4), do not change")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor (default: 0.95)")
parser.add_argument("--epsilon", type=float, default=1, help="Exploration rate (default: 1.0)")
parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate (default: 0.995)")
parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum epsilon (default: 0.01)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--num_episodes", type=int, default=10, help="Total number of episodes to train (default: 10)")
#parser.add_argument("--skip_states", type=int, default=30, help="Number of states to skip (default: 30), higher means faster training, but ")

args = parser.parse_args()

# Initialize the game and the agent
game = MsPacmanGame("MSPACMAN.bin", fps=60, scale_factor=5, display=False)
agent = TDLearningAgent(action_size=4, learning_rate=args.learning_rate, gamma=args.gamma, num_features=12)


score_per_episodes = []

e_start = 121

agent.load("weight_new_batch111.h5")


# For each episode
for e in range(e_start, e_start + 9):

    decayed_epsilon = args.epsilon * args.epsilon_decay ** e
    game.reset()  # Reset the game state
    state = game.get_state()  # Get initial state of the game
    total_reward = 0  # Initialize total_reward as zero
    step_count = 0  # To keep track of the number of steps

    while not game.game_over():
        

        if game.get_lives() == 0:
            break

        if np.random.rand() <= decayed_epsilon:
            action = np.random.choice([0, 1, 2, 3])
        else:
            action = agent.act(state)
        
        if args.display:
            game.render()

        next_state, reward, done = game.step(action)
        
        #only count experience if non zero reward (train on meaninful actions/states?)
        if reward != 0:
            agent.remember(state, action, reward, next_state, done)
            if args.learn and len(agent.memory) > args.batch_size:
                
                print('learning')
                
                agent.replay(args.batch_size)

        step_count += 1
        state = next_state
        total_reward += reward

    score_per_episodes.append((e, total_reward))

    if args.learn and args.epsilon > args.epsilon_min:
        args.epsilon *= args.epsilon_decay

    print(f"Episode: {e}/{e_start + 10}, Total Reward: {total_reward + 300}")

    if args.learn:
        agent.save(f"weight_new_batch{e}.h5")
        
        
