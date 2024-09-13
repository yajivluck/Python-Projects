# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:20:49 2023
@author: Kiran
"""

import argparse
import numpy as np
from Interface import MsPacmanGame
from CNNLearning import TDLearningAgentCNN

# Set up argument parser
parser = argparse.ArgumentParser(description="Train a TD Learning Agent on Ms. Pacman")
parser.add_argument("--weights_path", type=str, default=None, help="Path to the weights file to load (optional)")
parser.add_argument("--display", type=bool, default=False, help="Visualize the game (default: True)")
parser.add_argument("--learn", type=bool, default=True, help="Enable learning (default: False)")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor (default: 0.95)")
parser.add_argument("--epsilon", type=float, default=1.0, help="Exploration rate (default: 1.0)")
parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate (default: 0.995)")
parser.add_argument("--epsilon_min", type=float, default=0.01, help="Minimum epsilon (default: 0.01)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 32)")
parser.add_argument("--num_episodes", type=int, default=10, help="Total number of episodes to train (default: 10)")

args = parser.parse_args()

# Initialize the game and the agent
game = MsPacmanGame("MSPACMAN.bin", fps=60, scale_factor=5, display=True)
agent = TDLearningAgentCNN(action_size=4, learning_rate=args.learning_rate, gamma=args.gamma, num_features=12)


#weight_path = "C:/Users/Kiran/Desktop/Python Projects/Python-Projects/AI/Reinforcement Learning/PacMan_RL/Weights/epsilon_1/ms_pacman_model_10.h5"

# Load weights if the path is provided
if args.weights_path:
    agent.load(args.weights_path)
    

else:
    print('starting from scratch')

# To keep track of agent score per episode
score_per_episodes = []

# For each episode
for e in range(1,11):
    game.reset()  # Reset the game state
    state = game.get_state_with_rgb()  # Get initial state of the game
    total_reward = 0  # Initialize total_reward as zero
    step_count = 0  # To keep track of the number of steps

    while not game.game_over():
        if game.get_lives() == 0:
            break

        if np.random.rand() <= args.epsilon:
            action = np.random.choice([0, 1, 2, 3])
        else:
            action = agent.act(state)
        
        if args.display:
            game.render()

        next_state, reward, done = game.step(action, rgb = True)
        
        if reward != 0:
            agent.remember(state, action, reward, next_state, done)
            if args.learn and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)

        step_count += 1
        state = next_state
        total_reward += reward

    score_per_episodes.append((e, total_reward))

    if args.learn and args.epsilon > args.epsilon_min:
        args.epsilon *= args.epsilon_decay

    print(f"Episode: {e}/{args.num_episodes}, Total Reward: {total_reward}")

    if args.learn:
        agent.save(f"cnn{e}.h5")
        
        

