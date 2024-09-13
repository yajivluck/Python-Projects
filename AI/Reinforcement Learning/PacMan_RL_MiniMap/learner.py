import os
import numpy as np
import pickle
import random
from Predict_State import get_next_state
from GameEntity import MapDescriptor


class Learner(object):
    
    #Default name of files for weights from previous training session
    #Greedy in the Limit with Infinite Exploration (GLIE) files

    WEIGHTS_FILE = "weights.p"
    GLIE_FILE = "glie.p"

    def __init__(self, alpha=0.01, gamma=0.7):
        #Load weights/glie if existing, else defaulted to 1 (for all weights) and 0.25 glie value
        #self.weights = self._load_file(self.WEIGHTS_FILE, [1] * 30)
        
        
        self.weights = self._load_file(self.WEIGHTS_FILE, np.ones(30, dtype=np.float64))

        
        self.glie = self._load_file(self.GLIE_FILE, 0.25)
        
        
        #Default values of weights at specific indices
        self.weights[self._to_weight_index(12)] = -100
        self.weights[self._to_weight_index(37)] = 200
        self.weights[self._to_weight_index(62)] = 10
        self.weights[self._to_weight_index(87)] = 50
        self.weights[self._to_weight_index(112)] = 200
        

        self.alpha = alpha
        self.gamma = gamma
        
    #Helper function to load files (for weights and glie)
    def _load_file(self, filename, default):
        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                data = pickle.load(file)
                print(f'{filename} initialized from file.')
                return data
        else:
            print(f'Starting from scratch: {filename}')
            return default

    #Utility calculating function
    def _get_utility(self, state):
        state_rewards = self._get_state(state)
        utility = np.dot([self.weights[self._to_weight_index(i)] for i in range(len(state_rewards))], state_rewards)
        return utility
    
    #Given a game state, this function returns the optimal action to take as well as the optimal utility that said action brings
    def get_optimal_action(self, game):
        optimal_utility = float("-inf")
        optimal_action_count = 0
        optimal_action = None
    
        available_actions = game.available_actions()
    
        # If exploring, pick a random action and return immediately.
        if random.random() <= self.glie:
            action = random.choice(available_actions)
            return action, self._get_utility(get_next_state(game, action))
    
    
        #If exploiting and not exploring, get utility of every potential next state
        for action in available_actions:
            next_state = get_next_state(game, action)
            utility = self._get_utility(next_state)
    
            # Found a better action, reset the count and update the action.
            if utility > optimal_utility:
                optimal_utility = utility
                optimal_action_count = 1
                optimal_action = action
            # Found another optimal action, increment the count.
            elif utility == optimal_utility:
                optimal_action_count += 1
                # Randomly decide whether to replace the previous action.
                if random.randint(1, optimal_action_count) == 1:
                    optimal_action = action
    
            # Early exit if the utility is the maximum possible utility
            # if utility == MAX_UTILITY:
            #     break
    
        return optimal_action, optimal_utility

    #This is the core of the model learning behaviour where the updated weights are computed
    def update_weights(self, prev_state, action, game, guess_utility, reward):
        self.glie = max(0.001, self.glie - 1e-5)
                
        curr_state = game.sliced_map.map.copy()
        curr_state[2, 2] = \
            prev_state[3, 2] if action == 2 else \
            prev_state[2, 1] if action == 3 else \
            prev_state[2, 3] if action == 4 else \
            prev_state[1, 2]
    
        # state rewards is a numpy array
        state_rewards = self._get_state(curr_state)
        real_utility = reward + self.gamma * self.get_optimal_action(game)[1]
        #Error function 
        #error = 0.5 * (real_utility - guess_utility) ** 2
    
        # Vectorize the weight update step:
        # Create a mask to exclude certain indices
        excluded_indices = np.array([12, 37, 62, 87, 112])
        all_indices = np.arange(len(state_rewards))
        included_indices = np.setdiff1d(all_indices, excluded_indices)
    
        # Calculate the updates for the included indices
        weight_updates = (self.alpha * (real_utility - guess_utility) * state_rewards[included_indices] / 
                          np.vectorize(self._to_weight_norm)(included_indices % 25))
    
        # Apply the updates to the weights using NumPy's advanced indexing
        self.weights[np.vectorize(self._to_weight_index)(included_indices)] += weight_updates
    
   
    def _get_state(self, game_map):
       all_state = game_map.flatten()
       size = len(all_state)
       total_state = np.zeros((5, size))

       for i, classification in enumerate([MapDescriptor.BAD_GHOST,
                                           MapDescriptor.GOOD_GHOST,
                                           MapDescriptor.PELLET,
                                           MapDescriptor.POWER_UP,
                                           MapDescriptor.FRUIT]):
           total_state[i, all_state == classification] = 1
           
       


       return total_state.flatten()

    
    def _to_weight_index(self, i):
        return self._to_weight_map(i % 25) + int(i / 25) * 6

    def _to_weight_map(self, i):
        return [
            0, 1, 2, 1, 0,
            1, 3, 4, 3, 1,
            2, 4, 5, 4, 2,
            1, 3, 4, 3, 1,
            0, 1, 2, 1, 0
        ][i]

    def _to_weight_norm(self, i):
        return [
            4, 8, 4, 8, 4,
            8, 4, 4, 4, 8,
            4, 4, 1, 4, 4,
            8, 4, 4, 4, 8,
            4, 8, 4, 8, 4
        ][i]

    #Reformat weights from 1D list to 2D as to make it an easier 1-1 mapping of what
    #The weights are on the map
    def formatted_weights(self):
        # Create a list comprehension that formats the weights and groups them in 5
        lines = [
            ' '.join('{:+05.2f}'.format(self.weights[self._to_weight_index(i + j)])
                     for j in range(5))
            for i in range(0, 5 * 25, 5)
        ] 
        # Join the groups of 5 weights with a newline, adding an additional newline every 5 lines
        formatted_weights = "\n\n".join("\n".join(lines[i:i + 5]) for i in range(0, len(lines), 5))  
        return formatted_weights

    #Helper function to save weights/glie file (to train model over multiple sessions)
    def save(self):
        pickle.dump(self.weights, open(self.WEIGHTS_FILE, "wb"))
        pickle.dump(self.glie, open(self.GLIE_FILE, "wb"))
        print('saved weights and glie')