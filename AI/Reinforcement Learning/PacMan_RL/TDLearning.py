# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:07:06 2023

@author: Kiran
"""

import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.layers import LSTM


#Import interface of MSPacmanGame
#from Interface import MsPacmanGame
#rom path to mspacman game
#rom_path = 'MSPACMAN.bin'
#instantiate game variable
#game = MsPacmanGame(rom_path, fps=60, scale_factor=5)


# Setting up GPU usage if gpu available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Setting memory growth to true to allocate only as much GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Exception if GPU setup failed
        print(e)



class TDLearningAgent:
    
    def __init__(self, action_size=4, learning_rate=0.01, gamma=0.99, num_features=12):
        self.num_features = num_features  # Number of features
        self.action_size = action_size  # Number of actions
        self.memory = []  # Experience replay buffer
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.model = self._build_model()  # Build model
    
    # def _build_model(self):
    #     # Input layer
    #     feature_input = Input(shape=(self.num_features,))
        
    #     # One fully connected layer
    #     fc = Dense(6, activation='relu')(feature_input)
        
    #     # Output layer
    #     output = Dense(self.action_size, activation='linear')(fc)
        
    #     # Create the model
    #     model = Model(inputs=feature_input, outputs=output)
    #     model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
    #     return model
    
    
    def _build_model(self):
        # Input layer
        feature_input = Input(shape=(self.num_features,))
    
        # First fully connected layer
        fc1 = Dense(128, activation='relu')(feature_input)
        drop1 = Dropout(0.2)(fc1)  # Dropout layer for regularization
    
        # Second fully connected layer
        fc2 = Dense(64, activation='relu')(drop1)
        drop2 = Dropout(0.2)(fc2)  # Another dropout layer
    
        # Third fully connected layer
        fc3 = Dense(32, activation='relu')(drop2)
    
        # Output layer
        output = Dense(self.action_size, activation='linear')(fc3)
    
        # Create the model
        model = Model(inputs=feature_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    
        return model


    def update_model(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose = 0)[0])
        
        target_f = self.model.predict(state.reshape(1, -1), verbose = 0)
        target_f[0][action] = target
        self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    def act(self, state):
        act_values = self.model.predict(state.reshape(1, -1), verbose = 0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.update_model(state, action, reward, next_state, done)

    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        if os.path.exists(name):
            print('loading', name)
            self.model.load_weights(name)
