import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, concatenate

from tensorflow.keras.layers import LSTM, TimeDistributed, Reshape


# Setting up GPU usage if GPU available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Setting memory growth to true to allocate only as much GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Exception if GPU setup failed
        print(e)

class TDLearningAgentCNN:
    
    def __init__(self, action_size=4, learning_rate=0.01, gamma=0.99, num_features=12):  # Adjusted num_features
        self.num_features = num_features
        self.action_size = action_size
        self.memory = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = self._build_model()
    
    def _build_model(self):
        # Feature input
        feature_input = Input(shape=(self.num_features,))
        fc1 = Dense(64, activation='relu')(feature_input)
    
        # Image input
        image_input = Input(shape=(60,60,1))  # Adjusted shape
        conv1 = Conv2D(16, (3, 3), activation='relu')(image_input)
        conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
        flatten = Flatten()(conv2)
    
        # Merge the two inputs
        merged = concatenate([fc1, flatten])
        fc2 = Dense(32, activation='relu')(merged)
    
        # Output layer
        output = Dense(self.action_size, activation='linear')(fc2)
        
        # Create the model
        model = Model(inputs=[feature_input, image_input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    
        return model
    


    # def _build_model(self):
    #     # Feature input
    #     feature_input = Input(shape=(None, self.num_features))  # Adjusted for time-distributed input
    #     # TimeDistributed wrapper to apply a Dense layer to each time step independently
    #     fc1 = TimeDistributed(Dense(64, activation='relu'))(feature_input)
        
    #     # Adding LSTM layer to process the sequence of feature vectors
    #     lstm_features = LSTM(64, return_sequences=False)(fc1)  # Adjust as needed

    #     # Image input
    #     image_input = Input(shape=(60, 60, 1))  # Assuming single channel image input, adjust if needed
    #     # TimeDistributed wrapper to apply Conv2D layer to each time step independently
    #     conv1 = TimeDistributed(Conv2D(16, (3, 3), activation='relu'))(image_input)
    #     conv2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(conv1)
    #     flatten = TimeDistributed(Flatten())(conv2)
        
    #     # Adding LSTM layer to process the sequence of images
    #     lstm_images = LSTM(64, return_sequences=False)(flatten)  # Adjust as needed

    #     # Merge the two inputs
    #     merged = concatenate([lstm_features, lstm_images])
    #     fc2 = Dense(32, activation='relu')(merged)
    
    #     # Output layer
    #     output = Dense(self.action_size, activation='linear')(fc2)
        
    #     # Create the model
    #     model = Model(inputs=[feature_input, image_input], outputs=output)
    #     model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    
    #     return model



    def update_model(self, state, action, reward, next_state, done):
        state_feature, state_image = state
        next_state_feature, next_state_image = next_state

        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict([next_state_feature, next_state_image], verbose=0)[0])
        
        target_f = self.model.predict([state_feature, state_image], verbose=0)
        target_f[0][action] = target
        self.model.fit([state_feature, state_image], target_f, epochs=1, verbose=0)

    def act(self, state):
        state_feature, state_image = state
        act_values = self.model.predict([state_feature, state_image], verbose=0)
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
        print('saved model')
        self.model.save_weights(name)
    
    def load(self, name):
        if os.path.exists(name):
            self.model.load_weights(name)
