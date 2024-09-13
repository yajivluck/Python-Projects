import numpy as np
from ale_py import ALEInterface
import pygame
import cv2


from collections import deque


class MsPacmanGame:
    def __init__(self, rom_file_path, fps=30, scale_factor=5, display = False):
        # Initialize ALE and Pygame
        self.ale = ALEInterface()
        self.ale.loadROM(rom_file_path)
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.screen_data = np.zeros((self.screen_width, self.screen_height, 3), dtype=np.uint8)

        self.scale_factor = scale_factor
        
        self.total_reward = 0

        # Scale the display window
        scaled_width = self.screen_width * self.scale_factor
        scaled_height = self.screen_height * self.scale_factor

    
        
        if display:
            pygame.init()
            #full resolution game
            self.display = pygame.display.set_mode((scaled_width, scaled_height))
            pygame.display.set_caption('Ms. Pacman')
        

        self.clock = pygame.time.Clock()
        self.fps = fps  # Frames per second
        
        
    def get_total_reward(self):
        return self.total_reward
        
    def get_ram(self):
       # Get the current state of the game's RAM
       return self.ale.getRAM()
        
    def get_lives(self):
        return self.ale.lives()

    def reset(self):
        self.ale.reset_game()

    #This function returns the reward of an action
    def step(self, action, rgb = False):
        
        
        current_lives = self.get_lives()
        
        #print('action', action)
        #Translate index number to actual action
        action = self.get_possible_moves()[action]
        reward = self.ale.act(action)
        
        new_lives = self.get_lives()
        
        
        if current_lives != new_lives:
            reward -= 100
            

        #Add to total reward based on step 
        self.total_reward += reward
        
        next_state = self.get_state()
        if rgb:
            next_state = self.get_state_with_rgb()
        
        done = self.game_over()

        
        
        return next_state, reward, done
    
    
    #Function returns the current state of the game

    
    def get_state_with_rgb(self):
        current_screen = self.binary_threshold(threshold = 50)
        ghost_positions, ghost_states = self.get_ghost_pos()
        current_pos = self.get_player_pos()
        current_lives = self.get_lives()
        total_reward = self.get_total_reward()
        
        
        # Calculate distances between Ms. Pacman and each ghost
        distances = [np.sqrt((current_pos[0] - ghost_x)**2 + (current_pos[1] - ghost_y)**2) for ghost_x, ghost_y in ghost_positions]
    
    
        # Flatten the distances, ghost states, and Ms. Pacman's position into a single array
        flattened_features = distances + ghost_states + list(current_pos) + [current_lives, total_reward]
        
        # Ensure the flattened features list has 14 elements as expected by the model
        #assert len(flattened_features) == 14, "The features array does not have 14 elements."
    
        # Convert flattened_features to a numpy array and reshape it to (1, 14) for model input
        features_input = np.array(flattened_features).reshape(1, 12)
    
        # Reshape current_screen to (1, 160, 210, 3) for model input
        image_input = current_screen.reshape(1, 60, 60)
        
        return features_input,image_input

    def get_state(self):
        ghost_positions, ghost_states = self.get_ghost_pos()
        current_pos = self.get_player_pos()
        current_lives = self.get_lives()  # Current lives in the game
        total_reward = self.get_total_reward()  # Total score in the game
        
        # Calculate distances between Ms. Pacman and each ghost
        distances = [np.sqrt((current_pos[0] - ghost_x)**2 + (current_pos[1] - ghost_y)**2) for ghost_x, ghost_y in ghost_positions]
    
        # Flatten the distances, ghost states, and Ms. Pacman's position into a single array
        flattened_features = distances + ghost_states + list(current_pos) + [current_lives, total_reward]
        
        # Ensure the flattened features list has the expected number of elements
        # Adjust the length assertion as per the number of features now
        assert len(flattened_features) == (len(distances) + len(ghost_states) + 2 + 2), "The features array does not have the expected number of elements."
        
        # Convert flattened_features to a numpy array and reshape it for model input
        features_input = np.array(flattened_features).reshape(1, -1)
        
        return features_input
    
    
    def get_slice(self, radius):
        full_screen = self.get_current_screen_rgb()
        center_x, center_y = self.get_player_pos()
    
        radius = min(radius, self.screen_height // 2, self.screen_width // 2)
    
        # Desired slice dimensions
        slice_height = slice_width = 2 * radius
    
        # Extract the slice with horizontal and vertical wrapping and padding
        zero_padded_slice = self.extract_slice_with_wrapping(full_screen, center_x, center_y, radius, slice_height, slice_width)
    
        return zero_padded_slice

    def extract_slice_with_wrapping(self, full_screen, center_x, center_y, radius, slice_height, slice_width):
        zero_padded_slice = np.zeros((slice_height, slice_width, 3), dtype=full_screen.dtype)
    
        for i in range(slice_height):
            for j in range(slice_width):
                # Calculate the corresponding position on the full_screen
                horizontal_idx = (center_x - radius + j) % self.screen_width
                vertical_idx = (center_y - radius + i) % self.screen_height
    
                # Check if the indices are within the bounds of the full_screen
                if (0 <= horizontal_idx < self.screen_width) and (0 <= vertical_idx < self.screen_height):
                    zero_padded_slice[i, j, :] = full_screen[vertical_idx, horizontal_idx, :]
                else:
                    # Zero padding for out of bounds areas
                    zero_padded_slice[i, j, :] = 0
    
        return zero_padded_slice
    
    

    def binary_threshold(self, threshold):
        
        rgb_array = self.get_slice(radius = 30)
        
        # Convert the RGB array to a grayscale image
        gray_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Apply binary thresholding
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        
        return binary_image
    

    def get_current_screen_rgb(self, process = False):
        
        if process:
            self.processed_screen = self.preprocess_frame(self.screen_data)
            return self.processed_screen
        
        self.ale.getScreenRGB(self.screen_data)
        return self.screen_data


    def render(self, radius=None):
        if radius is not None:
            # If radius is specified, get a slice of the screen
            #frame_data = self.get_slice(radius)
            
            #frame_data = self.binary_threshold(threshold = 50)
            pass
            
        else:
            # Otherwise, get the full screen
            self.get_current_screen_rgb(process=False)
            frame_data = self.screen_data

        # Convert the screen data or slice to a Pygame surface
        frame = pygame.surfarray.make_surface(frame_data)

        # Rotate and flip the frame
        frame = pygame.transform.rotate(frame, -90)
        frame = pygame.transform.flip(frame, True, False)

        # Scale the frame
        frame = pygame.transform.scale(frame, (frame_data.shape[1] * self.scale_factor, frame_data.shape[0] * self.scale_factor))

        # Blit the frame to the display surface and update the display
        self.display.blit(frame, (0, 0))
        pygame.display.flip()

        # Control the game's frame rate
        self.clock.tick(self.fps)
        
    def get_possible_moves(self):       
        all_moves = self.ale.getLegalActionSet()
        #Only moves possible in Ms.Pacman is Up Down Left Right
        
        return all_moves[2:6]
    
    def game_over(self):
        return self.ale.game_over()

    def close(self):
        pygame.quit()
        
       
    def play(self, debug=False):
       self.reset()
       while not self.game_over():
           current_RGB = self.get_current_screen_rgb()
           
           for event in pygame.event.get():
               if event.type == pygame.QUIT:
                   self.close()
                   return


           action = np.random.choice([0,1,2,3])       
           _, reward, _ = self.step(action)
                     
           #print(self.get_lives(), self.game_over())

           if debug:
               ms_pacman_pos = self.get_player_pos()
               ghost_positions, ghost_state = self.get_ghost_pos()
               distances = [ self.calculate_distance(ms_pacman_pos,ghost_pos) for ghost_pos in ghost_positions]
                   
               #print(ghost_state)
               
               #Distance of 10 is bad
               #print(min(distances))
               

           #self.render_lite()
           
           
           self.render(radius = None)
           
           
            
 
    def get_ghost_pos(self):  
        ram = self.get_ram()
        
        new_ghosts_ram = [
            (int(ram[6]), int(ram[12]), (ram[1] >> 7) & 1),
            (int(ram[7]), int(ram[13]), (ram[2] >> 7) & 1),
            (int(ram[8]), int(ram[14]), (ram[3] >> 7) & 1),
            (int(ram[9]), int(ram[15]), (ram[4] >> 7) & 1)
        ]

        ghost_positions = [(x, y) for x, y, _ in new_ghosts_ram]
        ghost_state = [state for _, _, state in new_ghosts_ram]
         
        return ghost_positions, ghost_state
        
    
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    
    
    def get_player_pos(self):
        """Updates the internal state of the game."""
        # Get new states from RAM.
        ram = self.get_ram()
        new_ms_pacman_position = (int(ram[10]), int(ram[16]))
       
        return new_ms_pacman_position
    
    
    
    
    
    
    
    
    
    
    
    


        
        

#Example usage
# rom_path = 'MSPACMAN.bin'
# game = MsPacmanGame(rom_path, fps=60, scale_factor=5, display = True)
# game.play(debug=True)

