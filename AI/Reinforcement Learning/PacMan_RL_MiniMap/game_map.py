import cv2
import numpy as np
from MapViewer import get_submap
from GameEntity import MapDescriptor


#Game Map Class that defines attributes of the visualization of the game's map for our agent
class GameMap(object):
    
    WIDTH, HEIGHT, DOMINANT_COLOR = 20, 14, 74
    
    def __init__(self, image, game_map=None):
        # Discard everything but map.
        if game_map is not None:
            self._map = game_map
            return

        self.image = image[2:170]

        height, width = self.image.shape
        self._width_step = width / self.WIDTH
        self._height_step = height / self.HEIGHT
        self.classify()

    @classmethod
    def from_map(cls, game_map):
        return cls(None, game_map)

    @property
    def map(self):
        """Map of GameMapObjects."""
        return self._map
                
    def classify_histogram(self, histogram):
       # We can precalculate the thresholds for the counts instead of the ratios to avoid division in each call.
       wall_threshold = 0.40 * self._width_step * self._height_step
       power_up_threshold = 0.25 * self._width_step * self._height_step
       pellet_threshold = 0.05 * self._width_step * self._height_step

       primary_count = histogram[self.DOMINANT_COLOR]

       if primary_count >= wall_threshold:
           return MapDescriptor.WALL
       elif primary_count >= power_up_threshold:
           return MapDescriptor.POWER_UP
       elif primary_count >= pellet_threshold:
           return MapDescriptor.PELLET
       else:
           return MapDescriptor.EMPTY

    def classify_partition(self, partition):
       # Computing the histogram can often not be vectorized due to the nature of the function.
       # We need to keep it as is, but ensure that no unnecessary calculations are done.
       histogram = cv2.calcHist([partition], [0], None, [256], [0, 256]).flatten()
       return self.classify_histogram(histogram)

    def classify(self):
        # Make sure the steps are integers
        width_step = int(self._width_step)
        height_step = int(self._height_step)
    
        # Calculate all partition indices upfront
        width_starts = (np.arange(self.WIDTH) * width_step).astype(int)
        height_starts = (np.arange(self.HEIGHT) * height_step).astype(int)
    
        # Initialize the map with empty descriptors
        self._map = np.full((self.HEIGHT, self.WIDTH), MapDescriptor.EMPTY, dtype=np.uint8)
    
        # Process each partition
        for i, w_start in enumerate(width_starts):
            for j, h_start in enumerate(height_starts):
                partition = self.image[h_start:h_start + height_step, w_start:w_start + width_step]
                self._map[j, i] = self.classify_partition(partition)

    
    def to_image(self):
        # Initialize the image with zeros, with shape derived from map dimensions.
        image_shape = (self.HEIGHT, self.WIDTH, 3)
        image = np.zeros(image_shape, dtype=np.uint8)
        
        # Convert each classification to a color vector using a vectorized operation.
        for classification_code in np.unique(self._map):
            # Find the color corresponding to the current classification code.
            color = MapDescriptor.to_color(classification_code)
            # Apply this color to all positions in the image array where the classification code matches.
            image[self._map == classification_code] = color
        
        # Resize the image to the desired scale.
        upscaled_image = cv2.resize(image, (160, 168), interpolation=cv2.INTER_NEAREST)
        return upscaled_image

#Sliced game map class which basically takes a subsection of the map centered around Ms.Pacman with a radius predefined

#TODO CHECK IF CHANGING RADIUS HERE MAKES BETTER MODEL (SEES MORE?)
class SlicedGameMap(object):
    RADIUS = 2
    def __init__(self, game_map, ms_pacman_position):
        #Get slice funtoin from MapViewer 
        self._map = get_submap(game_map, ms_pacman_position, self.RADIUS)

    @property
    def map(self):
        return self._map

    
    def to_image(self):
       # Create a blank image array based on the map shape and desired color depth.
       image = np.zeros((*self._map.shape, 3), dtype=np.uint8)
       
       # Vectorized operation to convert each classification to a color.
       for classification in np.unique(self._map):
           image[self._map == classification] = MapDescriptor.to_color(classification)
       
       # Upscale the image to the desired size.
       upscaled_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_NEAREST)
       return upscaled_image
    
    
    
