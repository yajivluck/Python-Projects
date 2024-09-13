class Mobs(object):

    """Mobs that can move in Ms.PacMan have a position and direction of motion (N,S,E,W)
    This class accesses information on the Ms.PacMan game from specific RAM address accesses.
    Without those accesses, emulating the game on Stella and training the model has poor 
    performance that it becomes quite impossible to observe the agent's behaviour in real time
    as the output is 'laggy'
    """
    #instantiation of position/direction
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    @classmethod
    #given a position and ram, return an instance of the object with its updated direction
    def update(cls, position, ram):
        direction = cls._get_direction(ram)
        return Mobs(position, direction)
    
    @classmethod
    #Method returns the direction that the object is currently moving towards as a tuple (i,j)
    #Uses bit-wise comparison to return between four different directions
    def _get_direction(cls, ram):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        return directions[ram & 3]

#Ghost Class inheriting from the Mobs Class
class Ghost(Mobs):

    """Ghost have two different states: (Good or Bad). Good being we can encourage 
    our agent for being in contact with the ghost in that state and Bad (default state)
    where we discourage our agent from being in contact with the ghosts"""

    #instantiate a ghost object that inherits from the Mobs class
    #To keep track of its own position and direction. Set state of
    #Ghost as class attribute
    def __init__(self, position, direction, state):
        super(Ghost, self).__init__(position, direction)
        self.state = state

    #Get Direction from ram as well as state of ghost. Direction is
    #located in the same bits so it is retrieved by the Mobs super class
    #methodand state is retrieved as the 7th bit in the ram.
    
    #7th bit = 1 -> Good ghost
    #7th bit = 0 -> Bad Ghost
    @classmethod
    def update(cls, position, ram):
        direction = cls._get_direction(ram)
        state = (ram >> 7) & 1  
        return cls(position, direction, state)

#Fruit class inherits from the Mobs class
class Fruit(Mobs):

    """Fruits in Ms.PacMan offers additional points which can
    be used to reward the agent that comes into contact with one"""


    #instantiation of a fruit by its position, direction is done
    #the same way as for any Mobs through its super class
    #Another attribute "exists" is defined as not every game states
    #hold a fruit
    def __init__(self, position, direction, exists):
        super(Fruit, self).__init__(position, direction)
        self.exists = exists

    #Update the fruit's attributes the same way as the other Mobs Objects
    #through ram address. Return an instance of the class with the updated attributes.
    @classmethod
    def update(cls, position, ram, exists):
        # Direction is probably bad...
        direction = cls._get_direction(ram)
        return cls(position, direction, exists)

####


#Map Descriptor class that maps every possible objects whether they are Mobs or the Map itself of the game.
#This class relates each objects with an index to identify them. Each object (index) is assigned a reward number
#and a color (RGB) that will guide out agent into recognizing what is good/bad for its performance and how to
#Recognize these objects based on their color.
class MapDescriptor(object):
    """Game map object enumerations."""

    EMPTY = 0
    WALL = 1
    PELLET = 2
    POWER_UP = 3
    GOOD_GHOST = 4
    BAD_GHOST = 5
    FRUIT = 6
    MS_PACMAN = 7

    # Mapping of map objects to their rewards
    REWARDS = {
        WALL: 0,
        PELLET: 200,
        POWER_UP: 50,
        GOOD_GHOST: 100,
        BAD_GHOST: -250,
        FRUIT: 75,
        MS_PACMAN: 0,
        EMPTY: 0
    }

    # Mapping of map objects to their colors (BGR)
    COLORS = {
        WALL: [111, 111, 228],  # Pink-ish
        PELLET: [255, 255, 255],  # White
        POWER_UP: [255, 255, 0],  # Cyan
        GOOD_GHOST: [0, 255, 0],  # Green
        BAD_GHOST: [0, 0, 255],  # Red
        FRUIT: [255, 0, 255],  # Magenta
        MS_PACMAN: [0, 255, 255],  # Yellow
        EMPTY: [136, 28, 0]  # Dark blue (default)
    }

    @classmethod
    def to_reward(cls, classification):
        return cls.REWARDS.get(classification, 0)

    #This class will return dark blue as a default color if no object is found. This is useful as when a pellet
    #is captured by Ms.PacMan, this area of the map no longer holds anything which we can train the agent to perceive an
    #Area with nothing as unfavorable (or neutral) as it is not getting any more rewards from it which can still be mapped
    #To a color.
    @classmethod
    def to_color(cls, classification):
        return cls.COLORS.get(classification, [136, 28, 0])  # Default to dark blue if not found



