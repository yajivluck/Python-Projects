from MapViewer import get_submap
from game_map import GameMap
from GameEntity import MapDescriptor

#Helper function to determine if a position is within some bounds
def is_position_within_bounds(position, height, width):
    x, y = position
    return 0 <= x < height and 0 <= y < width

#Get next state of the game
def get_next_state(game, action):
    #Get the move that ms.pacman last did
    move = game.action_to_move(action)
    #Get the next position of ms.pacman based on said move
    new_pos = game.get_next_position(game.ms_pacman_position, move)
    
    # Create a fresh game map from the blank map template
    game_map = GameMap.from_map(game._blank_map.map.copy())
    
    # Place fruit on the map if it exists
    if game.fruit.exists:
        game_map.map[game.fruit.position] = MapDescriptor.FRUIT
    
    # Update the positions of the ghosts
    for ghost in game.ghosts:
        #New pos of the ghost 
        new_ghost_position = game.get_next_position(ghost.position, ghost.direction)
        #Bool to check if the ghost new position is the same as ms.pacman's new position
        ghost_on_new_pos = ghost.position == new_pos
        #Bool to check if the ghost's new position is within the game's boundaries
        ghost_within_bounds = is_position_within_bounds(new_ghost_position, game_map.HEIGHT, game_map.WIDTH)
        
        #If the ghost new position is out of bounds or at the same position as ms.pacman
        if not ghost_within_bounds or ghost_on_new_pos:
            # If ghost is not within bounds or on the new position of Ms. Pacman,
            # it should remain in its current position.
            
            #Set ghost_marker based on ghost's state bit(0 -> bad, 1 -> good)
            ghost_marker = MapDescriptor.GOOD_GHOST if ghost.state else MapDescriptor.BAD_GHOST
            #Update map at corresponding position
            game_map.map[ghost.position] = ghost_marker
            
        #If ghost within bounds and not on ms.pacman's new position and it isn't trying to move into a wall
        elif game_map.map[new_ghost_position] != MapDescriptor.WALL:
            # Only move the ghost if it's not trying to move into a wall.
            
            #Set ghost_marker appropriately
            ghost_marker = MapDescriptor.GOOD_GHOST if ghost.state else MapDescriptor.BAD_GHOST
            #Update map accordingly
            game_map.map[new_ghost_position] = ghost_marker
    
    
    #TODO CHECK FOR 2->3 SEE IF BETTER IMPROVEMENT OF MODEL AS IT SEES MORE (LESS ODDS OF NOT SEEING ANYTHING
    #BEYOND ITS SLICE IF THAT IS THE ONLY THING IT CONSIDERS AS ITS SPACE STATE)
    # Return a slice of the game map centered on Ms. Pacman's new position
    return get_submap(game_map, new_pos, 2)

