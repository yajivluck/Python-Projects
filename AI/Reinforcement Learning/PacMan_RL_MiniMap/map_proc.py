import numpy as np
from collections import deque
from GameEntity import MapDescriptor


def get_slice(game_map, pac_pos, radius):

    min_i = pac_pos[0] - radius
    max_i = pac_pos[0] + radius + 1
    min_j = pac_pos[1] - radius
    max_j = pac_pos[1] + radius + 1

    # Calculate vertical and horizontal slices, ensuring they are within the map bounds
    vertical_slice = slice(max(min_i, 0), min(max_i, game_map.HEIGHT))
    horizontal_slice = slice(max(min_j, 0), min(max_j, game_map.WIDTH))

    # Extract the slice from the map
    map_slice = game_map.map[vertical_slice, horizontal_slice]

    # Handle horizontal wraparound by concatenating the opposite side of the board
    if min_j < 0:
        map_slice = np.hstack((game_map.map[vertical_slice, min_j:], map_slice))
    elif max_j > game_map.WIDTH:
        overflow_width = max_j - game_map.WIDTH
        map_slice = np.hstack((map_slice, game_map.map[vertical_slice, :overflow_width]))

    # Add walls for vertical overflow
    if min_i < 0 or max_i > game_map.HEIGHT:
        # Determine the number of rows of walls needed and create them
        top_wall_height = abs(min_i) if min_i < 0 else 0
        bottom_wall_height = max_i - game_map.HEIGHT if max_i > game_map.HEIGHT else 0

        # Create the top and/or bottom walls
        top_wall = np.full((top_wall_height, map_slice.shape[1]), fill_value=MapDescriptor.WALL, dtype=np.uint8)
        bottom_wall = np.full((bottom_wall_height, map_slice.shape[1]), fill_value=MapDescriptor.WALL, dtype=np.uint8)

        # Concatenate the walls with the map slice
        map_slice = np.vstack((top_wall, map_slice, bottom_wall))


    return hide_cells_behind_wall(map_slice)


def hide_cells_behind_wall(map_slice, wall_value=1):
    """Hides cells which cannot be reached by Ms. PacMan, assumed to be at the center of the slice.

    Args:
        map_slice: Map slice matrix.
        wall_value: The value that represents a wall in the map slice matrix.

    Returns:
        Map slice with unreachable cells set to the wall value.
    """
    height, width = map_slice.shape
    center = (height // 2, width // 2)  # Assuming the center is always in the middle of the map slice

    # Initialize all cells as walls (unreachable)
    shadowed_map = np.full((height, width), wall_value)
    visited = np.zeros((height, width), dtype=bool)
    neighbor_queue = deque([center])

    # Directions to check (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while neighbor_queue:
        y, x = neighbor_queue.popleft()
        
        # Skip if this cell is a wall or has been visited
        if map_slice[y, x] == wall_value or visited[y, x]:
            continue

        # Mark cell as visited and copy the cell value to the shadowed map
        visited[y, x] = True
        shadowed_map[y, x] = map_slice[y, x]

        # Add non-visited, non-wall neighbors to the queue
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                neighbor_queue.append((ny, nx))

    return shadowed_map

