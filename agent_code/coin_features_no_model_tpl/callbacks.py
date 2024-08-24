import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

events_pre = []  # needed to check events without recalculating stuff

def setup(self):
    """
    Set up your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are independent of the game state.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    features = []

    _, _, bomb_available, (ax, ay) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    # bombs = game_state['bombs']  # [(x, y), countdown]

    features.append(ax)
    features.append(ay)
    # features.append(int(bomb_available))

    blocked = is_blocked(game_state)
    features.extend(blocked)
    events_pre.append(blocked)

    # make positions relative
    nearest_coins = k_nearest_coins(ax, ay, coins)
    for (cx, cy) in nearest_coins:
        features.append(cx - ax)
        features.append(cy - ay)

    # Direction to the nearest coin (one-hot encoded)
    if nearest_coins:
        direction_idx = direction_to_nearest_coin(ax, ay, blocked, nearest_coins[0], arena)
    else:
        direction_idx = None  # If no coins are available
    events_pre.append(direction_idx)

    # One-hot encode the direction (0: right, 1: down, 2: left, 3: up)
    direction_one_hot = [0, 0, 0, 0]
    if direction_idx is not None:
        direction_one_hot[direction_idx] = 1
    features.extend(direction_one_hot)

    # todo:
    # k nearest crates
    # direction to nearest crate
    # 4 bomb positions
    # is in explosion radius (with bomb timer, inverse intensity)
    # positions of other agents
    # embedding of entire game state?

    # and return them as a vector
    return features  # length: 1+1+4+6+4 = 16


def is_blocked(game_state: dict):
    """Returns a vector of 4 values, for each direction whether it
    is blocked by a wall, crate, agent, bomb or (imminent) explosion."""
    _, _, _, (ax, ay) = game_state('self')
    arena = game_state['field']
    explosions = game_state['explosion_map']  # Non-zero values = explosion
    bombs = game_state['bombs']  # [(x, y), countdown], if it explodes in the next step add that to explosion map
    others = [(x, y) for _, _, _, (x, y) in game_state['others']]

    directions = [(ax + 1, ay), (ax, ay + 1), (ax - 1, ay), (ax, ay - 1)]  # Right, Down, Left, Up
    blocked = []

    for (x, y) in directions:
        if arena[x, y] == -1 or arena[x, y] == 1:
            blocked.append(1)  # Blocked by a wall or crate
        elif (x, y) in others:
            blocked.append(1)  # Blocked by another agent
        elif explosions[x, y] > 0:
            blocked.append(1)  # Blocked by an imminent explosion
        elif any((bx == x and by == y) for (bx, by), countdown in bombs if countdown <= 1):
            blocked.append(1)  # Blocked by an imminent bomb explosion
        else:
            blocked.append(0)  # The direction is not blocked

    return np.array(blocked)


def k_nearest_coins(ax, ay, coins):
    k = 3  # hyperparameter

    # Calculate distances between the agent and each coin
    distances = [(x, y, np.linalg.norm([ax - x, ay - y])) for (x, y) in coins]

    # Sort coins based on the distance
    distances.sort(key=lambda item: item[2])

    # Get the k nearest coins (we only need the coordinates)
    nearest_coins = [(x, y) for (x, y, _) in distances[:k]]

    return nearest_coins


def direction_to_nearest_coin(ax, ay, blocked, nearest_coin, arena):
    cx, cy = nearest_coin

    # Calculate the priority for moving in each direction based on the relative position of the coin
    directions = [
        ('right', (ax + 1, ay), 0, cx > ax),  # Right
        ('down', (ax, ay + 1), 1, cy > ay),  # Down
        ('left', (ax - 1, ay), 2, cx < ax),  # Left
        ('up', (ax, ay - 1), 3, cy < ay)  # Up
    ]

    # Sort directions by whether they move closer to the coin, considering blocked paths
    directions.sort(key=lambda x: (not x[3], blocked[x[2]]))  # Priority: closer to coin, not blocked

    # Choose the first valid direction that is not blocked
    for direction, (x, y), idx, _ in directions:
        if not blocked[idx] and arena[x, y] == 0:  # Ensure the path is free (not blocked, no walls/crates)
            return idx

    return None  # If all directions are blocked
