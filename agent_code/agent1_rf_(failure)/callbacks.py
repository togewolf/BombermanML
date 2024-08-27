import os
import pickle
import random
from collections import deque

from sklearn.ensemble import RandomForestClassifier
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if not os.path.isfile("model.pt"):
        self.logger.info("Setting up model from scratch.")

        self.model = RandomForestClassifier(n_estimators=50)  # Use 10 trees in the forest

        dummy_X = np.zeros((1, 6))  # entire field without walls plus the direction to the nearest coin
        dummy_y = [0]  # A single dummy action index
        self.model.fit(dummy_X, dummy_y)


    elif self.train:
        self.logger.info("Loading model form saved state.")
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)


    else:
        self.logger.info("Loading model from saved state.")
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    valid_actions_list = valid_actions(game_state)
    num_valid_actions = len(valid_actions_list)
    # Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:  # we left it like this from the template, the values seem sensible for now
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        random_action = np.random.choice(valid_actions_list,
                                         p=[0.8 / (num_valid_actions - 2)] * (num_valid_actions - 2) + [0.1, 0.1])
        # print("Invalid prediction. Random action: " + random_action)
        return random_action  # so that the bomb probability does not go up when there are less valid actions

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state)

    if self.model is None:
        # Default to random action if the model is not trained yet
        return np.random.choice(valid_actions_list,
                                p=[0.8 / (num_valid_actions - 2)] * (num_valid_actions - 2) + [0.1, 0.1])

    # Predict the best action based on the features
    action_index = self.model.predict([features])[0]
    predicted_action = ACTIONS[action_index]
    # print(predicted_action)

    # Ensure the action is valid
    if predicted_action in valid_actions_list:
        # print("predicted action: " + predicted_action)
        return predicted_action
    else:
        # Choose a random valid action if the predicted action is not valid
        random_action = np.random.choice(valid_actions_list,
                                         p=[0.8 / (num_valid_actions - 2)] * (num_valid_actions - 2) + [0.1, 0.1])
        # print("Invalid prediction. Random action: " + random_action)
        return random_action


temp = 'WAIT'


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

        # Example features (you need to expand this based on the actual game environment):
    features = []

    # Extract agent position
    agent_x, agent_y = game_state['self'][3]
    features.append(agent_x)
    features.append(agent_y)

    # Extract the positions of bombs
    bomb_positions = [position for position, _ in game_state['bombs']]
    features.extend([coord for pos in bomb_positions for coord in pos])

    # Extract positions of coins
    coin_positions = game_state['coins']
    features.extend([coord for pos in coin_positions for coord in pos])

    # Extract opponent positions
    opponent_positions = [pos for _, _, _, pos in game_state['others']]
    features.extend([coord for pos in opponent_positions for coord in pos])

    # what I am doing here is probably abysmal, but I just want to see if it works
    field = game_state['field']

    field[agent_x, agent_y] = 3  # mark position of agent in grid with 3

    for coin in coin_positions:  # mark coins with twos
        x, y = coin
        field[x, y] = 2

    for bomb in bomb_positions:  # mark bombs with negative twos
        x, y = bomb
        field[x, y] = -2

    for op in opponent_positions:
        x, y = op
        field[x, y] = 4

    feature_vector = field.flatten()
    feature_vector = feature_vector[feature_vector != -1]  # remove unchanging walls

    # Add the action that brings the agent closer to the closest coin

    def find_closest_coin_direction(agent_x, agent_y, coins, field):
        """ Find the direction to the closest coin considering obstacles. """
        directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        deltas = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, -1), 'DOWN': (0, 1)}

        def is_valid(x, y):
            """ Check if a position is within bounds and not an obstacle. """
            return 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] != -1

        def bfs(start, goal):
            """ Perform BFS to find the shortest path from start to goal. """
            queue = deque([start])
            came_from = {start: None}
            while queue:
                current = queue.popleft()
                if current == goal:
                    break
                for direction in directions:
                    next_pos = (current[0] + deltas[direction][0], current[1] + deltas[direction][1])
                    if is_valid(next_pos[0], next_pos[1]) and next_pos not in came_from:
                        queue.append(next_pos)
                        came_from[next_pos] = current
            return came_from

        min_distance = float('inf')
        best_direction = 'WAIT'
        for coin in coins:
            came_from = bfs((agent_x, agent_y), coin)
            if coin in came_from:
                # Calculate the path length (number of steps)
                path_length = 0
                current = coin
                while came_from[current] is not None:
                    current = came_from[current]
                    path_length += 1
                if path_length < min_distance:
                    min_distance = path_length
                    best_direction = get_best_direction(came_from, (agent_x, agent_y), coin)

        temp = best_direction
        return best_direction

    def get_best_direction(came_from, start, goal):
        """ Determine the best direction to move towards the goal. """
        directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        deltas = {'LEFT': (-1, 0), 'RIGHT': (1, 0), 'UP': (0, -1), 'DOWN': (0, 1)}
        for direction in directions:
            next_pos = (start[0] + deltas[direction][0], start[1] + deltas[direction][1])
            if next_pos == goal:
                return direction
        return 'WAIT'

    # the following just takes way too long:
    #closest_coin_direction = find_closest_coin_direction(agent_x, agent_y, coin_positions, field)
    #direction_features = {'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'WAIT': 4}
    #feature_vector = np.append(feature_vector, direction_features.get(closest_coin_direction, 4))
    # print(direction_features.get(closest_coin_direction, 4))

    # TODO: More features I plan to add:
    # Lag-features: let the model know the features of the previous rounds
    # Whether action 'bomb' is available
    # Position of the closest coin/crate/enemy
    # Bomb timers
    # ...

    # Return the feature vector as a flattened numpy array
    return np.array(feature_vector)


def valid_actions(game_state: dict):
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # valid_actions.append('BOMB') causes divide by zero when agent dies. For coin collection it does not need to anyway.
    return valid_actions

