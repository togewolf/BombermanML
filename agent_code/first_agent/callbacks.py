import os
import pickle
import random

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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = None  # No pre-trained model at the beginning, no weights for rf
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
    # Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:  # we left it like this from the template, the values seem sensible for now
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state)

    if self.model is None:
        # Default to random action if the model is not trained yet
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # Predict the best action based on the features
    action_index = self.model.predict([features])[0]
    # TODO: limit to valid actions (copy from coin collector or rule-based agent)
    return ACTIONS[action_index]


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

    # field = game_state['field'] (somehow use crate positions)

    # TODO: More features I plan to add:
    # Lag-features: let the model know the features of the previous rounds
    # Whether action 'bomb' is available
    # Position of the closest coin/crate/enemy
    # Bomb timers
    # ...

    # Return the feature vector as a flattened numpy array
    return np.array(features)
