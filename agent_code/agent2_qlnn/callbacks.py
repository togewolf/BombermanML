import os
import pickle
from collections import deque
from .dqlnn_model import Agent

ACTIONS = ['RIGHT', 'DOWN', 'LEFT', 'UP', 'WAIT', 'BOMB']


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
    self.last_positions = deque(maxlen=3)
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = Agent(self.logger, gamma=0.9, epsilon=1.0, lr=1e-3, input_dims=28, batch_size=64)

    elif self.train:
        self.logger.info("Loading model from saved state.")
        self.model = Agent(self.logger, gamma=0.9, epsilon=1.0, lr=5e-4, input_dims=28, batch_size=64)

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
    action = self.model.choose_action(game_state, self.train)
    self.logger.info(ACTIONS[action])
    return ACTIONS[action]
