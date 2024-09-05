import os
import pickle
from collections import deque
from .dqlnn_model import Agent, state_to_features

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

    # load specific snapshot of model; if set to zero, the default (model/model.pt) will be loaded
    # note that training from a snapshot n will save new snapshots as n+10, n+20 etc., thus overriding some of the previous snapshots
    self.start_from_snapshot = 0

    model_filename = 'model/model.pt' if self.start_from_snapshot == 0 else 'model/snapshots/model-' + str(self.start_from_snapshot) + '.pt'

    if not os.path.isfile(model_filename):
        self.logger.info("Setting up model from scratch.")
        self.model = Agent(self.logger)

    else:
        self.logger.info("Loading model from saved state.")
        with open(model_filename, 'rb') as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action = self.model.choose_action(state_to_features(game_state), self.train)
    self.logger.info("Action: " + ACTIONS[action])
    return ACTIONS[action]
