import os
from .rf_agent import Agent
from .utils import Action
import joblib


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
    # Load specific snapshot of model; if set to zero, the default (model/model.pt) will be loaded
    # Note that training from a snapshot n will save new snapshots as n+10, n+20 etc., thus overriding some previous snapshots
    self.start_from_snapshot = 0
    self.round_count = 0

    # Metrics
    self.cumulative_reward = 0
    self.very_bad_agent = 0.0
    self.kills = 0
    self.opponents_eliminated = 0

    model_filename = 'model/model_rf.pkl' if self.start_from_snapshot == 0 else 'model/snapshots/model-' + str(self.start_from_snapshot) + '.pkl'

    if not os.path.isfile(model_filename):
        self.logger.info("Setting up model from scratch.")
        self.model = Agent(self.logger, n_estimators=100, input_dims=60, batch_size=64)

    else:
        self.logger.info("Loading model from saved state.")
        self.model = joblib.load(model_filename)
        self.model.logger = self.logger


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action = self.model.choose_action(game_state, self.train)
    action_string = Action.to_str(action)
    self.logger.info("Action: " + action_string)
    return action_string
