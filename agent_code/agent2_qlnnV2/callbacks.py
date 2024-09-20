import os
import torch
from .dqlnn_model import Agent
from .utils import Action


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

    # Metrics
    self.cumulative_reward = 0
    self.very_bad_agent = 0.0
    self.kills = 0
    self.opponents_eliminated = 0

    model_filename = 'model/model.pt' if self.start_from_snapshot == 0 else 'model/snapshots/model-' + str(self.start_from_snapshot) + '.pt'

    if not os.path.isfile(model_filename):
        self.logger.info("Setting up model from scratch.")
        self.model = Agent(self.logger, gamma=0.9, epsilon=1.0, lr=1e-4, input_dims=52, batch_size=64)

    else:
        self.logger.info("Loading model from saved state.")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            self.logger.info("GPU not found!")
        self.model = torch.load(model_filename, map_location=device)
        self.model.Q_eval.device = device


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
