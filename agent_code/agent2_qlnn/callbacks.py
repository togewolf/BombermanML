from .dqlnn_model import Agent, state_to_features
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

    self.agent = Agent(self.logger, 'model', 100)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action = self.agent.choose_action(game_state, state_to_features(game_state, *self.agent.get_history()), self.train)

    self.logger.info("Action: " + Action.to_str(action))
    return Action.to_str(action)
