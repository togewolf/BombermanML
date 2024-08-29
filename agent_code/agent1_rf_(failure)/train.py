from collections import namedtuple, deque

import pickle
from typing import List
from sklearn.ensemble import RandomForestClassifier

import events as e
from .callbacks import state_to_features, temp

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_MAP = {action: idx for idx, action in enumerate(ACTIONS)}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Our own added events:
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    # Example: Set up an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if self_action == temp:
        events.append(MOVED_TOWARD_COIN)
        # events.append(PLACEHOLDER_EVENT)
        # TODO: add the following events and reward them:
        # ...

        # state_to_features is defined in callbacks.py
        # Append the transition to the deque
        self.transitions.append(
            Transition(
                state_to_features(old_game_state),
                ACTION_MAP[self_action],
                state_to_features(new_game_state),
                reward_from_events(self, events)
            )
        )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param events:
    :param last_action:
    :param last_game_state:
    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            ACTION_MAP[last_action],
            None,  # No next state at the end of the round
            reward_from_events(self, events)
        )
    )
    # Train the model using the collected transitions
    if len(self.transitions) > 0:
        X = []
        y = []
        for transition in self.transitions:
            if transition.state is not None:
                X.append(transition.state)
                y.append(transition.action)

        # Train the model on the features and actions
        self.model.fit(X, y)

    # Store the model
    with open("model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.MOVED_LEFT: -.05,
        e.MOVED_RIGHT: -.05,
        e.MOVED_UP: -.05,
        e.MOVED_DOWN: -.05,
        e.WAITED: -.1,
        e.INVALID_ACTION: -1,
        e.MOVED_TOWARD_COIN: 0.1
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
        # TODO: complete the rewards
    }
    reward_sum = sum(game_rewards.get(event, 0) for event in
                     events)  # it is good python to use sum instead of for loop as in template
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
