from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, events_pre

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'  # toward the closest coin
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'
DROPPED_BOMB_THAT_CAN_DESTROY_CRATE = 'DROPPED_BOMB_THAT_CAN_DESTROY_CRATE'  # reward for each crate
DROPPED_BOMB_WHILE_ENEMY_NEAR = 'DROPPED_BOMB_WHILE_ENEMY_NEAR'  # reward for each enemy
IS_IN_BOMB_EXPLOSION_RADIUS = 'IS_IN_BOMB_EXPLOSION_RADIUS'  # perhaps differentiate between enemy and own bombs
USELESS_BOMB = 'USELESS BOMB'  # no crate or enemy reachable by bomb  # or just punish each dropped bomb and reward usefulness
TRAPPED_SELF = 'TRAPPED_SELF'  # placed bomb in entrance of dead end with length < 4 and went into dead end -> death imminent
MOVED_IN_BLOCKED_DIRECTION = 'MOVED_IN_BLOCKED_DIRECTION'


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
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

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if not old_game_state:
        return

    # Retrieve the precomputed blocked directions and nearest coin direction index from events_pre
    blocked = events_pre[0]
    nearest_coin_direction = events_pre[1]

    _, _, _, (ax, ay) = old_game_state['self']

    # Map actions to positions
    action_to_direction = {
        'RIGHT': (ax + 1, ay),
        'DOWN': (ax, ay + 1),
        'LEFT': (ax - 1, ay),
        'UP': (ax, ay - 1)
    }

    if self_action in action_to_direction:
        # Check if the action was in a blocked direction
        if blocked and blocked[['RIGHT', 'DOWN', 'LEFT', 'UP'].index(self_action)]:
            events.append(MOVED_IN_BLOCKED_DIRECTION)

        # Check if the action was toward the nearest coin
        if nearest_coin_direction is not None and ['RIGHT', 'DOWN', 'LEFT', 'UP'][nearest_coin_direction] == self_action:
            events.append(MOVED_TOWARD_COIN)

    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.WAITED: -0.02,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -7.5,
        e.COIN_FOUND: 1,  # Crate destroyed that contains coin
        e.GOT_KILLED: -7.5,
        e.INVALID_ACTION: -1,
        e.OPPONENT_ELIMINATED: 3,
        e.SURVIVED_ROUND: 7.5,
        e.BOMB_DROPPED: -0.2,
        MOVED_TOWARD_COIN: 0.2,  # todo: get those from features (features of old game state)
        # MOVED_TOWARD_CRATE: 0.1,
        MOVED_IN_BLOCKED_DIRECTION: -1,
        # DROPPED_BOMB_THAT_CAN_DESTROY_CRATE: 0.2,  # 0.2 per crate that the bomb can reach  # todo here
        # DROPPED_BOMB_WHILE_ENEMY_NEAR: 0.4,  # todo here
        # IS_IN_BOMB_EXPLOSION_RADIUS: -0.2,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
