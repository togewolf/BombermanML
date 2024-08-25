import pickle
from typing import List

import events as e
from .dqlnn_model import state_to_features

import csv  # to store scores

ACTIONS = ['RIGHT', 'DOWN', 'LEFT', 'UP', 'WAIT', 'BOMB']
ACTION_MAP = {action: idx for idx, action in enumerate(ACTIONS)}

# Events
MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'  # toward the closest coin
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'
DROPPED_BOMB_THAT_CAN_DESTROY_CRATE = 'DROPPED_BOMB_THAT_CAN_DESTROY_CRATE'  # reward for each crate
DROPPED_BOMB_WHILE_ENEMY_NEAR = 'DROPPED_BOMB_WHILE_ENEMY_NEAR'  # reward for each enemy
IS_IN_BOMB_EXPLOSION_RADIUS = 'IS_IN_BOMB_EXPLOSION_RADIUS'  # perhaps differentiate between enemy and own bombs
USELESS_BOMB = 'USELESS BOMB'  # no crate or enemy reachable by bomb  # or just punish each dropped bomb and reward usefulness
TRAPPED_SELF = 'TRAPPED_SELF'  # placed bomb in entrance of dead end with length < 4 and went into dead end -> death imminent. A similar feature for locating trapped enemies could be useful too
MOVED_IN_BLOCKED_DIRECTION = 'MOVED_IN_BLOCKED_DIRECTION'
MOVED_BACK_AND_FORTH = 'MOVED_BACK_AND_FORTH'

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    pass

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

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if new_features[7] == 1:
        events.append(IS_IN_BOMB_EXPLOSION_RADIUS)

    # Retrieve the precomputed blocked directions and nearest coin direction index from events_pre
    blocked = old_features[3:7]
    nearest_coin_direction = old_features[8:12]

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
        if blocked[['RIGHT', 'DOWN', 'LEFT', 'UP'].index(self_action)]:
            events.append(MOVED_IN_BLOCKED_DIRECTION)  # should not happen, since this is prohibited in the choose_action function

        # Check if the action was toward the nearest coin
        if nearest_coin_direction[['RIGHT', 'DOWN', 'LEFT', 'UP'].index(self_action)]:
            events.append(MOVED_TOWARD_COIN)

    _, _, _, (ax, ay) = new_game_state['self']
    new_position = (ax, ay)

    # Update the last positions deque
    self.last_positions.append(new_position)

    # Check if the agent is moving back and forth
    if len(self.last_positions) > 2:
        if self.last_positions[0] == self.last_positions[2]:
            events.append(MOVED_BACK_AND_FORTH)


    self.model.store_transition(old_features, ACTION_MAP[self_action],
                                reward_from_events(self, events), new_features, done=False)
    self.model.learn()


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

    self.model.store_transition(state_to_features(last_game_state), ACTION_MAP[last_action],
                                reward_from_events(self, events), None, done=True)

    # write scores to csv file
    score = last_game_state['self'][1]
    file_name = 'scores.csv'
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([score])

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
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
        e.WAITED: -0.1,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -15,
        e.COIN_FOUND: 1,  # Crate destroyed that contains coin
        e.GOT_KILLED: -5,
        e.INVALID_ACTION: -0.5,
        e.OPPONENT_ELIMINATED: 3,
        e.SURVIVED_ROUND: 7.5,
        e.BOMB_DROPPED: -0.5,
        MOVED_TOWARD_COIN: 0.2,
        # MOVED_TOWARD_CRATE: 0.1,  # todo
        MOVED_IN_BLOCKED_DIRECTION: -0.5,
        # DROPPED_BOMB_THAT_CAN_DESTROY_CRATE: 0.2,  # reward per crate that the bomb can reach  # todo
        # DROPPED_BOMB_WHILE_ENEMY_NEAR: 0.4,  # todo
        IS_IN_BOMB_EXPLOSION_RADIUS: -0.5,
        MOVED_BACK_AND_FORTH: -0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
