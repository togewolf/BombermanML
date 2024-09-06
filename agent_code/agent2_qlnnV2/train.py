import pickle
from typing import List
import os
import csv  # to store scores

import events as e
from .dqlnn_model import state_to_features

ACTIONS = ['RIGHT', 'DOWN', 'LEFT', 'UP', 'WAIT', 'BOMB']
ACTION_MAP = {action: idx for idx, action in enumerate(ACTIONS)}

# Events
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'
IS_STUCK = "IS_STUCK"
DROPPED_BOMB_THAT_CAN_DESTROY_CRATE = 'DROPPED_BOMB_THAT_CAN_DESTROY_CRATE'  # reward for each crate
DROPPED_BOMB_NEXT_TO_CRATE = "DROPPED_BOMB_NEXT_TO_CRATE"
DROPPED_BOMB_WHILE_ENEMY_NEAR = 'DROPPED_BOMB_WHILE_ENEMY_NEAR'
IS_IN_BOMB_EXPLOSION_RADIUS = 'IS_IN_BOMB_EXPLOSION_RADIUS'
DROPPED_BOMB_NOT_AT_CROSSING = 'DROPPED_BOMB_NOT_AT_CROSSING'  # see is_at_crossing function for the idea behind this
DID_OPPOSITE_OF_LAST_ACTION = 'DID_OPPOSITE_OF_LAST_ACTION'  # idea: prevent oscillating back and forth
MOVED_TOWARD_ENEMY_IN_DEAD_END = 'MOVED_TOWARD_ENEMY_IN_DEAD_END'  # see direction_to_enemy_in_dead_end
DROPPED_BOMB_ON_TRAPPED_ENEMY = 'DROPPED_BOMB_ON_TRAPPED_ENEMY'  # enemy dies for sure, so high reward


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.save_frequency = 500  # store a snapshot every n rounds


previous_action = 'WAIT'


def followed_direction(one_hot_direction, action):
    action_to_index = {
        'RIGHT': 0,
        'DOWN': 1,
        'LEFT': 2,
        'UP': 3
    }
    # Check if self_action exists in the mapping and corresponds to the one-hot array
    if action in action_to_index:
        action_index = action_to_index[action]
        return one_hot_direction[action_index] == 1
    else:
        return False


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

    old_features = state_to_features(old_game_state, self.logger)
    new_features = state_to_features(new_game_state, self.logger)

    crate_count = old_features[1]
    # self.logger.info("Crates reachable: " + str(crate_count))
    if self_action == 'BOMB':
        bomb_position = old_game_state['self'][3]

        # Check adjacent tiles for crates
        x, y = bomb_position
        adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        crate_adjacent = False
        for pos in adjacent_positions:
            if old_game_state['field'][pos] == 1:
                crate_adjacent = True
                break
        if crate_adjacent:
            events.append(DROPPED_BOMB_NEXT_TO_CRATE)  # we only want this event one time

        if crate_count > 0:
            for i in range(crate_count):
                events.append(DROPPED_BOMB_THAT_CAN_DESTROY_CRATE)

        others = old_game_state['others']
        others = [t[3] for t in others]
        _, _, _, (ax, ay) = old_game_state['self']
        if len(others):
            for (ox, oy) in others:
                if abs(ax - ox) + abs(ay - oy) < 4:
                    events.append(DROPPED_BOMB_WHILE_ENEMY_NEAR)
                    if abs(ax - ox) + abs(ay - oy) < 3:
                        events.append(DROPPED_BOMB_WHILE_ENEMY_NEAR)
                        if abs(ax - ox) + abs(ay - oy) < 2:
                            events.append(
                                DROPPED_BOMB_WHILE_ENEMY_NEAR)  # higher reward for dropping bombs closer to enemies.

        if old_features[40] == 0:
            events.append(DROPPED_BOMB_NOT_AT_CROSSING)

        if old_features[52] == 1:
            events.append(DROPPED_BOMB_ON_TRAPPED_ENEMY)

    if new_features[2] == 1:
        events.append(IS_IN_BOMB_EXPLOSION_RADIUS)

    if new_features[3] == 1:
        events.append(IS_STUCK)

    if followed_direction(old_features[8:12], self_action):
        events.append(MOVED_TOWARD_CRATE)

    if followed_direction(old_features[47:51], self_action):
        events.append(MOVED_TOWARD_ENEMY_IN_DEAD_END)
    self.logger.info("Dead end features: " + str(new_features[47:51]))

    global previous_action
    if (self_action == 'UP' and previous_action == 'DOWN') or (self_action == 'DOWN' and previous_action == 'UP') or \
            (self_action == 'LEFT' and previous_action == 'RIGHT') or (
            self_action == 'RIGHT' and previous_action == 'LEFT'):
        events.append(DID_OPPOSITE_OF_LAST_ACTION)
    self.logger.info("Previous action: " + previous_action)
    self.logger.info("Current action: " + self_action)
    previous_action = self_action

    crates_left = old_features[26]
    if crates_left:
        direction_to_crate = old_features[8:12]
        # self.logger.info("Direction and distance to nearest crate: " + str(direction_to_crate))
        if self_action == ACTIONS[next((i for i, x in enumerate(direction_to_crate) if x > 0), 5)]:
            events.append(MOVED_TOWARD_CRATE)

    self.logger.info("Disallowed actions: " + str(new_features[20:26]))

    self.model.store_transition(old_features, ACTION_MAP[self_action],
                                reward_from_events(self, events), new_features, done=False)
    self.model.learn()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.model.store_transition(state_to_features(last_game_state, self.logger), ACTION_MAP[last_action],
                                reward_from_events(self, events), None, done=True)
    self.model.learn(end_epoch=True)

    # save snapshot of the model
    os.makedirs('model/snapshots', exist_ok=True)
    if self.model.epoch % self.save_frequency == 0:
        with open('model/snapshots/model-' + str(self.model.epoch) + '.pt', 'wb') as file:
            pickle.dump(self.model, file)

    # write scores to csv file
    score = last_game_state['self'][1]

    with open('model/scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([score])

    # Store the model
    with open('model/model.pt', 'wb') as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get to en/discourage certain behavior.
    """
    game_rewards = {
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.WAITED: -0.3,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -5,
        e.GOT_KILLED: -5,
        e.INVALID_ACTION: -1,
        e.OPPONENT_ELIMINATED: 5,
        e.BOMB_DROPPED: -0.5,
        e.SURVIVED_ROUND: 5,
        DROPPED_BOMB_THAT_CAN_DESTROY_CRATE: 0.5,  # reward per crate that the bomb can reach
        DROPPED_BOMB_WHILE_ENEMY_NEAR: 2,
        IS_STUCK: -0.75,
        MOVED_TOWARD_CRATE: 0.2,
        DROPPED_BOMB_NEXT_TO_CRATE: 1.5,
        DROPPED_BOMB_NOT_AT_CROSSING: -1,
        DID_OPPOSITE_OF_LAST_ACTION: -0.3,
        MOVED_TOWARD_ENEMY_IN_DEAD_END: 1,
        DROPPED_BOMB_ON_TRAPPED_ENEMY: 10  # enemy dies for sure
    }
    reward_sum = sum(game_rewards[event] for event in events if event in game_rewards)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
