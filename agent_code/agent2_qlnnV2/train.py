import pickle
from typing import List
import os
import csv  # to store scores

import events as e
from .dqlnn_model import state_to_features, last_actions_deque

ACTIONS = ['RIGHT', 'DOWN', 'LEFT', 'UP', 'WAIT', 'BOMB']
ACTION_MAP = {action: idx for idx, action in enumerate(ACTIONS)}

# Events
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'  # todo
DROPPED_BOMB_THAT_CAN_DESTROY_CRATE = 'DROPPED_BOMB_THAT_CAN_DESTROY_CRATE'  # reward for each crate
DROPPED_BOMB_NEXT_TO_CRATE = "DROPPED_BOMB_NEXT_TO_CRATE"
DROPPED_BOMB_WHILE_ENEMY_NEAR = 'DROPPED_BOMB_WHILE_ENEMY_NEAR'  # todo reward for each enemy
IS_IN_BOMB_EXPLOSION_RADIUS = 'IS_IN_BOMB_EXPLOSION_RADIUS'
TRAPPED_SELF = 'TRAPPED_SELF'  # todo placed bomb in entrance of dead end with length < 4 and went into dead end -> death imminent.
# A similar feature for locating trapped enemies could be useful too
# Or just prevent this from occuring at all. If we can calculate this,
# we can just block the action that would lead to this event.
MOVED_IN_BLOCKED_DIRECTION = 'MOVED_IN_BLOCKED_DIRECTION'  # todo (or just prevent, see above)
WAITED_IN_EXPLOSION_RADIUS = "WAITED_IN_EXPLOSION_RADIUS"
MOVED_AWAY_FROM_DANGER = "MOVED_AWAY_FROM_DANGER"  # todo: danger map with higher values closer to bombs, gets reward when danger value decreases, e.g. moves away from bomb


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.save_frequency = 500  # store a snapshot every n rounds


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

    crate_count = old_features[1]
    self.logger.info("Crates reachable: " + str(crate_count))
    if self_action == 'BOMB':
        bomb_position = old_game_state['self'][3]

        # Check adjacent tiles for crates
        x, y = bomb_position
        adjacent_positions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        for pos in adjacent_positions:
            if old_game_state['field'][pos] == 1:  # Assuming 1 represents a crate
                events.append(DROPPED_BOMB_NEXT_TO_CRATE)

        if crate_count > 0:
            for i in range(crate_count):
                events.append(DROPPED_BOMB_THAT_CAN_DESTROY_CRATE)

    if old_features[2] == 1 and self_action == 'WAIT':
        events.append(WAITED_IN_EXPLOSION_RADIUS)

    if new_features[2] == 1:
        events.append(IS_IN_BOMB_EXPLOSION_RADIUS)

    last_actions_deque.append(ACTION_MAP[self_action])

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

    self.model.store_transition(state_to_features(last_game_state), ACTION_MAP[last_action],
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
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.WAITED: -0.05,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -15,
        e.GOT_KILLED: -5,
        e.INVALID_ACTION: -0.5,
        e.OPPONENT_ELIMINATED: 3,
        e.SURVIVED_ROUND: 7.5,
        e.BOMB_DROPPED: -0.5,
        # MOVED_TOWARD_CRATE: 0.1,  # todo
        MOVED_IN_BLOCKED_DIRECTION: -0.5,
        DROPPED_BOMB_THAT_CAN_DESTROY_CRATE: 1,  # reward per crate that the bomb can reach
        # DROPPED_BOMB_WHILE_ENEMY_NEAR: 1,  # todo
        IS_IN_BOMB_EXPLOSION_RADIUS: -0.3,
        WAITED_IN_EXPLOSION_RADIUS: -0.75
    }
    reward_sum = sum(game_rewards[event] for event in events if event in game_rewards)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
