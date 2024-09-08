import pickle
from typing import List
import os
import csv

import events as e
from .dqlnn_model import state_to_features, nearest_objects
from .utils import Action

# Events
MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'  # toward the closest coin
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
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
    self.save_frequency = 500 # store a snapshot every n rounds

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

    #if new_features[7] == 1:
    #    events.append(IS_IN_BOMB_EXPLOSION_RADIUS)

    # Retrieve the precomputed blocked directions and nearest coin direction index from events_pre
    #blocked = old_features[3:7]
    #nearest_coin_direction = old_features[8:12]

    _, _, _, (x1, y1) = old_game_state['self']
    _, _, _, (x2, y2) = new_game_state['self']
    new_position = (x2, y2)

    old_coin = nearest_objects(old_features['conv_features'][0], old_game_state['coins'])
    new_coin = nearest_objects(new_features['conv_features'][0], new_game_state['coins'])

    if old_features['conv_features'][0][old_coin[0][0], old_coin[0][1]] > new_features['conv_features'][0][new_coin[0][0], new_coin[0][1]]:
       events.append(MOVED_TOWARD_COIN)
    else:
       events.append(MOVED_AWAY_FROM_COIN)

    # Update the last positions deque
    self.last_positions.append(new_position)

    # Check if the agent is moving back and forth
    if len(self.last_positions) > 2:
        if self.last_positions[0] == self.last_positions[2]:
            events.append(MOVED_BACK_AND_FORTH)


    self.model.store_transition(old_features, new_features, Action.from_str(self_action),
                                reward_from_events(self, events), done=False)
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

    self.model.store_transition(state_to_features(last_game_state), None, Action.from_str(last_action),
                                reward_from_events(self, events), done=True)
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
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_UP: 0,#-0.01,
        e.MOVED_DOWN: 0,#-0.01,
        e.MOVED_LEFT: 0,#-0.01,
        e.MOVED_RIGHT: 0,#-0.01,
        e.WAITED: -0.3,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 0,#5,
        e.KILLED_SELF: -5,#-15,
        e.COIN_FOUND: 0,#1,  # Crate destroyed that contains coin
        e.GOT_KILLED: -5,
        e.INVALID_ACTION: -0.3,#-0.5,
        e.OPPONENT_ELIMINATED: 0,#3,
        e.SURVIVED_ROUND: 0,#7.5,
        e.BOMB_DROPPED: 0,#-0.5,
        MOVED_TOWARD_COIN: 0.3,
        MOVED_AWAY_FROM_COIN: -0.3,
        # MOVED_TOWARD_CRATE: 0.1,  # todo
        MOVED_IN_BLOCKED_DIRECTION: 0,#-0.5,
        # DROPPED_BOMB_THAT_CAN_DESTROY_CRATE: 0.2,  # reward per crate that the bomb can reach  # todo
        # DROPPED_BOMB_WHILE_ENEMY_NEAR: 0.4,  # todo
        IS_IN_BOMB_EXPLOSION_RADIUS: 0,#-0.5,
        MOVED_BACK_AND_FORTH: 0,#-0.75
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
