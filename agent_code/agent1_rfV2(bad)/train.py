import joblib
from typing import List
import os
import csv  # to store scores

import events as e
from .rf_agent import state_to_features
from .utils import Action

# Events
IS_REPEATING_ACTIONS = "IS_REPEATING_ACTIONS"
DROPPED_BOMB_THAT_CAN_DESTROY_CRATE_BONUS_FOR_AMOUNT = 'DROPPED_BOMB_THAT_CAN_DESTROY_CRATE_BONUS_FOR_AMOUNT'
DROPPED_BOMB_THAT_CAN_DESTROY_CRATE = "DROPPED_BOMB_THAT_CAN_DESTROY_CRATE"
DROPPED_BOMB_WHILE_ENEMY_NEAR = 'DROPPED_BOMB_WHILE_ENEMY_NEAR'
DROPPED_BOMB_NOT_AT_CROSSING = 'DROPPED_BOMB_NOT_AT_CROSSING'  # see is_at_crossing function for the idea behind this
DID_OPPOSITE_OF_LAST_ACTION = 'DID_OPPOSITE_OF_LAST_ACTION'  # idea: prevent oscillating back and forth
FOLLOWED_DIRECTION_SUGGESTION = 'FOLLOWED_DIRECTION_SUGGESTION'
DID_NOT_FOLLOW_DIRECTION_SUGGESTION = 'DID_NOT_FOLLOW_DIRECTION_SUGGESTION'  # so it does not oscillate to gain reward
DROPPED_BOMB_ON_TRAPPED_ENEMY = 'DROPPED_BOMB_ON_TRAPPED_ENEMY'
DROPPED_BOMB_NEXT_TO_ENEMY = 'DROPPED_BOMB_NEXT_TO_ENEMY'
WAITED_ON_A_BOMB = 'WAITED_ON_A_BOMB'
ENEMY_GOT_KILL = 'ENEMY_GOT_KILL'
WAITED_IN_EXPLOSION_ZONE = 'WAITED_IN_EXPLOSION_ZONE'
IS_NEAR_BORDER = 'IS_NEAR_BORDER'
MOVED_TOWARD_ENEMY_IN_DEAD_END = 'MOVED_TOWARD_ENEMY_IN_DEAD_END'
IGNORED_ENEMY_IN_DEAD_END = 'IGNORED_ENEMY_IN_DEAD_END'  # so it does not oscillate near trapped enemy to gain reward
IS_NEXT_TO_ENEMY = 'IS_NEXT_TO_ENEMY'


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    self.save_frequency = 20  # store a snapshot every n rounds
    self.previous_action = 'WAIT'


def followed_direction(one_hot_direction, action):
    """
    Checks whether one-hot-direction corresponds to action string
    """
    action_to_index = {
        'RIGHT': 0,
        'DOWN': 1,
        'LEFT': 2,
        'UP': 3
    }
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

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if not old_game_state or self.model.old_features is None or self.model.new_features is None:
        return

    old_features = self.model.old_features
    new_features = self.model.new_features

    """    crate_count = old_features[1]

    others = old_game_state['others']
    others = [t[3] for t in others]

    if self_action == 'BOMB':
        if old_features[26] == 0:
            events.append(DROPPED_BOMB_NOT_AT_CROSSING)

        if crate_count > 0:
            events.append(DROPPED_BOMB_THAT_CAN_DESTROY_CRATE)
            # the agent does not have to drop it directly next to the crate, which often is advantageous
            for i in range(crate_count):  # bonus reward for each crate
                events.append(DROPPED_BOMB_THAT_CAN_DESTROY_CRATE_BONUS_FOR_AMOUNT)

        _, _, _, (ax, ay) = old_game_state['self']
        if len(others):
            for (ox, oy) in others:
                if abs(ax - ox) + abs(ay - oy) < 4:
                    events.append(DROPPED_BOMB_WHILE_ENEMY_NEAR)
                    if abs(ax - ox) + abs(ay - oy) < 3:
                        events.append(DROPPED_BOMB_WHILE_ENEMY_NEAR)  # higher reward for dropping it closer

        if old_features[28] == 1:
            events.append(DROPPED_BOMB_NEXT_TO_ENEMY)  # reward for dropping it directly next to the enemy

        if old_features[27] == 1:
            events.append(DROPPED_BOMB_ON_TRAPPED_ENEMY)
    else:
        if old_features[28] == 1:
            events.append(IS_NEXT_TO_ENEMY)  # without dropping a bomb

    if self.previous_action == 'BOMB' and self_action == 'WAIT':
        events.append(WAITED_ON_A_BOMB)
    elif self_action == 'WAIT' and old_features[2]:
        events.append(WAITED_IN_EXPLOSION_ZONE)

    if new_features[3] == 1:
        events.append(IS_REPEATING_ACTIONS)"""

    if followed_direction(old_features[0:4], self_action):
        events.append(FOLLOWED_DIRECTION_SUGGESTION)
    else:
        events.append(DID_NOT_FOLLOW_DIRECTION_SUGGESTION)

    _, _, _, (ax, ay) = new_game_state['self']
    if ax < 3 or ax > 14:
        events.append(IS_NEAR_BORDER)
    if ay < 3 or ay > 14:
        events.append(IS_NEAR_BORDER)  # Twice so it is punished extra for being in corners

    if (self_action == 'UP' and self.previous_action == 'DOWN') or (
            self_action == 'DOWN' and self.previous_action == 'UP') or \
            (self_action == 'LEFT' and self.previous_action == 'RIGHT') or (
            self_action == 'RIGHT' and self.previous_action == 'LEFT'):
        events.append(DID_OPPOSITE_OF_LAST_ACTION)

    if self.model.enemy_got_kill:
        self.logger.info("Enemy got kill!")
        events.append(ENEMY_GOT_KILL)
    """
    if any(old_features[52:56]):
        if followed_direction(old_features[52:56], self_action):
            events.append(MOVED_TOWARD_ENEMY_IN_DEAD_END)
        else:
            events.append(IGNORED_ENEMY_IN_DEAD_END)"""

    self.previous_action = self_action

    reward = reward_from_events(self, events)
    self.cumulative_reward += reward

    # For the plotter
    if e.KILLED_OPPONENT in events:
        self.kills += 1
    if e.OPPONENT_ELIMINATED in events:
        self.opponents_eliminated += 1

    self.model.store_transition(old_features, Action.from_str(self_action),
                                reward, new_features, done=False)
    self.model.learn()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.model.store_transition(state_to_features(self.model, last_game_state, self.logger), Action.from_str(last_action),
                                reward_from_events(self, events), None, done=True)
    self.model.learn()

    self.round_count += 1

    os.makedirs('model/snapshots', exist_ok=True)
    if self.round_count % self.save_frequency == 0:
        model_snapshot_filename = f'model/snapshots/model-round-{self.round_count}.pkl'

        # Save the Random Forest model using joblib
        joblib.dump(self.model, model_snapshot_filename)

    # Gather metrics for the round
    score = last_game_state['self'][1]
    survived = 1 if e.SURVIVED_ROUND in events else 0
    suicide = 1 if e.KILLED_SELF in events else 0

    csv_file_path = 'model/training_metrics.csv'
    file_exists = os.path.isfile(csv_file_path)

    # Save cumulative training reward, kills, and suicides
    with open('model/training_metrics.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ['Round', 'Score', 'Survived', 'CumulativeReward', 'Kills', 'Suicides', 'OpponentsEliminated'])
        writer.writerow(
            [self.round_count, score, survived, self.cumulative_reward, self.kills, suicide, self.opponents_eliminated])

    # Reset tracking variables for the next round
    self.cumulative_reward = 0
    self.kills = 0
    self.opponents_eliminated = 0

    with open('model/scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([score])

    with open('model/survivals.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([survived])

    # Store the model
    joblib.dump(self.model, 'model/model.pkl')


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
        e.COIN_COLLECTED: 8,
        e.KILLED_OPPONENT: 10,
        # e.KILLED_SELF: -5,  # Better to kill oneself than if the enemy gets the kill
        e.GOT_KILLED: -10,
        e.INVALID_ACTION: -1.5,  # Usually happens before death
        e.OPPONENT_ELIMINATED: 4,  # Helps the agent make other agents kill themselves, balances ENEMY_GOT_KILL
        e.BOMB_DROPPED: -0.75,
        e.SURVIVED_ROUND: 10,
        DROPPED_BOMB_THAT_CAN_DESTROY_CRATE_BONUS_FOR_AMOUNT: 0.5,  # Bonus reward per crate that the bomb can reach
        DROPPED_BOMB_WHILE_ENEMY_NEAR: 1,
        DROPPED_BOMB_NEXT_TO_ENEMY: 1.5,
        IS_REPEATING_ACTIONS: -0.5,
        DROPPED_BOMB_THAT_CAN_DESTROY_CRATE: 1.5,
        DROPPED_BOMB_NOT_AT_CROSSING: -0.5,
        DID_OPPOSITE_OF_LAST_ACTION: -0.2,
        FOLLOWED_DIRECTION_SUGGESTION: 0.3,
        DID_NOT_FOLLOW_DIRECTION_SUGGESTION: -0.2,
        WAITED_ON_A_BOMB: -1,  # Sometimes it is okay or necessary to do that, but usually it is best to avoid.
        WAITED_IN_EXPLOSION_ZONE: -0.3,
        IS_NEAR_BORDER: -0.1,
        MOVED_TOWARD_ENEMY_IN_DEAD_END: 0.7,
        IGNORED_ENEMY_IN_DEAD_END: -0.3,
        IS_NEXT_TO_ENEMY: -0.3,
        ENEMY_GOT_KILL: -5  # We do not want it to learn to help other enemies get kills
    }
    # If the overall rewards are too high, the agent has too little incentive to end the game quickly and becomes 'lazy'
    # If the overall rewards are too low/negative, it will learn to die as soon as possible
    balance_reward = 0.1
    reward_sum = sum(game_rewards[event] for event in events if event in game_rewards) + balance_reward
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
