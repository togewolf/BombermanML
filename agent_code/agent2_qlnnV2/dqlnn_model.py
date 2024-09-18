import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import numpy as np
import torch.nn.functional as f
from collections import deque
import random
from heapq import heapify, heappop, heappush

from .utils import Action, Direction


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, l1_dims, l2_dims, l3_dims, l4_dims, l5_dims, dropout_rate=0.1):
        super(DeepQNetwork, self).__init__()
        # torch.autograd.set_detect_anomaly(True)  # remove when it works
        self.input_dims = input_dims
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.l3_dims = l3_dims
        self.l4_dims = l4_dims
        self.l5_dims = l5_dims

        self.l1 = nn.Linear(self.input_dims, self.l1_dims)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.l2 = nn.Linear(self.l1_dims, self.l2_dims)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.l3 = nn.Linear(self.l2_dims, self.l3_dims)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.l4 = nn.Linear(self.l3_dims, self.l4_dims)
        self.dropout4 = nn.Dropout(p=dropout_rate)
        self.l5 = nn.Linear(self.l4_dims, self.l5_dims)
        self.dropout5 = nn.Dropout(p=dropout_rate)
        self.lo = nn.Linear(self.l5_dims, 6)  # There are always six possible actions

        # Optimizer (see https://pytorch.org/docs/stable/optim.html#algorithms)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # LR Scheduler (see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
        self.scheduler = scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        # self.scheduler = scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=50)

        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("GPU not found!")
        self.to(self.device)

    def forward(self, state, train):
        x = f.relu(self.l1(state))
        if train:
            x = self.dropout1(x)
        x = f.relu(self.l2(x))
        if train:
            x = self.dropout2(x)
        x = f.relu(self.l3(x))
        if train:
            x = self.dropout3(x)
        x = f.relu(self.l4(x))
        if train:
            x = self.dropout4(x)
        x = f.relu(self.l5(x))
        if train:
            x = self.dropout5(x)
        actions = self.lo(x)
        # actions_prob = torch.sigmoid_(actions)  # todo
        # softmax/sigmoid causes really weird errors here
        return actions


class Agent:
    def __init__(self, logger, gamma, epsilon, lr, input_dims, batch_size, max_mem_size=100000, eps_end=0.03,
                 eps_dec=1e-2):
        self.logger = logger
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size

        # feature-related
        self.total_coins = 0
        self.previous_scores = {}
        self.bomb_owners = {}
        self.last_actions_deque = deque(maxlen=4)
        self.coordinate_history = deque([], maxlen=20)

        # network dimensions
        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, l1_dims=2 ** 9, l2_dims=2 ** 9,
                                   l3_dims=2 ** 8, l4_dims=2 ** 7, l5_dims=2 ** 7)

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.mem_cntr = 0
        self.iteration = 0
        self.epoch = 0

    def store_transition(self, features, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = features
        if state_ is not None:
            self.new_state_memory[index] = state_
        else:
            self.new_state_memory[index] = features
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, game_state, train):
        features = state_to_features(self, game_state, self.logger)
        disallowed = features[8:14]  # Result of function of immortality

        if np.random.random() > self.epsilon or not train:
            state = torch.tensor([features], dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state, train).squeeze()

            for i in range(6):
                if disallowed[i] == 1:
                    actions[i] = -9999

            action = torch.argmax(actions).item()

            # action_probabilities = actions.clone().detach().softmax(dim=1).squeeze()  # todo
            # action = int(torch.multinomial(action_probabilities, 1).item())

        else:
            p = [.20, .20, .20, .20, .10, .10]

            for i in range(6):
                if disallowed[i] == 1:
                    p[i] = 0

            total_prob = np.sum(p)
            if total_prob > 0:
                p /= total_prob
            else:
                p = [.20, .20, .20, .20, .10, .10]

            action = random.choices(range(6), weights=p, k=1)[0]

        # self.logger.info("Eps = " + str(self.epsilon) + ")")
        self.last_actions_deque.append(action)
        _, _, _, (ax, ay) = game_state['self']
        self.coordinate_history.append((ax, ay))
        return action

    def learn(self, end_epoch=False):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch, 1)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch, 1)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()

        self.Q_eval.optimizer.step()

        self.iteration += 1
        if end_epoch:
            self.epoch += 1

            # self.Q_eval.scheduler.step()
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        self.logger.info("LR = " + str(self.Q_eval.scheduler.get_last_lr()))


# Direction Mapping: invert direction by multiplying with -1
dUP = -2
dLEFT = -1
dNONE = 0
dRIGHT = 1
dDOWN = 2
INF = 999  # 999 represents infinity


def state_to_features(self, game_state: dict, logger) -> np.array:
    """
    Converts the game state to a feature vector.
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    if game_state['step'] == 1:
        self.total_coins = 0
        self.bomb_owners = {}
        self.last_actions_deque.clear()
        self.coordinate_history.clear()

    features = []

    # useful stuff for feature calculating functions, to avoid inefficient recalculations
    _, self_score, bomb_available, (ax, ay) = game_state['self']
    field = game_state['field']
    coins = game_state['coins']  # [(x, y)]
    crates = np.argwhere(field == 1)
    crates = [tuple(pos) for pos in crates]  # so they are in the same format as the coins
    bombs = game_state['bombs']  # [(x, y), countdown]
    bombs = [b[0] for b in bombs]
    others_full = game_state['others']
    others = [t[3] for t in others_full]
    dist, grad, crate_map = get_distance_map(ax, ay, field)
    danger_map = get_danger_map(game_state)
    in_danger = 1 if danger_map[ax, ay] > 0 else 1
    dead_end_map, dead_end_list = get_dead_end_map(field, others_full, bombs)

    dist_t, grad_t, _ = get_distance_map_with_temporaries(ax, ay, field, others, bombs)

    nearest_safe_t = get_nearest_safe_tile(ax, ay, danger_map, others, bombs, dist_t, grad_t, dist, True, field, False,
                                           logger)
    # the t marker means that temporary objects such as bombs and agents are also considered here

    direction_to_enemy_in_dead_end = get_direction_to_enemy_in_dead_end(self, ax, ay, dead_end_list, dist, grad,
                                                                        dead_end_map, logger)

    blocked = function_of_immortality(self, game_state, danger_map, others, bombs, dist_t, grad_t, dist, grad,
                                      nearest_safe_t, dead_end_list, direction_to_enemy_in_dead_end[4], logger)

    enemies_distances_and_directions = get_enemies_distances_and_directions(dist, others, grad)

    collected_coins_all = coins_collected(self, others_full, self_score)

    bomb_owners = track_bombs(self, bombs, others_full, danger_map)

    suggestion = direction_suggestion(ax, ay, danger_map, bombs, others, field, coins, crates, others, dist_t, grad_t,
                                      dist, grad, direction_to_enemy_in_dead_end, collected_coins_all, logger)

    # logger.info("In danger: " + str(in_danger))
    # logger.info(danger_map)
    # logger.info("Distances to nearest safe tiles: " + str(nearest_safe))
    # logger.info("Distance map: ")
    # logger.info(dist)
    # logger.info("Enemies distances: " +str(get_enemies_distances_and_directions(dist, others, grad)))
    # logger.info("is at crossing: " + str(is_at_crossing(ax, ay)))
    # logger.info(field)
    # logger.info(dead_end_map)
    # logger.info(nearest_safe)
    # logger.info("Direction suggestion: " + str(suggestion))
    # logger.info("bomb-owners: " + str(bomb_owners))
    # logger.info("Coins collected so far: " + str(collected_coins_all))

    # features
    features.append(int(bomb_available))  # feat 0
    features.append(crates_reachable(ax, ay, field))  # feat 1
    features.append(in_danger)  # feat 2
    features.append(is_repeating_actions(self))  # feat 3
    features.extend(suggestion)  # feat 4-7
    features.extend(blocked)  # 8-13
    features.extend(enemies_distances_and_directions)  # 14-25
    features.append(is_at_crossing(ax, ay))  # 26
    features.append(direction_to_enemy_in_dead_end[4])  # 27 whether it can bomb trap the enemy
    features.append(int(is_next_to_enemy(ax, ay, others)))  # 28
    features.extend(neighboring_explosions_or_coins(ax, ay, danger_map, coins))  # 29-33
    features.extend(nearest_coin_dist(dist, coins, grad))  # 34-37
    features.extend(get_enemies_relative_positions(ax, ay, others))  # 38-49
    features.extend([np.sqrt(ax), np.sqrt(ay)])  # 50-51  Idea: learn to avoid borders and especially the corners
    # features.extend(neighboring_crate_count(ax, ay, field))
    # features.extend(do_not_get_surrounded(others, ax, ay, radius=5))

    return features


def coins_collected(self, others, self_score):  # others is the full version [(str, int-score, bool, (int,int))]
    """
    Calculate the total number of coins collected by all agents.
    :param others: List of opponent agents in the format (name, score, bomb, (x, y)).
    :param self_score: Current score of the agent.
    :param self: the instance of the agent class, in this context used for storing previous scores and total coins
    :return: Total number of coins collected by all agents so far.
    """
    # Create a list to include our agent along with the others
    all_agents = others + [("self", self_score, True, (0, 0))]  # Include our agent as "self"

    for agent_name, current_score, _, _ in all_agents:
        if agent_name in self.previous_scores:
            score_diff = current_score - self.previous_scores[agent_name]

            if score_diff == 1:
                # Agent collected a coin
                self.total_coins += 1
            elif score_diff > 1:
                # If the score increased by more than 1, add score_diff % 5 to exclude kills
                # An agent can kill multiple other agents in the same step and collect a coin
                self.total_coins += score_diff % 5

        self.previous_scores[agent_name] = current_score

    return self.total_coins


def direction_suggestion(ax, ay, danger_map, bombs, enemies, field, coins, crates, others, dist_t, grad_t, dist, grad,
                         dteide, collected_coins_all, logger):
    """
    Selects a goal tile: trappable enemy, coin, escape direction, crate, enemy in order of decreasing importance.
    The agent is rewarded for following the directions, but not forced.
    Returns a one-hot direction vector and a bool, whether it can drop a bomb on a trapped enemy,
    which is used as a separate feature.
    """
    suggestion = [0] * 4

    if any(dteide):  # there is a trappable enemy nearby
        logger.info("Current goal: trappable enemy")
        return dteide[:4]

    dist_c = 999
    if len(coins):
        nearest_coin = nearest_objects(dist, coins, 1)
        dist_c = dist[nearest_coin[0]]

    if dist_c < 30:
        logger.info("Current goal: coin")
        return get_k_nearest_objects(dist_t, coins, grad_t)

    safe_direction = safer_direction(ax, ay, danger_map, bombs, enemies, field)
    if any(safe_direction):
        logger.info("Current goal: safest direction: " + str(safe_direction))
        return safe_direction

    coins_left_in_crates = True
    if collected_coins_all == 9 or collected_coins_all == 50:  # classic or loot-crate or coin-heaven
        coins_left_in_crates = False

    if len(crates) and coins_left_in_crates:
        logger.info("Current goal: crate")
        # todo: if own bomb exists, favor crates that are targeted by the bomb -> be close when a coin is found
        return k_nearest_crates_feature(field, dist_t, crates, grad_t)

    if len(others):
        logger.info("Current goal: enemy")
        # If is at border of map, suggest direction away from border, because it is easy to be killed there
        if ax < 3 or ax > 14 or ay < 3 or ay > 14:
            logger.info("Current goal: move away from border")
            direction_to_center = direction_to_object((8, 8), grad)
            direction_one_hot = [
                int(direction_to_center == dRIGHT),
                int(direction_to_center == dDOWN),
                int(direction_to_center == dLEFT),
                int(direction_to_center == dUP)
            ]
            return direction_one_hot
        return get_k_nearest_objects(dist, others, grad)

    # might as well blow up the last empty crates
    if len(crates):
        logger.info("Current goal: crate")
        return k_nearest_crates_feature(field, dist_t, crates, grad_t)

    return suggestion


def get_k_nearest_objects(dist, objects, grad, k=1):
    """
    :returns the direction to the nearest object(s) as one-hot
    """

    if not len(objects):  # do not waste time when there are no objects
        return [0, 0, 0, 0]

    nearest = nearest_objects(dist, objects, k)
    features = []
    for o in nearest:
        direction = direction_to_object(o, grad)
        direction_one_hot = [
            int(direction == dRIGHT),
            int(direction == dDOWN),
            int(direction == dLEFT),
            int(direction == dUP),
        ]
        features.extend(direction_one_hot)
    return features


def nearest_coin_dist(dist, coins, grad):
    """
    :returns the direction to the nearest coin as one-hot scaled by the sqrt of its distance.
    """

    if not len(coins):  # do not waste time when there are no objects
        return [0, 0, 0, 0]

    nearest = nearest_objects(dist, coins, 1)[0]
    features = []

    direction = direction_to_object(nearest, grad)
    distance = np.sqrt(dist[nearest])

    direction_one_hot = [
        int(direction == dRIGHT) * distance,
        int(direction == dDOWN) * distance,
        int(direction == dLEFT) * distance,
        int(direction == dUP) * distance,
    ]
    features.extend(direction_one_hot)
    return features


def k_nearest_crates_feature(field, dist, crates, grad, k=1):
    """
    :returns the direction to the nearest crate as one-hot.
    Useful for finding crates when there are few left.
    It is not the direction to the crate tile, but to the closest tile next to the crate.
    """
    if not len(crates):
        return [0, 0, 0, 0]

    # convert crates into empty locations next to crates
    locations = set()  # so that the same location is not added twice
    for cx, cy in crates:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if field[cx + dx, cy + dy] == 0:
                locations.add((cx + dx, cy + dy))
    locations = list(locations)

    nearest_crates = nearest_objects(dist, locations, k)
    features = []
    for crate in nearest_crates:
        direction = direction_to_object(crate, grad)
        direction_one_hot = [
            int(direction == dRIGHT),
            int(direction == dDOWN),
            int(direction == dLEFT),
            int(direction == dUP)
        ]
        features.extend(direction_one_hot)

    return features


def get_distance_map(ax, ay, arena):
    """
    :returns distance to every position and corresponding gradient calculated with Dijkstra
    """
    dist = np.full_like(arena, INF)
    grad = np.full_like(arena, dNONE)
    scanned = np.full_like(arena, False)

    crates = np.full_like(arena, INF)

    dist[ax, ay] = 0
    crates[ax, ay] = 0

    pq = [(0, (ax, ay))]
    heapify(pq)

    while pq:
        d, (x, y) = heappop(pq)

        if scanned[x, y]:
            continue
        scanned[x, y] = True

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if arena[x + dx, y + dy] == 0:  # changed != 1 to == 0 so that crates block the way too
                # relax node
                if d + 1 < dist[x + dx, y + dy]:
                    dist[x + dx, y + dy] = d + 1
                    grad[x + dx, y + dy] = -dx - 2 * dy
                    crates[x + dx, y + dy] = crates[x, y] + arena[x + dx, y + dy]
                    heappush(pq, (d + 1, (x + dx, y + dy)))

    return dist, grad, crates


def get_distance_map_with_temporaries(ax, ay, arena, others, bombs):
    """
    :returns distance to every position and corresponding gradient calculated with Dijkstra
    This one additionally considers temporary objects (enemies and bombs).
    """
    arena = np.copy(arena)  # avoid accidentally changing the original
    dist = np.full_like(arena, INF)
    grad = np.full_like(arena, dNONE)
    scanned = np.full_like(arena, False)

    crates = np.full_like(arena, INF)

    dist[ax, ay] = 0
    crates[ax, ay] = 0

    pq = [(0, (ax, ay))]
    heapify(pq)

    for o in others:
        arena[o] = 1

    for b in bombs:
        arena[b] = 1

    while pq:
        d, (x, y) = heappop(pq)

        if scanned[x, y]:
            continue
        scanned[x, y] = True

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if arena[x + dx, y + dy] == 0:  # changed != 1 to == 0 so that crates block the way too
                # relax node
                if d + 1 < dist[x + dx, y + dy]:
                    dist[x + dx, y + dy] = d + 1
                    grad[x + dx, y + dy] = -dx - 2 * dy
                    crates[x + dx, y + dy] = crates[x, y] + arena[x + dx, y + dy]
                    heappush(pq, (d + 1, (x + dx, y + dy)))

    return dist, grad, crates


def nearest_objects(dist, objects, k=1):
    """
    Sort coins by distance and get the k nearest ones
    """
    objects.sort(key=lambda obj: dist[obj[0], obj[1]])
    objects = objects[:k]

    while len(objects) < k:
        objects.append((-1, -1))

    return objects


def get_danger_map(game_state):
    """
    Returns maps with danger levels: 0 for no danger, 4 for active explosions.
    For each tile in bomb radius: danger = 4 - countdown
    -> it is 4 when the bomb explodes in the next step
    """
    bombs = game_state['bombs']
    field = game_state['field']
    explosion_map = np.copy(game_state['explosion_map'])  # We do not want to accidentally change the original
    explosion_radius = 3  # Bombs affect 3 fields in each direction

    # Set all active explosions to danger level 4
    explosion_map = np.where(explosion_map > 0, 4, explosion_map)

    for (bx, by), t in bombs:  # For every bomb
        danger_level = 4 - t
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # In every direction
            for r in range(explosion_radius + 1):  # From 0 to 3
                x, y = bx + r * dx, by + r * dy
                if 0 < x < 17 and 0 < y < 17:
                    if field[x, y] == -1:  # Stop at walls
                        break
                    # Only update the danger level if it's lower than the current one
                    # This ensures we don't overwrite higher danger levels with lower ones
                    if explosion_map[x, y] < danger_level:
                        explosion_map[x, y] = danger_level

    return explosion_map


def direction_to_object(obj, grad):
    """
    Follow gradient to agent, to determine path
    """
    if obj == (-1, -1):
        return dNONE
    x, y = obj

    direction = dNONE
    while grad[x, y] != dNONE:
        direction = -grad[x, y]
        if grad[x, y] == dUP:
            y -= 1
        elif grad[x, y] == dDOWN:
            y += 1
        elif grad[x, y] == dLEFT:
            x -= 1
        elif grad[x, y] == dRIGHT:
            x += 1

    return direction


def last_four_actions(self):
    """
    returns the last 4 actions as a list of integers.
    """
    return list(self.last_actions_deque) + [4] * (
            self.last_actions_deque.maxlen - len(self.last_actions_deque))  # padded with WAIT


def last_action(self):
    """
    returns the last action as a one-hot. Causes oscillation problem!
    """
    one_hot = [0] * 6
    if len(self.last_actions_deque):
        one_hot[self.last_actions_deque[-1]] = 1
    return one_hot


def is_repeating_actions(self):
    """
    returns 1 when the agent is repeating actions such that it is stuck in one location,
    e.g. 4 times wait, left-right-left-right, up-down-up-down
    """
    # If fewer than 4 actions have been taken, it cannot be stuck
    if len(self.last_actions_deque) < 4:
        return 0

    # Convert actions to a list for easier processing
    actions = list(self.last_actions_deque)

    # Check if all actions are WAIT
    if actions.count(4) == 4:
        return 1

    # Check for patterns indicating the agent is stuck
    if actions[0] == actions[2] and actions[1] == actions[3] and actions[0] != actions[1] or actions == [4] * 4:
        return 1

    return 0


def is_repeating_positions(self):
    """
    Determines if the agent is likely stuck based on its movement history.
    :return: 1 if the agent is likely stuck, 0 otherwise.
    Not called as a feature, but to make a stuck agent drop a bomb to force it to move.
    """
    min_unique_positions = 5
    # Not enough history to make a decision
    if len(self.coordinate_history) < 20:
        return 0

    # Calculate the number of unique positions visited in the last 20 steps
    unique_positions = len(set(self.coordinate_history))

    if unique_positions < min_unique_positions:
        return 1
    else:
        return 0


def is_at_crossing(ax, ay):
    """
    Idea: If bombs are not dropped at a crossing, two directions are blocked which lessens the impact.
    This feature enables us to punish that.
    """
    return int((ax % 2 == 1) and (ay % 2 == 1))


def crates_reachable(ax, ay, field):
    """
    :return: How many crates would a bomb dropped on this position destroy?
    """
    crate_count = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # in every direction
        for r in range(1, 4):  # from 1 to 3
            x, y = ax + r * dx, ay + r * dy
            if 0 < x < 17 and 0 < y < 17:
                if field[x, y] == -1:  # Stop at walls
                    break
                elif field[x, y] == 1:
                    crate_count += 1

    return crate_count


def neighboring_crate_count(ax, ay, field):
    """
    Reachable crates for the surrounding positions
    """
    neigh = [0, 0, 0, 0]
    directions = [(ax + 1, ay), (ax, ay + 1), (ax - 1, ay), (ax, ay - 1)]  # Right, Down, Left, Up

    for i, d in enumerate(directions):
        if field[d] == 0:
            neigh[i] = crates_reachable(ax, ay, field)

    return neigh


def path_can_be_blocked_by_enemy(dist_agent, safe_positions, others, field, predict=False):
    """
    For every enemy, use get_distance_map(ax, ay, arena) to get their distance map
    and then check for every one of the safe tiles whether dist_enemy[tile] > dist_agent[tile]
    and only return those that fulfill this.
    """
    safe_mask = np.ones(len(safe_positions), dtype=bool)

    for o in others:
        if dist_agent[o] < 10:
            ex, ey = o
            dist_enemy, _, _ = get_distance_map(ex, ey, field)

            for i, safe_pos in enumerate(safe_positions):
                dist_agent_to_safe = dist_agent[safe_pos]
                dist_enemy_to_safe = dist_enemy[safe_pos]

                # If the enemy can reach the safe tile before or at the same time as the agent, it is not safe
                #if predict:
                    #dist_enemy_to_safe -= 1  # The enemy could move closer while our agent places a bomb
                if dist_enemy_to_safe <= dist_agent_to_safe:
                    safe_mask[i] = False

    return [tuple(pos) for pos in np.array(safe_positions)[safe_mask]]  # [(a,b), (c,d), ...]


def get_nearest_safe_tile(ax, ay, danger_map, others, bombs, dist_t, grad_t, dist, in_danger, field, predict, logger,
                          danger=False):
    """Idea: when an agent finds itself in the explosion radius of a bomb, this should point the agent
    in the direction of the nearest safe tile, especially useful for avoiding placing a bomb and then
    walking in a dead end waiting for the bomb to blow up. Should also work for escaping enemy bombs.
    Predict is true when this is called in the context of predicting whether there is an escape route
    if the agent drops a bomb now, in which case we cannot allow ignoring tiles reachable by enemies
    to find a safe tile if there is no other option. If there are no safe tiles found in the first pass,
    the function is called again with danger = True to at least find tiles that are less safe because
    they could be reached by an enemy or are in the explosion zone of a bomb that has been dropped later
    than the current one. """

    if not in_danger:
        return [0, 0, 0, 0]

    distances = [15, 15, 15, 15]  # Default to 15 if no safe tile is found in a direction
    all_safe_tiles = []
    nearest_safe_tiles = [None, None, None, None]
    radius = 4  # Limit the search to a radius of 4 tiles around the agent

    # Define the search boundaries to ensure they are within the field limits
    x_min = max(1, ax - radius)
    x_max = min(16, ax + radius + 1)
    y_min = max(1, ay - radius)
    y_max = min(16, ay + radius + 1)

    def is_safe_tile(x, y):
        """Check if the tile is safe based on the danger map and other conditions."""
        if not danger:
            if dist_t[x, y] < 5 and (x, y) not in others and (x, y) not in bombs and danger_map[x, y] == 0:
                return True
            return False
        else:
            if dist_t[x, y] < 5 and (x, y) not in others and (x, y) not in bombs and danger_map[x, y] < danger_map[ax, ay]:
                return True
            return False

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if is_safe_tile(x, y):
                all_safe_tiles.append((x, y))  # Collect all safe tiles

    if not danger:
        # remove tiles that could be reached by an enemy
        all_safe_tiles = path_can_be_blocked_by_enemy(dist, all_safe_tiles, others, field, predict)

    for safe_tile in all_safe_tiles:
        # enter the distance to the tile in the correct direction
        safe_x, safe_y = safe_tile
        dir_to_tile = direction_to_object(safe_tile, grad_t)

        if dir_to_tile == dRIGHT and safe_x > ax:
            distance = dist[safe_x, safe_y]
            if distance < distances[0]:  # Right
                distances[0] = distance
                nearest_safe_tiles[0] = safe_tile
        elif dir_to_tile == dDOWN and safe_y > ay:
            distance = dist[safe_x, safe_y]
            if distance < distances[1]:  # Down
                distances[1] = distance
                nearest_safe_tiles[1] = safe_tile
        elif dir_to_tile == dLEFT and safe_x < ax:
            distance = dist[safe_x, safe_y]
            if distance < distances[2]:  # Left
                distances[2] = distance
                nearest_safe_tiles[2] = safe_tile
        elif dir_to_tile == dUP and safe_y < ay:
            distance = dist[safe_x, safe_y]
            if distance < distances[3]:  # Up
                distances[3] = distance
                nearest_safe_tiles[3] = safe_tile

    for i, safe_tile in enumerate(nearest_safe_tiles):
        # Check whether the path to the safe tile could be blocked by an explosion
        if distances[i] < 5 and not distances[i] == 0 and not distances.count(15) == 3:
            x, y = ax, ay
            while (x, y) != safe_tile:
                dir_to_tile = direction_to_object(safe_tile, grad_t)

                # Update cx, cy based on the direction and ensure it stays within bounds
                if dir_to_tile == dRIGHT and x < 16:
                    x += 1
                elif dir_to_tile == dDOWN and y < 16:
                    y += 1
                elif dir_to_tile == dLEFT and x > 1:
                    x -= 1
                elif dir_to_tile == dUP and y > 1:
                    y -= 1
                else:
                    break

                if danger_map[x, y] > danger_map[ax, ay] and not ((x, y) == safe_tile and danger_map[safe_tile] < 3):
                    distances[i] = 15
                    break

    if all(d > 5 - danger_map[ax, ay] for d in distances) and not predict and not danger:
        return get_nearest_safe_tile(ax, ay, danger_map, others, bombs, dist_t, grad_t, dist, in_danger, field, predict,
                                     logger, danger=True)

    return distances


def function_of_immortality(self, game_state, danger_map, others, bombs, dist_t, grad_t, dist, grad, nearest_safe_t,
                            dend_list, can_drop_bomb_on_trapped, logger):
    """
    Is it over-engineered? Perhaps.
    Was it worth it? Yes

    Returns a binary vector of length six, each bit signifying whether that action is "allowed" (0) or "disallowed" (1).
    This function is directly used to prohibit the disallowed actions in the choose_action function,
    significantly speeding up the training progress.
    """
    disallowed = [0, 0, 0, 0, 0, 0]  # right down left up wait bomb
    _, _, bomb_available, (ax, ay) = game_state['self']
    disallowed[5] = int(not bomb_available)  # Cannot bomb if it has no bomb.
    current_danger = danger_map[ax, ay]
    # logger.info("Current danger: " + str(current_danger))

    field = game_state['field']

    directions = [(ax + 1, ay), (ax, ay + 1), (ax - 1, ay), (ax, ay - 1)]  # Right, Down, Left, Up

    # Do not walk into explosion that is more progressed than the one on the current tile, or a wall, or a bomb
    for i, (dx, dy) in enumerate(directions):
        if field[dx, dy] != 0 or danger_map[dx, dy] == 4 or (dx, dy) in bombs:
            # This still allows stepping in bomb radius if it can escape in the next step
            disallowed[i] = 1

    # logger.info("min(nearest_safe) = " + str(min(nearest_safe_t)))

    if current_danger > 0:  # Is in imminent explosion zone
        logger.info("Nearest_safe: " + str(nearest_safe_t))
        for i, safe_dist in enumerate(nearest_safe_t):
            if safe_dist > 5 - current_danger:
                disallowed[i] = 1  # Disallow directions where the agent would not be able to escape in time
        if min(nearest_safe_t) == 5 - current_danger:
            disallowed[4] = 1
            disallowed[5] = 1  # Do not wait or bomb, escape

    # logger.info("Disallowed1: " + str(disallowed))

    if (ax, ay) in bombs:  # Is standing on a bomb
        for i, safe_dist in enumerate(nearest_safe_t):
            if safe_dist > 5 - current_danger:  # For example, it is a dead end and the agent would trap itself there
                disallowed[i] = 1
        for i, (dx, dy) in enumerate(directions):
            if field[dx, dy] != 0:  # Danger or obstacle ahead
                disallowed[i] = 1

    # Disallow bomb placement if it would likely lead to self death
    if bomb_available:
        # Predict the danger level after placing a bomb
        future_danger_map = np.copy(danger_map)
        future_danger_map[ax, ay] = 1  # Bomb placed on current position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, 4):
                x, y = ax + r * dx, ay + r * dy
                if 0 < x < 17 and 0 < y < 17:
                    if field[x, y] == -1:  # Stop at walls
                        break
                    future_danger_map[x, y] = 1
                else:
                    break

        # Predict the nearest safe tile if it laid a bomb
        future_nearest_safe = get_nearest_safe_tile(ax, ay, future_danger_map, others, bombs, dist_t, grad_t, dist,
                                                    True, field, True, logger)
        logger.info("Predicted future nearest safe if bomb: " + str(future_nearest_safe))

        # Distance map ignores enemies and bombs, so we manually have to check whether something blocks that direction
        for i, d in enumerate(directions):
            if d in others or d in bombs:
                future_nearest_safe[i] = 15

        if all(x > 4 for x in future_nearest_safe):  # There would be no safe tile reachable if it dropped a bomb now
            disallowed[5] = 1  # Disallow placing a bomb

    # logger.info("Disallowed before dead end: " + str(disallowed))

    dedend, is_on_tile_before = do_not_the_dead_end(ax, ay, dend_list, others, dist, grad)
    logger.info("Result of do not enter dead end function: " + str(dedend))
    # Prevent it from entering dead end if dangerous, but allow it to enter dead end as a last resort
    if not (disallowed[4] == 1 and all(d or d_ded for d, d_ded in zip(disallowed[0:4], dedend))):
        for i in range(4):
            if dedend[i]:
                disallowed[i] = 1

    if any(dedend) and not is_on_tile_before:  # Do not wait, except when already at the exit
        disallowed[4] = 1
        disallowed[5] = 1

    if not disallowed[5] and can_drop_bomb_on_trapped:
        disallowed[0:5] = [1] * 5
        # Forces agent to drop a bomb to get a sure kill.
        # Without this: unnecessarily follows enemy all the way into the dead end / gives the enemy time to kill itself
    elif can_drop_bomb_on_trapped and not disallowed[4] and not danger_map[ax, ay] > 0:
        disallowed[0:4] = [1] * 4  # If it has no bomb, at least block the enemy in the dead end

    # Try to prevent getting "in die Zange genommen".
    # todo: improve this, a lot. The agent still dies exactly because of this
    nearby_enemies = [other for other in others if dist_t[other[0], other[1]] <= 2]
    if len(nearby_enemies) >= 2:
        # Check if agent is trapped between two walls
        walls_count = sum(1 for (dx, dy) in directions if field[dx, dy] == -1)
        if walls_count >= 2:
            disallowed[4] = 1  # Disallow wait
            disallowed[5] = 1  # Disallow bomb

    if all(disallowed):  # Should only happen if the agent now dies certainly, or if there is an error in this function
        disallowed = [0] * 6

    # One last pass du disallow obviously invalid actions that might have been re-allowed earlier in this function
    for i, d in enumerate(directions):
        if danger_map[d] == 4 or field[d] != 0 or d in bombs:
            disallowed[i] = 1
    if not bomb_available:
        disallowed[5] = 1

    # Forces agent to drop a bomb to become unstuck
    if is_repeating_positions(self) and disallowed[5] == 0:
        disallowed[0:5] = [1] * 5

    return disallowed


######################################################################### Reviewed and cleaned until here

def get_enemies_distances_and_directions(dist, others, grad):
    """
    :returns the direction to the nearest enemies as one-hot scaled by the sqrt of their distance, an array of length 12.
    """

    features = []
    max_dist = np.sqrt(
        30)  # Max distance value to be used in the features, the high INF value for the distances "confuses" the model

    for other in others[:3]:  # Limit to 3 enemies
        direction = direction_to_object(other, grad)
        direction_one_hot = [
            int(direction == dRIGHT) * min(np.sqrt(dist[other]), max_dist),
            int(direction == dDOWN) * min(np.sqrt(dist[other]), max_dist),
            int(direction == dLEFT) * min(np.sqrt(dist[other]), max_dist),
            int(direction == dUP) * min(np.sqrt(dist[other]), max_dist),
        ]
        features.extend(direction_one_hot)

    # Pad with zeros if there are fewer than 3 enemies
    while len(features) < 12:
        features.extend([0, 0, 0, 0])

    return features


def get_enemies_relative_positions(ax, ay, others):
    """
    :returns the direction to the nearest enemies as one-hot scaled by the sqrt of their distance, an array of length 12.
    """

    features = []

    for other in others[:3]:
        ox, oy = other  # Enemy's position

        # Compute relative position (directional vector) of the enemy
        dx = ox - ax  # Horizontal difference
        dy = oy - ay  # Vertical difference

        # Horizontal component (Right/Left)
        right_component = max(0, dx)
        left_component = max(0, -dx)

        # Vertical component (Down/Up)
        down_component = max(0, dy)
        up_component = max(0, -dy)

        # Add these components for the current enemy to the feature list
        features.extend([right_component, down_component, left_component, up_component])

    # Pad with zeros if there are fewer than 3 enemies
    while len(features) < 12:
        features.extend([0, 0, 0, 0])

    return np.sqrt(features)


def get_dead_end_map(field, others, bombs):
    """
    Returns map of all dead ends.
    The idea is to teach the agent to avoid these when enemies are nearby and to follow enemy agents inside
    and then placing a bomb behind them when they make the mistake of entering one.
    """
    dead_end_map = np.zeros_like(field)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Down, Left, Up
    dead_end_list = []

    def is_dead_end(x, y):
        """Check if a tile is part of a dead end and trace the full dead end path."""
        dead_end_path = [(x, y)]  # Start from this tile
        current_x, current_y = x, y
        prev_x, prev_y = -1, -1

        while True:
            open_neighbors = []
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                # Only count open tiles, avoid the previous one
                if (nx, ny) != (prev_x, prev_y) and field[nx, ny] == 0 \
                        and (nx, ny) not in bombs:
                    open_neighbors.append((nx, ny))

            if len(open_neighbors) == 1:  # Dead-end continues if only one neighbor is open
                prev_x, prev_y = current_x, current_y
                current_x, current_y = open_neighbors[0]
                dead_end_path.append((current_x, current_y))
            else:
                break  # If more than one neighbor is open, it's not part of a dead end

        if len(dead_end_path) > 1:
            for px, py in dead_end_path:  # [:-1] if you do not want to mark the tile before the dead end
                dead_end_map[px, py] = 1

            # Check if there is an enemy in this dead end
            enemy_in_dead_end = None
            for other in others:
                _, _, _, pos = other
                if pos in dead_end_path:  # If enemy is in the dead end path
                    enemy_in_dead_end = other
                    break

            dead_end_list.append({
                'closed_end': dead_end_path[0],
                'open_end': dead_end_path[-2],
                'tile_before_open_end': dead_end_path[-1],  # the tile before the actual entrance
                'enemy': enemy_in_dead_end,
                'path': dead_end_path[0:-1]
            })

    # Iterate over every cell in the field
    for x in range(1, 16):
        for y in range(1, 16):
            if field[x, y] == 0:  # Free tile, check for dead end
                open_neighbors = sum(
                    1 for dx, dy in directions if field[x + dx, y + dy] == 0 and (x + dx, y + dy) not in bombs)
                if open_neighbors == 1:  # Only one open neighbor indicates a potential dead end entrance
                    is_dead_end(x, y)

    return dead_end_map, dead_end_list


def track_bombs(self, bombs, others, explosion_map):  # bombs is list of positions, others is full version of others
    """Returns for each bomb the bomb location and name of its owner. When a bomb appears that was
    not in the list in the previous step, the enemy that laid it must be on it. Save it as owner of that bomb.
    self.bomb_owners = Dictionary to track bomb owners: {bomb_position: agent_name}
    """
    # Check for new bombs that appeared in the current step
    current_bomb_positions = [bomb_pos for bomb_pos in bombs]

    for bomb_pos in current_bomb_positions:
        if bomb_pos not in self.bomb_owners:
            # Find the agent standing on the bomb position (who must have placed the bomb)
            for agent_name, _, _, (ax, ay) in others:
                if (ax, ay) == bomb_pos:
                    self.bomb_owners[bomb_pos] = agent_name
                    break

    # Remove bombs that have exploded
    exploded_bombs = [bomb_pos for bomb_pos in self.bomb_owners if explosion_map[bomb_pos] == 4]
    for bomb_pos in exploded_bombs:
        del self.bomb_owners[bomb_pos]
    return self.bomb_owners


def get_direction_to_enemy_in_dead_end(self, ax, ay, dead_end_list, dist, grad, dead_map, logger):
    """
    If there is the open end of a dead end containing an enemy closer to our agent than the manhattan distance of the
    enemy to the open end, return one-hot directions to the enemy.
    Also return whether the agent is less than four steps away
    from the closed end of the dead end, because if it drops a bomb then,
    the enemy is dead (except when another agent blows up a crate that belonged to the closed end).
    This can then easily be rewarded, enabling the agent to learn this behavior without the need for artificial constraints.
    """
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Down, Left, Up  # Current, Right, Down, Left, Up
    one_hot_direction = [0, 0, 0, 0]  # Placeholder for one-hot direction (Right, Down, Left, Up)
    can_trap_enemy = 0

    # Iterate over each dead end to find one that contains an enemy
    for dead_end in dead_end_list:
        enemy = dead_end['enemy']
        if enemy is None:
            continue  # No enemy in this dead end, skip

        open_end = dead_end['tile_before_open_end']
        closed_end = dead_end['closed_end']
        current_enemy_name, _, _, (ex, ey) = enemy

        # Check if the agent is closer to the open end than the enemy is
        agent_to_open_end_dist = dist[open_end[0], open_end[1]]
        enemy_to_open_end_dist = abs(ex - open_end[0]) + abs(
            ey - open_end[1])  # Manhattan distance (ignores curved dead ends)

        if agent_to_open_end_dist <= enemy_to_open_end_dist + 4:  # 4 because
            logger.info("Found nearby enemy in dead end: " + str(enemy[0]))
            other_enemy_would_get_kill = False
            for d in directions:
                adjacent_pos = (closed_end[0] + d[0], closed_end[1] + d[1])
                if adjacent_pos in self.bomb_owners:
                    if self.bomb_owners[adjacent_pos] != current_enemy_name:
                        # Enemy is blocked by the bomb of another enemy that would get the kill
                        other_enemy_would_get_kill = True
                        break  # No need to check further directions if one is found

            logger.info("Other enemy would get kill: " + str(other_enemy_would_get_kill))
            if other_enemy_would_get_kill:  # todo still gifts enemies free points -> forbid it completely somewhere
                continue
            direction = direction_to_object((ex, ey), grad)
            if direction == dRIGHT:
                one_hot_direction = [1, 0, 0, 0]
            elif direction == dDOWN:
                one_hot_direction = [0, 1, 0, 0]
            elif direction == dLEFT:
                one_hot_direction = [0, 0, 1, 0]
            elif direction == dUP:
                one_hot_direction = [0, 0, 0, 1]

            # Check if the agent can trap the enemy with a bomb (agent is near the closed end)
            agent_to_closed_end_dist = dist[closed_end[0], closed_end[1]]
            if agent_to_closed_end_dist <= 3 and dead_map[ax, ay] > 0 and \
                    (closed_end[0] == ex == ax or closed_end[1] == ey == ay):  # it has to be a straight dead end
                can_trap_enemy = 1

            break  # Stop after finding the first valid enemy in a dead end

    # Append whether an enemy is in a dead end and if the agent can trap them
    one_hot_direction.append(can_trap_enemy)
    return one_hot_direction


def do_not_the_dead_end(ax, ay, dead_end_list, others, dist, grad):
    """
    Do not enter a dead end if an enemy is nearby (within 4 tiles) and not inside the same dead end.
    Also, leave the dead end immediately if otherwise an enemy could reach the exit faster than you.
    :return: One-hot direction (inverted, blocked) to the dead-end entrance if an enemy is nearby, else a zero-vector.
    """

    one_hot_direction = [0, 0, 0, 0]  # Right, Down, Left, Up
    current_dead_end = None
    is_on_tile_before = False
    for dead_end in dead_end_list:
        # If the agent is anywhere along the dead-end path
        if (ax, ay) in dead_end['path'] or (ax, ay) == dead_end['tile_before_open_end']:
            current_dead_end = dead_end
            break

    # If the agent is not in any dead end, return
    if current_dead_end is None:
        return one_hot_direction, True

    if (ax, ay) == current_dead_end['tile_before_open_end']:
        is_on_tile_before = True

    nearby_enemies = [other for other in others]  # if dist[other] < 10] does not work
    nearby_enemies = [enemy for enemy in nearby_enemies if enemy not in current_dead_end['path']]

    if not nearby_enemies:
        return one_hot_direction, True  # No enemies nearby, no need to worry about the dead end

    # Find the nearest enemy and check the condition
    nearest_enemy = min(nearby_enemies, key=lambda e: dist[e])  # Find the nearest enemy
    nearest_enemy_distance = dist[nearest_enemy]  # Distance to the nearest enemy

    # Distance from agent to the 'tile_before_open_end'
    distance_to_exit = dist[current_dead_end['tile_before_open_end']]

    # If the agent cannot reach the exit in time based on the condition
    if 2 * distance_to_exit + 1 > nearest_enemy_distance:
        # Use the direction_to_object to find the direction toward the exit
        direction = direction_to_object(current_dead_end['tile_before_open_end'], grad)
        if direction == dNONE:
            direction = direction_to_object(current_dead_end['tile_before_open_end'],
                                            grad)  # exit blocked, presumably by own bomb

        # Convert direction to one-hot vector of blocked directions
        if direction == dRIGHT:
            one_hot_direction = [0, 1, 1, 1]
        elif direction == dDOWN:
            one_hot_direction = [1, 0, 1, 1]
        elif direction == dLEFT:
            one_hot_direction = [1, 1, 0, 1]
        elif direction == dUP:
            one_hot_direction = [1, 1, 1, 0]

        if not any(one_hot_direction):
            # print("Escape dead end:" + str(current_dead_end) + "position:" + str(ax) + str(ay))
            directions = [(ax + 1, ay), (ax, ay + 1), (ax - 1, ay), (ax, ay - 1)]  # Right, Down, Left, Up
            for i, d in enumerate(directions):
                if d == current_dead_end['open_end']:
                    one_hot_direction[i] = 1

    return one_hot_direction, is_on_tile_before


def is_next_to_enemy(ax, ay, others, d=1):
    """
    Checks if the agent is adjacent (d=1) or close to any enemy.
    Can be used to reward attacking enemy agents when they are close.
    """
    for ex, ey in others:
        if abs(ax - ex) + abs(ay - ey) == d:
            return True

    return False


def neighboring_explosions_or_coins(ax, ay, danger_map, coins):
    """
    Lets the agent "see" the explosions around it in the hope that it learns to avoid them better.
    """
    neigh = [0, 0, 0, 0, 0]
    directions = [(ax, ay), (ax + 1, ay), (ax, ay + 1), (ax - 1, ay), (ax, ay - 1)]  # Current, Right, Down, Left, Up

    neigh[0] = danger_map[ax, ay]

    for i, d in enumerate(directions):
        if d in coins:
            neigh[i] = -1  # -1 for coins
        neigh[i] = danger_map[d]

    return neigh


def do_not_get_surrounded(others, ax, ay, radius=5):
    # Not used so far
    """
    Returns a one-hot vector for the escape direction if the agent is in danger of being surrounded by two or more enemies.
    The agent is considered surrounded if two or more enemies are within a radius of 5 tiles and they are not all on one side.
    """
    if not len(others):
        return [0, 0, 0, 0]

    directions = ['right', 'down', 'left', 'up']
    escape_vector = [0, 0, 0, 0]

    # Find enemies within the 5-tile radius
    nearby_enemies = []
    for ex, ey in others:
        if abs(ex - ax) <= radius and abs(ey - ay) <= radius:
            nearby_enemies.append((ex, ey))

    if len(nearby_enemies) < 2:
        return escape_vector  # Not surrounded, no need to escape

    # Determine on which side of the agent the enemies are
    side_count = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
    for ex, ey in nearby_enemies:
        if ex > ax:  # Right side
            side_count['right'] += 1
        elif ex < ax:  # Left side
            side_count['left'] += 1
        if ey > ay:  # Down side
            side_count['down'] += 1
        elif ey < ay:  # Up side
            side_count['up'] += 1

    # Check if enemies are all on one side
    zero_sides = sum(1 for side in side_count.values() if side == 0)

    if zero_sides >= 2:
        return escape_vector  # All enemies are on one side, no need to escape

    min_enemy_count = min(side_count.values())

    # Mark all directions that have the minimum enemy count
    for i, direction in enumerate(directions):
        if side_count[direction] == min_enemy_count:
            escape_vector[i] = 1

    return escape_vector


def safer_direction(ax, ay, danger_map, bombs, enemies, field):
    """
    Problem: The agent often runs into danger after dropping a bomb. This function returns directions
    away from explosions or enemies if the enemy is on a bomb, thus supporting a safer escape
    """
    if not (ax, ay) in bombs:
        return [0, 0, 0, 0]

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Down, Left, Up
    danger_levels = [99, 99, 99, 99]

    for i, direction in enumerate(directions):
        dx, dy = direction
        neighbor = ax + dx, ay + dy
        if field[neighbor] == 0 and neighbor not in bombs and neighbor not in enemies and danger_map[neighbor] < 3:
            danger_levels[i] = 0
            tiles_in_this_direction = []
            x, y = ax, ay

            # Create a cone-shaped area of tiles
            for v in range(1, 5):  # Cone length of 4
                for h in range(-v, v + 1):
                    nx, ny = x + dx * v + h * dy, y + dy * v + h * dx
                    tiles_in_this_direction.append((nx, ny))

            for tile in tiles_in_this_direction:
                if 0 < tile[0] < 17 and 0 < tile[1] < 17:
                    if tile in bombs or tile in enemies:
                        danger_levels[i] += 3
                    if danger_map[tile] > 0:
                        danger_levels[i] += 1

    min_danger = min(danger_levels)
    best_direction_index = danger_levels.index(min_danger)

    return [1 if i == best_direction_index else 0 for i in range(4)]


# todo:
"""
today:
- clean code, order all the functions in a plausible order (importance, grouped by purpose etc.)


Immortality:
Improving function of immortality makes no sense anymore (except for in die Zange genommen problem), simply add features 
such that the agent can learn to avoid death even better


How to make the agent better aside from that:

Things our features are just not good enough for, yet:
- if bomb next to crate, do not let enemy be closer to crate so that if there is a coin the enemy gets it
- prioritizing coins (prefer to collect those that would be collected by an enemy if the agent collected another first)
- getting kills in complex multi-agent situations (make get_enemy_distances_and_directions better by not making each a
    one-hot, instead there should always be 2 non zero-values, the relative coordinates.)
- prioritizing attacking 'weak' enemies - there will be teams with bad implementations we could farm points off of
    example: agent does not move -> keep location history of other agents and move toward "braindead" agents



Scientific investigations:

try significantly increasing the batch size. Result: 100% cpu usage an noisy computer, results are not better

experiment: add many layers and see what happens

Why exactly does performance decrease after some amount of training steps? Why the periodic ups and downs?

Networks with layer sizes decreasing from input to output appear to be better. Investigate. 

Test our agent against others from github in 2 vs 2 and record avg scores

LSTM cells, e.g. can remember spots where the bomb could destroy many crates


Usability:

Do something so that the logger outputs from the state to feature function output only once

Add victory rate to plotter

Save current version of the model in another test folder, so we can test future versions against that one 
to determine if there has been improvement

Code cleanup! Make the code more similar to Oles version

Remove unneeded functions

How can the training process be accelerated? Add logger statements for what functions need lots of time.
supposition: it is the function of immortality / calculating the distance map twice with and without bombs and enemies

"""
