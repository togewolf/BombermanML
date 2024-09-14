import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import numpy as np
import torch.nn.functional as f
from collections import deque
import random
from heapq import heapify, heappop, heappush


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, l1_dims, l2_dims, l3_dims, l4_dims, l5_dims, dropout_rate=0.1):
        # seems to work better without dropout
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
        # actions_prob = torch.sigmoid_(actions)
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

        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, l1_dims=2 ** 9, l2_dims=2 ** 9,
                                   l3_dims=2 ** 8, l4_dims=2 ** 7, l5_dims=2 ** 7)  # experiment here

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
        features = state_to_features(game_state, self.logger)
        blocked = features[8:14]  # [0,0,0,0,0,0]

        if np.random.random() > self.epsilon or not train:
            state = torch.tensor([features], dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state, train).squeeze()

            for i in range(6):
                if blocked[i] == 1:
                    actions[i] = -9999

            action = torch.argmax(actions).item()

            # action_probabilities = actions.clone().detach().softmax(dim=1).squeeze()
            # action = int(torch.multinomial(action_probabilities, 1).item())

        else:
            p = [.20, .20, .20, .20, .10, .10]

            for i in range(6):
                if blocked[i] == 1:
                    p[i] = 0

            total_prob = np.sum(p)
            if total_prob > 0:
                p /= total_prob
            else:
                p = [.20, .20, .20, .20, .10, .10]
                self.logger.info("All actions disallowed!")

            action = random.choices(range(6), weights=p, k=1)[0]

        # self.logger.info("Eps = " + str(self.epsilon) + ")")
        last_actions_deque.append(action)
        _, _, _, (ax, ay) = game_state['self']
        coordinate_history.append((ax, ay))
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


def state_to_features(game_state: dict, logger) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param logger: imported the logger here to log logs
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    features = []

    # useful stuff for feature calculating functions, to avoid inefficient recalculations
    _, _, bomb_available, (ax, ay) = game_state['self']
    field = game_state['field']
    coins = game_state['coins']  # [(x, y)]
    crates = np.argwhere(field == 1)
    crates = [tuple(pos) for pos in crates]  # so they are in the same format as the coins
    bombs = game_state['bombs']  # [(x, y), countdown]
    bombs = [b[0] for b in bombs]
    others = game_state['others']
    others = [t[3] for t in others]
    dist, grad, crate_map = distance_map(ax, ay, field)
    danger_zone = danger_map(game_state)
    in_danger = False
    if danger_zone[ax, ay] > 0:
        in_danger = True
    # nearest_safe = nearest_safe_tile(ax, ay, danger_zone, others, bombs, dist, grad, in_danger)
    dend_map, dend_list = dead_end_map(field, others, bombs)

    dist_t, grad_t, _ = distance_map_with_temporaries(ax, ay, field, others, bombs)
    nearest_safe_t = nearest_safe_tile(ax, ay, danger_zone, others, bombs, dist_t, grad_t, True, field)
    # the t marker means that temporary objects such as bombs and agents are also considered here

    dteide = direction_to_enemy_in_dead_end(ax, ay, dend_list, dist, grad, dend_map)

    blocked = disallowed_actions(game_state, danger_zone, others, bombs, dist_t, grad_t, nearest_safe_t, dend_map,
                                 dend_list, dteide[4], logger)

    edad = enemies_distances_and_directions(dist, others, grad)

    suggestion = direction_suggestion(field, coins, crates, others, dist_t, grad_t, dist, grad, dteide, logger)

    # logger.info("In danger: " + str(in_danger))
    # logger.info(danger_zone)
    # logger.info("Distances to nearest safe tiles: " + str(nearest_safe))
    # logger.info("Distance map: ")
    # logger.info(dist)
    # logger.info("Enemies distances: " +str(enemies_distances_and_directions(dist, others, grad)))
    # logger.info("is at crossing: " + str(is_at_crossing(ax, ay)))
    # logger.info(field)
    # logger.info(dend_map)
    # logger.info(nearest_safe)

    # features
    features.append(int(bomb_available))  # feat 0
    features.append(crates_reachable(ax, ay, field))  # feat 1
    features.append(in_danger)  # feat 2
    features.append(is_repeating_actions())  # feat 3
    features.extend(suggestion)  # feat 4-7
    features.extend(blocked)  # 8-13
    features.extend(edad)  # 14-25
    features.append(is_at_crossing(ax, ay))  # 26
    features.extend(last_action())  # 27-32
    features.append(dteide[4])  # 33 whether it can bomb trap the enemy
    features.append(int(is_next_to_enemy(ax, ay, others)))  # 34
    features.extend(neighboring_explosions_or_coins(ax, ay, danger_zone, coins))  # 35-39
    # features.extend(do_not_get_surrounded(others, ax, ay, radius=5))

    return features


def direction_suggestion(field, coins, crates, others, dist_t, grad_t, dist, grad, dteide, logger):
    """
    Selects a goal tile: trappable enemy, coin, crate, enemy in order of decreasing importance.
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

    if dist_c < 20:
        logger.info("Current goal: coin")
        return k_nearest_objects_feature(dist_t, coins, grad_t)

    coins_left_in_crates = True  # Todo

    if len(crates) and coins_left_in_crates:
        logger.info("Current goal: crate")
        return k_nearest_crates_feature(field, dist_t, crates, grad_t)

    if len(others):
        logger.info("Current goal: enemy")
        return k_nearest_objects_feature(dist, others, grad)

    # todo: if suggestion is still 0 here, this means our agent is the only thing left on the map. Do a victory dance.

    return suggestion


def k_nearest_objects_feature(dist, objects, grad, k=1):
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


def k_nearest_crates_feature(field, dist, crates, grad, k=1):
    """
    :returns the direction to the nearest crate as one-hot.
    Useful for finding crates when there are few left.
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


def nearest_crate_feature(field, dist, crates, grad, k=1):
    """
    :returns the direction to the nearest location next to a crate as one-hot.
    Useful for finding crates when there are few left.
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
            int(direction == dRIGHT) * np.sqrt(dist[crate]),
            # for some reason, high feature values cause the model to converge slowly
            int(direction == dDOWN) * np.sqrt(dist[crate]),
            int(direction == dLEFT) * np.sqrt(dist[crate]),
            int(direction == dUP) * np.sqrt(dist[crate])
        ]
        features.extend(direction_one_hot)

    return features


def distance_map(ax, ay, arena):
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


def distance_map_with_temporaries(ax, ay, arena, others, bombs):
    """
    :returns distance to every position and corresponding gradient calculated with Dijkstra
    This one additionally considers enemies and bombs.
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


def danger_map(game_state):
    """
    Returns maps with danger levels: 0 for no danger
    4 for active explosions
    For each tile in bomb radius: danger = 4 - countdown
    -> it is 4 when the bomb explodes in the next step
    """
    bombs = game_state['bombs']
    field = game_state['field']
    explosion_map = np.copy(game_state['explosion_map'])
    explosion_radius = 3  # Bombs affect 3 fields in each direction

    explosion_map = np.where(explosion_map > 0, 4, explosion_map)

    for (bx, by), t in bombs:  # for every bomb
        danger_level = 4 - t
        explosion_map[bx, by] = danger_level
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # in every direction
            for r in range(explosion_radius + 1):  # from 0 to 3
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


def relative_bomb_positions(game_state, ax, ay):
    bombs = game_state['bombs']
    rel_positions = []
    for (bx, by), countdown in bombs[:4]:  # Up to 4 bombs
        rel_positions.append(bx - ax)
        rel_positions.append(by - ay)
        rel_positions.append(countdown)
    while len(rel_positions) < 12:  # Ensure fixed length of 8 (4 bombs x 2 coordinates)
        rel_positions.append(-1)  # Fill with -1 if fewer than 4 bombs
    return rel_positions


def player_map(ax, ay, others, arena):
    players = np.full_like(arena, 0)
    players[ax, ay] = 1
    for (x, y) in others:
        players[x, y] = -1

    return players


last_actions_deque = deque(maxlen=4)


def last_k_actions():
    """
    :returns the last 4 actions.
    """
    return list(last_actions_deque) + [4] * (last_actions_deque.maxlen - len(last_actions_deque))  # padded with WAIT


def last_action():
    """
    returns the last action as a one-hot
    """
    one_hot = [0] * 6
    if len(last_actions_deque):
        one_hot[last_actions_deque[-1]] = 1
    return one_hot


def is_repeating_actions():
    """
    returns 1 when the agent is repeating actions such that it is stuck in one location,
    e.g. 4 times wait, left-right-left-right, up-down-up-down
    """
    # If fewer than 4 actions have been taken, it cannot be stuck
    if len(last_actions_deque) < 4:
        return 0

    # Convert actions to a list for easier processing
    actions = list(last_actions_deque)

    # Check if all actions are WAIT
    if actions.count(4) == 4:
        return 1

    # Check for patterns indicating the agent is stuck
    if actions[0] == actions[2] and actions[1] == actions[3] and actions[0] != actions[1]:
        return 1

    return 0


coordinate_history = deque([], maxlen=20)


def is_repeating_positions():
    """
    Determines if the agent is likely stuck based on its movement history.
    :return: 1 if the agent is likely stuck, 0 otherwise.
    Not called as a feature, but to make a stuck agent drop a bomb to force it to move.
    """
    min_unique_positions = 5
    # Not enough history to make a decision
    if len(coordinate_history) < 20:
        return 0

    # Calculate the number of unique positions visited in the last 20 steps
    unique_positions = len(set(coordinate_history))

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


def neighboring(ax, ay, game_state, k=2):
    """
    Returns a flattened array of size (2k+1)^2 showing the surroundings of the agent (only crates and walls).
    """
    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']

    # Initialize a fixed-size array
    neighbors = np.full((2 * k + 1, 2 * k + 1), -1)  # Using -1 as default for outside field

    # Compute the actual boundaries in the field
    x_min, x_max = max(0, ax - k), min(field.shape[0], ax + k + 1)
    y_min, y_max = max(0, ay - k), min(field.shape[1], ay + k + 1)

    # Determine the position within the neighbors array where the field data will be placed
    x_start = max(0, k - ax)
    y_start = max(0, k - ay)

    # Place the relevant slice of the field into the neighbors array
    neighbors[x_start:x_start + (x_max - x_min), y_start:y_start + (y_max - y_min)] = field[x_min:x_max, y_min:y_max]

    # Add bombs
    for (bx, by), countdown in bombs:
        if x_min <= bx < x_max and y_min <= by < y_max:
            neighbors[bx - x_min + x_start, by - y_min + y_start] = -2

    # Add coins
    for (cx, cy) in coins:
        if x_min <= cx < x_max and y_min <= cy < y_max:
            neighbors[cx - x_min + x_start, cy - y_min + y_start] = 2

    return neighbors.flatten()


def free_distance(ax, ay, field):
    """
    :return: in all four directions the sqrt of the distance the agent can "see", e.g. if there is a wall next to the agent
    zero in that direction. The distance can be at most 14 when the agent is at the edge of the field and a whole
    row is free
    """
    distances = [0, 0, 0, 0]  # Distances for right, down, left, and up

    # Right direction
    for x in range(ax + 1, field.shape[0]):
        if field[x, ay] != 0:
            break
        distances[0] += 1

    # Down direction
    for y in range(ay + 1, field.shape[1]):
        if field[ax, y] != 0:
            break
        distances[1] += 1

    # Left direction
    for x in range(ax - 1, -1, -1):
        if field[x, ay] != 0:
            break
        distances[2] += 1

    # Up direction
    for y in range(ay - 1, -1, -1):
        if field[ax, y] != 0:
            break
        distances[3] += 1

    return np.sqrt(distances)


def path_can_be_blocked_by_enemy(dist, safe_pos, others, field, max_steps=4):
    """
    todo: quite inefficient. Slows down training significantly
    Use BFS to check if an enemy can block the path to the safe tile within `max_steps` steps.
    Compare the agent's distance to the safe tile and the enemy's distance.
    """
    sx, sy = safe_pos
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Down, Left, Up

    agent_distance = dist[sx, sy]

    if agent_distance > 4:
        return False  # If the agent can't reach the safe tile, don't consider it blockable

    # BFS setup for enemies
    for o in others:
        ex, ey = o
        enemy_queue = deque([(ex, ey, 0)])  # (enemy_x, enemy_y, current steps)
        enemy_visited = {(ex, ey)}

        # BFS for the enemy to find if they can reach the safe tile before or at the same time as the agent
        while enemy_queue:
            x, y, steps = enemy_queue.popleft()

            # Stop if we've exceeded the agent's steps or maximum allowed steps
            if steps > agent_distance or steps > max_steps:
                continue

            # Check if the enemy is at the current position
            if (x, y) == (sx, sy):
                return True  # Enemy could block the path

            # Explore neighboring tiles
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if field[nx, ny] == 0 and (nx, ny) not in enemy_visited:  # Not visited yet
                    enemy_visited.add((nx, ny))
                    enemy_queue.append((nx, ny, steps + 1))

    return False


def nearest_safe_tile(ax, ay, danger_map, others, bombs, dist, grad, in_danger, field):
    """Idea: when an agent finds itself in the explosion radius of a bomb, this should point the agent
    in the direction of the nearest safe tile, especially useful for avoiding placing a bomb and then
    walking in a dead end waiting for the bomb to blow up. Should also work for escaping enemy bombs."""

    if not in_danger:
        return [0, 0, 0, 0]

    directions = [dRIGHT, dDOWN, dLEFT, dUP]
    distances = [15, 15, 15, 15]  # Default to 15 if no safe tile is found in a direction
    radius = 4  # Limit the search to a radius of 5 tiles around the agent for efficiency

    # Define the search boundaries to ensure they are within the field limits
    x_min = max(1, ax - radius)
    x_max = min(16, ax + radius + 1)
    y_min = max(1, ay - radius)
    y_max = min(16, ay + radius + 1)



    def is_safe_tile(x, y, allow_blocked_by_enemy=False):
        """Check if the tile is safe based on the danger map and other conditions."""
        if dist[x, y] != INF and (x, y) not in others and (x, y) not in bombs:
            if allow_blocked_by_enemy and (danger_map[x, y] == 0 or danger_map[x, y] < danger_map[ax, ay]):
                return True
            # Otherwise, ensure the path is not blocked by enemies or in an explosion
            return not path_can_be_blocked_by_enemy(dist, (x, y), others, field) and danger_map[x, y] == 0
        return False

    # First pass: check for tiles that cannot be blocked by enemies
    for direction in directions:
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if is_safe_tile(x, y, allow_blocked_by_enemy=False):
                    dir_to_tile = direction_to_object((x, y), grad)
                    if dir_to_tile == direction:
                        distance = dist[x, y]
                        direction_index = directions.index(direction)
                        distances[direction_index] = min(distances[direction_index], distance)

    # If all safe tiles are too far, perform a second pass allowing tiles blocked by enemies and lower danger level tiles
    if all(d > 5 - danger_map[ax, ay] for d in distances):
        for direction in directions:
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    if is_safe_tile(x, y, allow_blocked_by_enemy=True):
                        dir_to_tile = direction_to_object((x, y), grad)
                        if dir_to_tile == direction:
                            distance = dist[x, y]
                            direction_index = directions.index(direction)
                            distances[direction_index] = min(distances[direction_index], distance)

    return distances


def disallowed_actions(game_state, explosion_map, others, bombs, dist_t, grad_t, nearest_safe_t, dend_map, dend_list,
                       can_drop_bomb_on_trapped, logger):
    """
    can be used as feature or to block those actions in the choose_action function
    that would lead to certain death
    """
    disallowed = [0, 0, 0, 0, 0, 0]  # right down left up wait bomb
    _, _, bomb_available, (ax, ay) = game_state['self']
    disallowed[5] = int(not bomb_available)  # Cannot bomb if it has no bomb.
    current_danger = explosion_map[ax, ay]

    field = game_state['field']

    directions = [(ax + 1, ay), (ax, ay + 1), (ax - 1, ay), (ax, ay - 1)]  # Right, Down, Left, Up

    # do not walk into explosion that is more progressed than the one on the current tile, or a wall, or a bomb
    for i, (dx, dy) in enumerate(directions):
        if field[dx, dy] != 0 or explosion_map[dx, dy] == 4 or (dx, dy) in bombs:
            # this still allows stepping in bomb radius if it can escape in the next step
            disallowed[i] = 1

    # logger.info("min(nearest_safe) = " + str(min(nearest_safe_t)))

    if current_danger > 0:  # is in imminent explosion zone, bombs cannot be empty if this is the case
        logger.info("Nearest_safe: " + str(nearest_safe_t))
        for i, safe_dist in enumerate(nearest_safe_t):
            if safe_dist > 5 - current_danger:
                disallowed[i] = 1  # disallow directions where the agent would not be able to escape in time
        if min(nearest_safe_t) == 5 - current_danger:
            disallowed[4] = 1
            disallowed[5] = 1  # do not wait or bomb, escape


    logger.info("Disallowed1: " + str(disallowed))

    if (ax, ay) in bombs:  # is standing on a bomb, bombs cannot be empty if this is the case
        for i, safe_dist in enumerate(nearest_safe_t):
            if safe_dist > 4:  # it is a dead end and the agent would trap itself there
                disallowed[i] = 1
        for i, (dx, dy) in enumerate(directions):
            if field[dx, dy] != 0:  # Danger or obstacle ahead
                disallowed[i] = 1

    # disallow bomb placement if it would lead to certain death
    if bomb_available:
        # Predict the danger level after placing a bomb
        future_explosion_map = np.copy(explosion_map)
        future_explosion_map[ax, ay] = 1  # Bomb placed on current position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, 4):
                x, y = ax + r * dx, ay + r * dy
                if 0 < x < 17 and 0 < y < 17:
                    if field[x, y] == -1:  # Stop at walls
                        break
                    future_explosion_map[x, y] = 1
                else:
                    break

        # check whether the agent could escape the bomb
        future_nearest_safe = nearest_safe_tile(ax, ay, future_explosion_map, others, bombs, dist_t, grad_t, True,
                                                field)

        # distance map ignores enemies and bombs, so we manually have to check whether something blocks that direction
        for i, d in enumerate(directions):
            if d in others or d in bombs:
                future_nearest_safe[i] = 15

        if all(x > 4 for x in future_nearest_safe):
            disallowed[5] = 1  # Disallow placing a bomb

    logger.info("Disallowed before dead end: " + str(disallowed))

    dedend = do_not_enter_dead_end(ax, ay, dend_map, dend_list, others, dist_t, grad_t)
    # allow it to enter dead end as a last resort
    if not (disallowed[4] == 1 and all(d or d_ded for d, d_ded in zip(disallowed[0:4], dedend))):
        for i in range(4):
            if dedend[i]:
                disallowed[i] = 1

    logger.info("Disallowed after: " + str(disallowed))

    if not disallowed[5] and can_drop_bomb_on_trapped:
        disallowed[0:5] = [1] * 5
        # forces agent to drop a bomb to get a sure kill.
        # Without this: unnecessarily follows enemy all the way into the dead end
    elif can_drop_bomb_on_trapped and not disallowed[4] and not explosion_map[ax, ay] > 0:
        disallowed[0:4] = [1] * 4  # if it has no bomb, at least block the enemy in the dead end

    if all(disallowed):
        disallowed = [0] * 6
        for i, d in enumerate(directions):
            if explosion_map[d] == 4 or field[d] != 0 or d in bombs:
                disallowed[i] == 1

    if is_repeating_positions() and disallowed[5] == 0:
        disallowed[0:5] = [1] * 5  # forces agent to drop a bomb to become unstuck

    logger.info("DisallowedFinal: " + str(disallowed))

    return disallowed


def enemies_distances_and_directions(dist, others, grad):
    """
    :returns the direction to the nearest enemies as one-hot, an array of length 12.
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


def dead_end_map(field, others, bombs):
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
            for ex, ey in others:
                if (ex, ey) in dead_end_path:  # If enemy is in the dead end path
                    enemy_in_dead_end = (ex, ey)
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


def direction_to_enemy_in_dead_end(ax, ay, dead_end_list, dist, grad, dead_map):
    """
    If there is the open end of a dead end containing an enemy closer to our agent than the manhattan distance of the
    enemy to the open end, return one-hot directions to the enemy.
    Also return whether the agent is less than four steps away
    from the closed end of the dead end, because if it drops a bomb then,
    the enemy is dead (except when another agent blows up a crate that belonged to the closed end).
    This can then easily be rewarded, enabling the agent to learn this behavior without the need for artificial constraints.
    """
    one_hot_direction = [0, 0, 0, 0]  # Placeholder for one-hot direction (Right, Down, Left, Up)
    can_trap_enemy = 0

    # Iterate over each dead end to find one that contains an enemy
    for dead_end in dead_end_list:
        enemy_position = dead_end['enemy']
        if enemy_position is None:
            continue  # No enemy in this dead end, skip

        open_end = dead_end['tile_before_open_end']
        closed_end = dead_end['closed_end']
        ex, ey = enemy_position

        # Check if the agent is closer to the open end than the enemy is
        agent_to_open_end_dist = dist[open_end[0], open_end[1]]
        enemy_to_open_end_dist = abs(ex - open_end[0]) + abs(
            ey - open_end[1])  # Manhattan distance (ignores curved dead ends)

        if agent_to_open_end_dist <= enemy_to_open_end_dist + 2:
            # If agent is closer to the open end, find the direction to the enemy
            direction = direction_to_object(enemy_position, grad)
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


def do_not_enter_dead_end(ax, ay, dead_end_map, dead_end_list, others, dist, grad):
    """
    # todo does not work
    Do not enter a dead end if an enemy is nearby (within 4 tiles) and not inside the same dead end.
    :return: One-hot direction to the dead-end entrance if an enemy is nearby, else a zero-vector.
    """

    one_hot_direction = [0, 0, 0, 0]  # Right, Down, Left, Up
    current_dead_end = None
    for dead_end in dead_end_list:
        if (ax, ay) in dead_end['path']:  # If the agent is anywhere along the dead-end path
            current_dead_end = dead_end
            break

    # If the agent is not in any dead end, return a zero-vector
    if current_dead_end is None:
        return one_hot_direction

    nearby_enemies = [other for other in others if dist[other[0], other[1]] < 7]

    if not nearby_enemies:
        return one_hot_direction  # No enemies nearby, no need to worry about the dead end

    # Step 3: Find the nearest enemy and check the condition
    nearest_enemy = min(nearby_enemies, key=lambda e: dist[e[0], e[1]])  # Find the nearest enemy
    nearest_enemy_distance = dist[nearest_enemy[0], nearest_enemy[1]]  # Distance to the nearest enemy

    # Distance from agent to the 'tile_before_open_end' (where the agent can exit the dead-end)
    distance_to_exit = dist[current_dead_end['tile_before_open_end'][0], current_dead_end['tile_before_open_end'][1]]

    # If the agent cannot reach the exit in time based on the condition
    if 2 * distance_to_exit + 1 >= nearest_enemy_distance:
        # Step 4: Use the direction_to_object to find the direction toward the exit
        direction = direction_to_object(current_dead_end['tile_before_open_end'], grad)

        # Step 5: Convert direction to one-hot vector
        if direction == dRIGHT:
            one_hot_direction = [1, 0, 0, 0]
        elif direction == dDOWN:
            one_hot_direction = [0, 1, 0, 0]
        elif direction == dLEFT:
            one_hot_direction = [0, 0, 1, 0]
        elif direction == dUP:
            one_hot_direction = [0, 0, 0, 1]

    return one_hot_direction


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


# todo:
"""
- count coins and do not suggest to go to crate if no coin left
- pass more of the surrounding tiles to the agent for better decision making


Immortality:
?


How to make the agent better aside from that:

Things our features are just not good enough for, yet:
- do not block an enemy from escaping the bomb of another enemy so that the kill goes to the other enemy
- prioritizing coins (prefer to collect those that would be collected by an enemy if the agent collected another first)
- getting kills in complex multi-agent situations (make agent locations and directions better by not making each a
    one-hot, instead there should always be 2 non zero-values, the relative coordinates.)
- prioritizing attacking 'weak' enemies - there will be teams with bad implementations we could farm points off of
    example: agent does not move -> keep location history of other agents and move toward "braindead" agents



Scientific investigations:

try significantly increasing the batch size. Result: 100% cpu usage an noisy computer, results are not better

experiment: add many layers and see what happens

Why exactly does performance decrease after some amount of training steps? Why the periodic ups and downs?

Networks with layer sizes decreasing from input to output appear to be better. Investigate. 


Usability:

Add victory rate to plotter

Save current version of the model in another test folder, so we can test future versions against that one 
to determine if there has been improvement

Code cleanup! Make the code more similar to Oles version

Remove unneeded functions

How can the training process be accelerated? Add logger statements for what functions need lots of time.
supposition: it is the function of immortality / calculating the distance map twice with and without bombs and enemies

"""
