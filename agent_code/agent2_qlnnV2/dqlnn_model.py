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
    def __init__(self, lr, input_dims, l1_dims, l2_dims, l3_dims, l4_dims, dropout_rate=0.2):
        super(DeepQNetwork, self).__init__()
        # torch.autograd.set_detect_anomaly(True)  # remove when it works
        self.input_dims = input_dims
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.l3_dims = l3_dims
        self.l4_dims = l4_dims

        self.l1 = nn.Linear(self.input_dims, self.l1_dims)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.l2 = nn.Linear(self.l1_dims, self.l2_dims)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        # self.l3 = nn.Linear(self.l2_dims, self.l3_dims)
        # self.dropout3 = nn.Dropout(p=dropout_rate)
        # self.l4 = nn.Linear(self.l3_dims, self.l4_dims)
        # self.dropout4 = nn.Dropout(p=dropout_rate)
        self.lo = nn.Linear(self.l2_dims, 6)  # There are always six possible actions

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

    def forward(self, state, train):  # todo try prelu or other activation functions
        x = f.relu(self.l1(state))
        if train:
            x = self.dropout1(x)
        x = f.relu(self.l2(x))
        if train:
            x = self.dropout2(x)
        # x = f.relu(self.l3(x))
        # if train:
        #   x = self.dropout3(x)
        # x = f.relu(self.l4(x))
        # if train:
        #    x = self.dropout4(x)
        actions = self.lo(x)
        # actions_prob = torch.sigmoid_(actions)
        # softmax/sigmoid causes really weird errors here
        return actions


class Agent:
    def __init__(self, logger, gamma, epsilon, lr, input_dims, batch_size, max_mem_size=100000, eps_end=0.10,
                 eps_dec=3e-3):
        self.logger = logger
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, l1_dims=1024, l2_dims=1024,
                                   l3_dims=32, l4_dims=8)  # experiment here

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.mem_cntr = 0
        self.iteration = 0
        self.epoch = 0

    def store_transition(self, features, action, reward, state_, done):
        # todo: data augmentation for faster training, if needed
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
        features = state_to_features(game_state)

        # todo: build a constraints function that only allows safe and valid actions and block the invalid ones, making it impossible for the agent to kill itself
        #  Also block "getting stuck" e.g. repeating pattern of left right, or waiting more than three turns in a row

        # blocked = []

        if np.random.random() > self.epsilon or not train:
            state = torch.tensor([features], dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state, train).squeeze()

            # for i in range(6):
            #    if blocked[i] == 1:
            #        actions[i] = -9999

            action = torch.argmax(actions).item()

            # action_probabilities = actions.clone().detach().softmax(dim=1).squeeze()
            # action = int(torch.multinomial(action_probabilities, 1).item())

        else:
            p = [.20, .20, .20, .20, .10, .10]

            # for i in range(6):
            #    if blocked[i] == 1:
            #        p[i] = 0

            total_prob = np.sum(p)
            p /= total_prob

            action = random.choices(range(6), weights=p, k=1)[0]
            self.logger.info("Chose random action (Eps = " + str(self.epsilon) + ")")

        last_actions_deque.append(action)
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


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

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
    # bombs = game_state['bombs']  # [(x, y), countdown]
    dist, grad, crate_map = distance_map(ax, ay, field)
    danger_zone = danger_map(game_state, dist)
    in_danger = False
    if danger_zone[ax, ay] == 1:
        in_danger = True

    # features
    features.append(int(bomb_available))  # feat 0
    features.append(crates_reachable(ax, ay, field))  # feat 1
    features.append(in_danger)  # feat 2
    features.extend(k_nearest_coins_feature(dist, coins, grad, k=1))  # feat 3, 4, 5, 6
    features.extend(nearest_crate_feature(dist, crates, grad, k=1))  # feat 7, 8, 9, 10
    features.extend(free_distance(ax, ay, field))  # 11-14
    features.extend(nearest_safe_tile(game_state, dist, grad, in_danger))  # 15-18
    features.extend(is_blocked(game_state))  # 19-22
    features.extend(neighboring(ax, ay, game_state))  # 23-48  # todo perhaps a separate convolutional subnetwork for this
    # features.extend(last_k_actions())
    # todo: add all other features with their sizes in one line like this for clarity

    # todo:
    # bomb positions
    # is in explosion radius (with bomb timer, inverse intensity)
    # is at outer wall
    # positions of other agents

    return features


def k_nearest_coins_feature(dist, coins, grad, k=1):
    """
    :returns the direction to the nearest coin(s) as one-hot
    """

    if not len(coins):  # do not waste time when there are no coins
        return [0, 0, 0, 0]

    nearest_coins = nearest_objects(dist, coins, k)
    features = []
    for coin in nearest_coins:
        direction = direction_to_object(coin, grad)
        direction_one_hot = [
            int(direction == dRIGHT),
            int(direction == dDOWN),
            int(direction == dLEFT),
            int(direction == dUP),
        ]
        features.extend(direction_one_hot)

    return features


def nearest_crate_feature(dist, crates, grad, k=1):
    """
    :returns the direction to the nearest crate as one-hot.
    Useful for finding crates when there are few left.
    """
    if not len(crates):
        return [0, 0, 0, 0]

    nearest_crates = nearest_objects(dist, crates, k)
    features = []
    for crate in nearest_crates:
        direction = direction_to_object(crate, grad)
        direction_one_hot = [
            int(direction == dRIGHT),
            int(direction == dDOWN),
            int(direction == dLEFT),
            int(direction == dUP),
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
            if arena[x + dx, y + dy] != -1:
                # relax node
                if d + 1 < dist[x + dx, y + dy]:
                    dist[x + dx, y + dy] = d + 1
                    grad[x + dx, y + dy] = -dx - 2 * dy
                    crates[x + dx, y + dy] = crates[x, y] + arena[x + dx, y + dy]
                    heappush(pq, (d + 1, (x + dx, y + dy)))

    return dist, grad, crates


def is_blocked(game_state: dict):
    """Returns a vector of 4 values, for each direction whether it
    is blocked by a wall, crate, agent, todo bomb or (imminent) explosion."""
    _, _, _, (ax, ay) = game_state['self']
    arena = game_state['field']
    others = [(x, y) for _, _, _, (x, y) in game_state['others']]

    directions = [(ax + 1, ay), (ax, ay + 1), (ax - 1, ay), (ax, ay - 1)]  # Right, Down, Left, Up
    blocked = []

    for (x, y) in directions:
        if arena[x, y] == -1 or arena[x, y] == 1:
            blocked.append(1)  # Blocked by a wall or crate
        elif (x, y) in others:
            blocked.append(1)  # Blocked by another agent
        else:
            blocked.append(0)  # The direction is not blocked
    return np.array(blocked)


def nearest_objects(dist, objects, k=1):
    """
    Sort coins by distance and get the k nearest ones
    """
    objects.sort(key=lambda obj: dist[obj[0], obj[1]])
    objects = objects[:k]

    while len(objects) < k:
        objects.append((-1, -1))

    return objects


def danger_map(game_state, dist):
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    explosion_radius = 3  # Bombs affect up to 3 fields in each direction

    for (bx, by), t in bombs:  # for every bomb
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # in every direction
            for r in range(explosion_radius + 1):  # from 0 to 3
                if dist[bx + r * dx, by + r * dy] == INF:  # explosion blocked by wall
                    break
                explosion_map[bx + r * dx, by + r * dy] = max(t + 3, explosion_map[bx + r * dx, by + r * dy])

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
    in theory, this should provide enough information to escape dropped bomb
    :returns the last 4 actions. todo: one-hot
    """
    return list(last_actions_deque) + [4] * (last_actions_deque.maxlen - len(last_actions_deque))  # padded with WAIT


def crates_reachable(x, y, field):
    """
    :param x: x position
    :param y: y position
    :param field: 1 for crates, -1 for walls, 0 free space
    :return: How many crates would a bomb dropped on this position destroy?
    """
    bomb_radius = 3
    crate_count = 0
    for i in range(x - bomb_radius, x + bomb_radius + 1):
        if i < 1 or i > 16:
            break
        if field[i, y] == 1:
            crate_count += 1
    for i in range(y - bomb_radius, y + bomb_radius + 1):
        if i < 1 or i > 16:
            break
        if field[x, i] == 1:
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
            neighbors[bx - x_min + x_start, by - y_min + y_start] = 10

    # Add coins
    for (cx, cy) in coins:
        if x_min <= cx < x_max and y_min <= cy < y_max:
            neighbors[cx - x_min + x_start, cy - y_min + y_start] = -10

    return neighbors.flatten()


def free_distance(ax, ay, field):
    """
    :return: in all four directions the distance the agent can "see", eg. if there is a wall next to the agent
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

    return distances


def nearest_safe_tile(game_state, dist, grad, in_danger):
    """Idea: when an agent finds itself in the explosion radius of a bomb, this should point the agent
    in the direction of the nearest safe tile, especially useful for avoiding placing a bomb and then
    walking in a dead end waiting for the bomb to blow up. Should also work for escaping enemy bombs."""
    safe_tiles = []
    explosion_map = danger_map(game_state, dist)
    if not in_danger:
        return [0, 0, 0, 0]

    # Look for the nearest safe tile
    for x in range(game_state['field'].shape[0]):
        for y in range(game_state['field'].shape[1]):
            if explosion_map[x, y] == 0 and dist[x, y] != INF:
                safe_tiles.append((x, y))

    if not safe_tiles:
        return [0, 0, 0, 0]  # No safe tiles found

    nearest_tile = nearest_objects(dist, safe_tiles)

    direction = direction_to_object(nearest_tile, grad)

    direction_one_hot = [
        int(direction == dRIGHT),
        int(direction == dDOWN),
        int(direction == dLEFT),
        int(direction == dUP),
    ]
    return direction_one_hot


