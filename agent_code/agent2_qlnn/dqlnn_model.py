from venv import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import numpy as np
import torch.nn.functional as f
from collections import deque
import random
from heapq import heapify, heappop, heappush

from fontTools.misc.timeTools import epoch_diff
from sympy import false

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, l1_dims, l2_dims, l3_dims, l4_dims, dropout_rate=0.3):
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
        self.l3 = nn.Linear(self.l2_dims, self.l3_dims)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.l4 = nn.Linear(self.l3_dims, self.l4_dims)
        self.dropout4 = nn.Dropout(p=dropout_rate)
        self.lo = nn.Linear(self.l4_dims, 6)  # There are always six possible actions

        # Optimizer (see https://pytorch.org/docs/stable/optim.html#algorithms)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # LR Scheduler (see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
        self.scheduler = scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        #self.scheduler = scheduler.ReduceLROnPlateau(self.optimizer, ...)

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
        x = f.relu(self.l3(x))
        if train:
            x = self.dropout3(x)
        x = f.relu(self.l4(x))
        if train:
            x = self.dropout4(x)
        actions = self.lo(x)
        # actions_prob = torch.sigmoid_(actions)
        # softmax/sigmoid causes really weird errors here
        return actions


class Agent:
    def __init__(self, logger, gamma, epsilon, lr, input_dims, batch_size, max_mem_size=100000, eps_end=0.1,
                 eps_dec=1e-2):
        self.logger = logger
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size

        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, l1_dims=256, l2_dims=256,
                                   l3_dims=256, l4_dims=128)  # experiment here

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.mem_cntr = 0
        self.iteration = 0
        self.epoch = 0

    def store_transition(self, state, action, reward, state_, done):  # state means features here
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        if state_ is not None:
            self.new_state_memory[index] = state_
        else:
            self.new_state_memory[index] = state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, game_state, train):
        features = state_to_features(game_state)

        #blocked = features[3:7]
        #bomb_available = bool(features[2])
        #in_explosion_radius = bool(features[3])
        #if in_explosion_radius:
        #    blocked.append(1)  # do not wait in bomb radius
        #else:
        #    blocked.append(0)
        #if not bomb_available:
        #    blocked.append(1)
        #else:
        #    blocked.append(0)

        if np.random.random() > self.epsilon or not train:
            state = torch.tensor([features], dtype=torch.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state, train).squeeze()

            #for i in range(6):
            #    if blocked[i] == 1:
            #        actions[i] = -9999

            action = torch.argmax(actions).item()

            # action_probabilities = actions.clone().detach().softmax(dim=1).squeeze()
            # action = torch.multinomial(action_probabilities, 1).item()

        else:
            p = [.20, .20, .20, .20, .10, .10]

            #for i in range(6):
            #    if blocked[i] == 1:
            #        p[i] = 0

            total_prob = np.sum(p)
            p /= total_prob

            action = random.choices(range(6), weights=p, k=1)[0]
            self.logger.info("Chose random action (Eps = " + str(self.epsilon) + ")")

        #last_actions_deque.append(action)
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

            #self.Q_eval.scheduler.step()
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        self.logger.info("LR = " + str(self.Q_eval.scheduler.get_last_lr()))


# followed https://www.youtube.com/watch?v=wc-FxNENg9U

# Direction Mapping: invert direction by multiplying with -1
dUP = -2
dLEFT = -1
dNONE = 0
dRIGHT = 1
dDOWN = 2

INF = 999 # 999 represents infinity

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

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

    _, _, bomb_available, (ax, ay) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins'] # [(x, y)]
    bombs = game_state['bombs'] # [(x, y), countdown]

    dist, grad = distance_map(ax, ay, arena)

    #features.append(ax)
    #features.append(ay)
    #features.append(int(bomb_available))

    #blocked = is_blocked(game_state)
    #features.extend(blocked)

    #in_explosion_radius = is_in_explosion_radius(game_state, ax, ay)
    #features.append(in_explosion_radius)

    nearest_coins = nearest_objects(dist, coins)
    for coin in nearest_coins:
        direction = direction_to_object(coin, grad)
        direction_one_hot = [
            int(direction == dRIGHT),
            int(direction == dDOWN),
            int(direction == dLEFT),
            int(direction == dUP),
        ]
        features.extend(direction_one_hot)

    #bomb_positions = relative_bomb_positions(game_state, ax, ay)
    #features.extend(bomb_positions)

    #features.extend(last_k_actions())

    # todo:
    # k nearest crates
    # direction to nearest crate
    # 4 bomb positions relative
    # is in explosion radius (with bomb timer, inverse intensity)
    # is at outer wall
    # the last k actions
    # positions of other agents
    # embedding of entire game state?

    return features  # length: 1+1+1+4+6+4+12+1+4 = 32

# Use Dijkstra to calculate distance to every position and corresponding gradient
def distance_map(ax, ay, arena):
    dist = np.full_like(arena, INF)
    grad = np.full_like(arena, dNONE)
    scanned = np.full_like(arena, False)

    dist[ax, ay] = 0

    pq = [(0, (ax, ay))]
    heapify(pq)

    while pq:
        d, (x, y) = heappop(pq)

        if scanned[x, y]:
            continue
        scanned[x, y] = True

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if arena[x+dx, y+dy] == 0:
                # relax node
                if d + 1 < dist[x+dx, y+dy]:
                    dist[x+dx, y+dy] = d + 1
                    grad[x+dx, y+dy] = -dx - 2*dy
                    heappush(pq, (d + 1, (x+dx, y+dy)))

    return dist, grad


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


# Sort coins by distance and get the k nearest ones
def nearest_objects(dist, objects, k=1):
    objects.sort(key=lambda obj: dist[obj[0], obj[1]])
    objects = objects[:k]

    while len(objects) < k:
        objects.append((-1, -1))

    return objects

# Follow gradient to agent, to determine path
def direction_to_object(obj, grad):
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


def is_in_explosion_radius(game_state, ax, ay):
    bombs = game_state['bombs']
    arena = game_state['field']  # arena[x, y] == -1 indicates a wall
    explosion_radius = 3  # Bombs affect up to 3 fields in each direction

    for (bx, by), countdown in bombs:
        if bx == ax and abs(by - ay) <= explosion_radius:
            # Check for walls blocking the vertical explosion
            blocked = any(arena[bx, y] == -1 for y in range(min(by, ay) + 1, max(by, ay)))
            if not blocked:
                return 1

        if by == ay and abs(bx - ax) <= explosion_radius:
            # Check for walls blocking the horizontal explosion
            blocked = any(arena[x, by] == -1 for x in range(min(bx, ax) + 1, max(bx, ax)))
            if not blocked:
                return 1

    return 0


last_actions_deque = deque(maxlen=4)


def last_k_actions():
    return list(last_actions_deque) + [4] * (last_actions_deque.maxlen - len(last_actions_deque))  # padded with WAIT
