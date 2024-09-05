import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import numpy as np
import torch.nn.functional as f
from collections import deque
import random
from heapq import heapify, heappop, heappush

from sympy.codegen.cnodes import sizeof


class DeepQNetwork(nn.Module):
    def __init__(self, lr, dims, dropout_rate=0.3):
        super(DeepQNetwork, self).__init__()
        # torch.autograd.set_detect_anomaly(True)  # remove when it works

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv = nn.ModuleList([
            nn.Conv2d(dims['channels'], 16, 3, 1, 1),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.Conv2d(8, 4, 3, 1, 1),
        ])

        self.linear = nn.ModuleList([
            nn.Linear(1157, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 12),
        ])
        self.out = nn.Linear(12, dims['out'])  # There are always six possible actions

        self.optimizer = optim.Adam(self.parameters(), lr=lr)                   # see https://pytorch.org/docs/stable/optim.html#algorithms
        self.scheduler = scheduler.ExponentialLR(self.optimizer, gamma=0.98)    # see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("GPU not found!")
        self.to(self.device)

    def forward(self, state, train):
        x = state['conv_features']
        for conv in self.conv:
            x = f.relu(conv(x))
            #x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, state['lin_features']), 1)

        for lin in self.linear:
            x = f.relu(lin(x))
            if train:
                x = self.dropout(x)

        actions = self.out(x)
        return actions


class Agent:
    def __init__(self, logger, gamma=0.9, lr=1e-4, epsilon=1.0, eps_end=0.1, eps_dec=1e-2,
                 batch_size=64, symmetries=8, dims={'channels': 9,'height': 17, 'width': 17, 'linear': 1, 'out': 6}, max_mem_size=100000):
        self.logger = logger
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.symmetries = symmetries

        self.Q_eval = DeepQNetwork(self.lr, dims)

        self.state_memory = {
            'conv_features': np.zeros((self.mem_size, self.symmetries, dims['channels'], dims['height'], dims['width']), dtype=np.float32),
            'lin_features': np.zeros((self.mem_size, self.symmetries, dims['linear']), dtype=np.float32),
        }
        self.new_state_memory = {
            'conv_features': np.zeros((self.mem_size, self.symmetries, dims['channels'], dims['height'], dims['width']), dtype=np.float32),
            'lin_features': np.zeros((self.mem_size, self.symmetries, dims['linear']), dtype=np.float32),
        }
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.mem_cntr = 0
        self.iteration = 0
        self.epoch = 0

    def store_transition(self, state, action, reward, new_state, done):  # state means features here
        index = self.mem_cntr % self.mem_size

        self.state_memory['conv_features', index], self.state_memory['lin_features', index] = state_symmetries(state)
        self.new_state_memory['conv_features', index], self.new_state_memory['lin_features', index] = state_symmetries(new_state) if new_state is not None else state_symmetries(state)
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, state, train):

        if np.random.random() > self.epsilon or not train:
            state = {
                'conv_features': torch.tensor(np.array([state['conv_features']]), dtype=torch.float32).to(self.Q_eval.device),
                'lin_features': torch.tensor(np.array([state['lin_features']]), dtype=torch.float32).to(self.Q_eval.device),
            }
            actions = self.Q_eval.forward(state, train).squeeze()

            action = torch.argmax(actions).item()

            # action_probabilities = actions.clone().detach().softmax(dim=1).squeeze()
            # action = torch.multinomial(action_probabilities, 1).item()

        else:
            p = [.20, .20, .20, .20, .10, .10]

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

        sym = np.random.choice(range(self.symmetries))

        state_batch = {
            'conv_features': torch.tensor(self.state_memory['conv_features'][batch, sym]).to(self.Q_eval.device),
            'lin_features': torch.tensor(self.state_memory['lin_features'][batch, sym]).to(self.Q_eval.device),
        }
        new_state_batch = {
            'conv_features': torch.tensor(self.new_state_memory['conv_features'][batch, sym]).to(self.Q_eval.device),
            'lin_features': torch.tensor(self.new_state_memory['lin_features'][batch, sym]).to(self.Q_eval.device),
        }
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

def state_symmetries(state):
    conv_features = []
    for feature in state['conv_features']:
        symmetries = [ # see https://www2.math.upenn.edu/~kazdan/202F13/notes/symmetries-square.pdf
            feature,                            # I
            np.rot90(feature, k=1),             # R
            np.rot90(feature, k=2),             # R2
            np.rot90(feature, k=-1),            # R3
            np.fliplr(feature),                 # S
            np.fliplr(np.rot90(feature, k=1)),  # SR
            np.flipud(feature),                 # SR2
            np.flipud(np.rot90(feature, k=1)),  # SR3
        ]
        conv_features.append(symmetries)

    lin_features = []
    dir_mask = [False, True]
    for idx, feature in enumerate(state['lin_features']):
        if dir_mask[idx]:
            symmetries = [
                feature,                        # I
                rot90(feature, k=1),            # R
                rot90(feature, k=2),            # R2
                rot90(feature, k=-1),           # R3
                fliplr(feature),                # S
                fliplr(rot90(feature, k=1)),    # SR
                flipud(feature),                # SR2
                flipud(rot90(feature, k=1)),    # SR3
            ]
        else:
            symmetries = [feature] * 8

        lin_features.append(symmetries)

    conv_features = np.swapaxes(np.array(conv_features), 0, 1)
    lin_features = np.swapaxes(np.array(lin_features), 0, 1)

    return conv_features, lin_features

# Direction Mapping: invert direction by multiplying with -1
dUP = -2
dLEFT = -1
dNONE = 0
dRIGHT = 1
dDOWN = 2

def rot90(direction, k=1):
    for i in range(k % 4):
        if direction == dUP:
            direction = dRIGHT
        elif direction == dRIGHT:
            direction = dDOWN
        elif direction == dDOWN:
            direction = dLEFT
        elif direction == dLEFT:
            direction = dUP
        else:
            return dNONE

    return direction

def fliplr(direction):
    if direction == dLEFT:
        return dRIGHT
    elif direction == dRIGHT:
        return dLEFT
    else:
        return direction

def flipud(direction):
    if direction == dUP:
        return dDOWN
    elif direction == dDOWN:
        return dUP
    else:
        return direction

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

    conv_features = []
    lin_features = []

    _, _, bomb_available, (ax, ay) = game_state['self']
    others = [(x, y) for _, _, _, (x, y) in game_state['others']]
    arena = game_state['field']

    dist, grad, direction, crates = distance_map(ax, ay, arena)
    up, down, left, right = onehot_encode_direction(direction)
    coins = coin_map(game_state['coins'], arena)
    players = player_map(ax, ay, others, arena)
    danger = danger_map(game_state['bombs'], game_state['explosion_map'], dist)

    conv_features.append(dist)
    #conv_features.append(up)
    #conv_features.append(down)
    #conv_features.append(left)
    #conv_features.append(right)
    #conv_features.append(crates)
    #conv_features.append(coins)
    #conv_features.append(players)
    #conv_features.append(danger)

    lin_features.append(int(bomb_available))
    #lin_features.extend(last_k_actions())

    return {'conv_features': conv_features, 'lin_features': lin_features}

# Use Dijkstra to calculate distance to every position and corresponding gradient
def distance_map(ax, ay, arena):
    dist = np.full_like(arena, INF)
    grad = np.full_like(arena, dNONE)
    scanned = np.full_like(arena, False)

    crates = np.full_like(arena, INF)

    direction = np.full_like(arena, dNONE)
    direction[ax, ay + 1] = dDOWN
    direction[ax, ay - 1] = dUP
    direction[ax + 1, ay] = dLEFT
    direction[ax - 1, ay] = dRIGHT

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
            if arena[x+dx, y+dy] != -1:
                # relax node
                if d + 1 < dist[x+dx, y+dy]:
                    dist[x+dx, y+dy] = d + 1
                    grad[x+dx, y+dy] = -dx - 2*dy
                    direction[x+dx, y+dy] = direction[x, y]
                    crates[x+dx, y+dy] = crates[x, y] + arena[x+dx, y+dy]
                    heappush(pq, (d + 1, (x+dx, y+dy)))

    return dist, grad, direction, crates

def onehot_encode_direction(direction):
    up = [[int(d == dUP) for d in line] for line in direction]
    down = [[int(d == dDOWN) for d in line] for line in direction]
    left = [[int(d == dLEFT) for d in line] for line in direction]
    right = [[int(d == dRIGHT) for d in line] for line in direction]

    return up, down, left, right

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

# Sort coins by distance and get the k nearest ones
def nearest_objects(dist, objects, k=1):
    objects.sort(key=lambda obj: dist[obj[0], obj[1]])
    objects = objects[:k]

    while len(objects) < k:
        objects.append((-1, -1))

    return objects

def danger_map(bombs, explosion_map, dist):
    explosion_radius = 3  # Bombs affect up to 3 fields in each direction

    for (bx, by), t in bombs: # for every bomb
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # in every direction
            for r in range(explosion_radius + 1): # from 0 to 3
                if dist[bx + r*dx, by + r*dy] == INF: # explosion blocked by wall
                    break
                explosion_map[bx + r*dx, by + r*dy] = max(t + 3, explosion_map[bx + r*dx, by + r*dy])

    return explosion_map

def player_map(ax, ay, others, arena):
    players = np.full_like(arena, 0)
    players[ax, ay] = 1
    for (x, y) in others:
        players[x, y] = -1

    return players

def coin_map(coins, arena):
    map = np.full_like(arena, 0)
    for (x, y) in coins:
        map[x, y] = 1

    return map


last_actions_deque = deque(maxlen=4)
def last_k_actions():
    return list(last_actions_deque) + [4] * (last_actions_deque.maxlen - len(last_actions_deque))  # padded with WAIT
