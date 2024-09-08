import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import numpy as np
from collections import deque
import random
from heapq import heapify, heappop, heappush
from .utils import Direction, Distance

class DeepQNetwork(nn.Module):
    def __init__(self, dims, dropout_rate=0.3, lr=1e-4):
        """
        Defines the models' architecture:

        - Convolutional Part: Series of convolutional layers that process 2D data ('conv_features') from the map or an area around the agent
        - Linear Part: Series of fully connected layers that process the flattened output of the Convolutional Part, combined with some linear features ('lin_features')

        self.conv:      Convolutional layers as a ModuleList
        self.linear:    Linear layers as a ModuleList
        self.out:       Last linear layer using softmax instead of default activation function

        self.pool:      Pooling layer which is applied after every convolutional layer
        self.dropout:   Dropout layer which is applied to every linear layer during training
        self.act:       Activation function applied to every convolutional and linear layer

        self.optimizer: Momentum based learning, see https://pytorch.org/docs/stable/optim.html#algorithms
        self.scheduler: Adjusts lr over time, see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

        :param dims:            Input/Output dimensions defined in Agent
        :param dropout_rate:    Rate at which neurons are randomly disabled during training to make model more stable
        :param lr:              Rate at which model adapts to data, see scheduler for changing lr over time

        NOTE: after changing the properties of the input or some convolutional layers the value of `conv_output_dimension` may become incorrect
        raising an error that some matrices' shapes don't match. To fix this change the value of `conv_output_dimension` accordingly
        """
        super(DeepQNetwork, self).__init__()
        # torch.autograd.set_detect_anomaly(True)  # remove when it works

        self.conv = nn.ModuleList([
            nn.Conv2d(dims['channels'], 16, 3, 1, 1),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.Conv2d(8, 4, 3, 1, 1),
        ])

        conv_output_dimension = 1156 # -> run model once, then change value according to error message

        self.linear = nn.ModuleList([
            nn.Linear(conv_output_dimension + dims['linear'], 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 12),
        ])
        self.out = nn.Linear(12, dims['out'])

        self.pool = nn.MaxPool2d(1, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.act = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("GPU not found!")
        self.to(self.device)

    def forward(self, state, train):
        """
        Run a forward pass of the model

        :param state:   Features of the current game state
        :param train:   Whether training mode is enabled
        :return:        Decision of the model of which action to take as weighted vector
        """
        x = state['conv_features']
        for conv in self.conv:
            x = self.act(conv(x))
            x = self.pool(x)

        # flatten data and append linear features
        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, state['lin_features']), 1)

        for lin in self.linear:
            x = self.act(lin(x))
            if train:
                x = self.dropout(x)

        actions = self.out(x)
        return actions


class Agent:
    def __init__(self, logger, gamma=0.9, epsilon=1.0, eps_dec=1e-2, eps_end=0.1,
                 batch_size=64, symmetries=8, mem_size=100000):
        """
        Agent with memory of previous games, a model, and some hyperparameters

        feature_dims: Dictionary defining the dimension of features for the model and memory
        self.Q_eval: Model that will be trained, see DeepQNetwork.__init__ for more information

        Memory (self.state_memory, self.new_state_memory, self.action_memory, self.reward_memory, self.terminal_memory):
        - Arrays storing every previous state, new_state, chosen action and reward, used for training the model
        - All arrays should have the same dimensions, and their elements should correspond to each other as if they were a single array of tuples
        - State is stored in terms of its features, split into convolutional and linear features

        :param logger:      Logger passed down from game framework
        :param gamma:       Hyperparameter; Devaluation of future rewards, and thus representing uncertainty of the future, see learn() method
        :param epsilon:     Hyperparameter; Balance between exploration and exploitation, see choose_action() method
        :param eps_dec:     Decrease of epsilon hyperparameter every epoch
        :param eps_end:     Final value of epsilon hyperparameter
        :param batch_size:  Size of batches used in learning, see learn() method
        :param symmetries:  Number of symmetries that are generated by state_symmetries method
        :param mem_size:    Limit of memorized transitions
        """
        self.logger = logger
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.symmetries = symmetries

        # initialize model
        feature_dims = { 'channels': 9, 'height': 17, 'width': 17, 'linear': 5, 'out': 6}
        self.Q_eval = DeepQNetwork(feature_dims)

        # initialize memory
        self.state_memory = {
            'conv_features': np.zeros((self.mem_size, self.symmetries, feature_dims['channels'], feature_dims['height'], feature_dims['width']), dtype=np.float32),
            'lin_features': np.zeros((self.mem_size, self.symmetries, feature_dims['linear']), dtype=np.float32),
        }
        self.new_state_memory = {
            'conv_features': np.zeros((self.mem_size, self.symmetries, feature_dims['channels'], feature_dims['height'], feature_dims['width']), dtype=np.float32),
            'lin_features': np.zeros((self.mem_size, self.symmetries, feature_dims['linear']), dtype=np.float32),
        }
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        # initialize counters
        self.mem_cntr = 0
        self.iteration = 0
        self.epoch = 0

    def store_transition(self, state, new_state, action, reward, done):
        """
        Store a single state-action transition into memory
        After memory size exceeds limit new transitions will fill up from the beginning
        As always state is given in terms of its features, split into convolutional and linear features

        :param state:       State before action was taken
        :param new_state:   State after action was taken
        :param action:      Action taken by the Agent
        :param reward:      Reward given in this instance
        :param done:        Whether the transition ended the game
        """
        index = self.mem_cntr % self.mem_size

        self.state_memory['conv_features', index], self.state_memory['lin_features', index] = state_symmetries(state)
        self.new_state_memory['conv_features', index], self.new_state_memory['lin_features', index] = state_symmetries(new_state) if new_state is not None else state_symmetries(state)
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, state, train):
        """
        Use model to decide on the optimal decision or choose a random exploratory action during training

        self.epsilon: Probability that a random action will be chosen, defined in Agents constructor

        :param state:   State the chosen action should act on
        :param train:   Whether training mode and with that exploration is enabled
        :return:        Chosen action as index in ACTION array defined in utils.py
        """
        # epsilon greed
        if np.random.random() < self.epsilon and train:
            p = [1, 1, 1, 1, .5, .5]

            self.logger.info("Chose random action (Eps = " + str(self.epsilon) + ")")
            return random.choices(range(6), weights=p, k=1)[0]


        # transfer state to GPU
        state = {
            'conv_features': torch.tensor(np.array([state['conv_features']]), dtype=torch.float32).to(self.Q_eval.device),
            'lin_features': torch.tensor(np.array([state['lin_features']]), dtype=torch.float32).to(self.Q_eval.device),
        }
        # use model to evaluate actions, then pick optimal action
        actions = self.Q_eval.forward(state, train).squeeze()
        action = torch.argmax(actions).item()

        # action_probabilities = actions.clone().detach().softmax(dim=1).squeeze()
        # action = torch.multinomial(action_probabilities, 1).item()

        #last_actions_deque.append(action)
        return action

    def learn(self, end_epoch=False):
        """
        Use gradient descent to improve Q Function based on game experience from memory

        A random continuous batch of memory data is picked and a random symmetry applied
        Based on this batch the Q Function is adjusted

        self.batch_size:    Size of a continuous batch
        self.iteration:     Counter for learning iterations, will be incremented every time this function is called
        self.epoch:         Counter of epochs, usually representing the number of games played

        :param end_epoch:   Whether iteration marks the end of an epoch, thus adjusting learning parameters
        """
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        # choose random batch
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # choose symmetry
        sym = np.random.choice(range(self.symmetries))

        # transfer states to GPU
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

        # estimate value of original and resulting state
        q_eval = self.Q_eval.forward(state_batch, True)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch, True)

        # use actual reward to improve evaluation of original state
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        # use gradient descent to adjust model
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()

        # trigger optimizer step
        self.Q_eval.optimizer.step()

        # increment counters
        self.iteration += 1
        if end_epoch:
            self.epoch += 1

            # adjust learning parameters
            #self.Q_eval.scheduler.step()
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        self.logger.info("LR = " + str(self.Q_eval.scheduler.get_last_lr()))

        # followed https://www.youtube.com/watch?v=wc-FxNENg9U


def state_symmetries(state):
    """
    Transform a single state into an array of symmetric states for data augmentation

    Symmetries can be added or removed by editing all three for-loops, num_symmetries variable and Agent.symmetries;
    these places have to be consistent with each other

    2D features are automatically transformed, but for convolutional or linear features encoding direction as an
    onehot encoding the directions have to be mapped in the corresponding onehot_maps list to the index of the feature

    Implemented are the 8 square symmetries, see https://www2.math.upenn.edu/~kazdan/202F13/notes/symmetries-square.pdf

    :param state:   State being transformed
    :return:        Arrays of transformed features
    """
    # -> adjust number of symmetries here and in Agent.__init__()
    num_symmetries = 8

    conv_features = []
    for feature in state['conv_features']:
        # -> edits symmetries here
        symmetries = [
            feature,                        # I
            np.rot90(feature, k=1),         # R
            np.rot90(feature, k=2),         # R2
            np.rot90(feature, k=-1),        # R3
            np.fliplr(feature),             # S
            np.fliplr(np.rot90(feature)),   # SR
            np.flipud(feature),             # SR2
            np.flipud(np.rot90(feature)),   # SR3
        ]
        conv_features.append(symmetries)

    onehot_maps = [
        # -> define map onehot encoded directions of convolutional features here
        {Direction.UP: 1, Direction.DOWN: 2, Direction.LEFT: 3, Direction.RIGHT: 4},
    ]
    for onehot_map in onehot_maps:
        feature_buffer = {
            Direction.UP: conv_features[onehot_map[Direction.UP]],
            Direction.DOWN: conv_features[onehot_map[Direction.DOWN]],
            Direction.LEFT: conv_features[onehot_map[Direction.LEFT]],
            Direction.RIGHT: conv_features[onehot_map[Direction.RIGHT]],
        }

        for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
            # -> edits symmetries here
            conv_features[onehot_map[direction]][0] = feature_buffer[ direction                                     ][0]
            conv_features[onehot_map[direction]][1] = feature_buffer[ Direction.rot90(direction, k=1)               ][1]
            conv_features[onehot_map[direction]][2] = feature_buffer[ Direction.rot90(direction, k=2)               ][2]
            conv_features[onehot_map[direction]][3] = feature_buffer[ Direction.rot90(direction, k=-1)              ][3]
            conv_features[onehot_map[direction]][4] = feature_buffer[ Direction.fliplr(direction)                   ][4]
            conv_features[onehot_map[direction]][5] = feature_buffer[ Direction.fliplr(Direction.rot90(direction))  ][5]
            conv_features[onehot_map[direction]][6] = feature_buffer[ Direction.flipud(direction)                   ][6]
            conv_features[onehot_map[direction]][7] = feature_buffer[ Direction.flipud(Direction.rot90(direction))  ][7]

    # linear features that don't encode directional information don't have to be transformed
    lin_features = []
    for feature in state['lin_features']:
        symmetries = [feature] * num_symmetries
        lin_features.append(symmetries)

    onehot_maps = [
        # -> define map onehot encoded directions of linear features here
        {Direction.UP: 1, Direction.DOWN: 2, Direction.LEFT: 3, Direction.RIGHT: 4},
    ]
    for onehot_map in onehot_maps:
        feature_buffer = {
            Direction.UP: lin_features[onehot_map[Direction.UP]],
            Direction.DOWN: lin_features[onehot_map[Direction.DOWN]],
            Direction.LEFT: lin_features[onehot_map[Direction.LEFT]],
            Direction.RIGHT: lin_features[onehot_map[Direction.RIGHT]],
        }

        for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
            # -> edits symmetries here
            lin_features[onehot_map[direction]][0]  = feature_buffer[ direction                                     ][0]
            lin_features[onehot_map[direction]][1]  = feature_buffer[ Direction.rot90(direction, k=1)               ][1]
            lin_features[onehot_map[direction]][2]  = feature_buffer[ Direction.rot90(direction, k=2)               ][2]
            lin_features[onehot_map[direction]][3]  = feature_buffer[ Direction.rot90(direction, k=-1)              ][3]
            lin_features[onehot_map[direction]][4]  = feature_buffer[ Direction.fliplr(direction)                   ][4]
            lin_features[onehot_map[direction]][5]  = feature_buffer[ Direction.fliplr(Direction.rot90(direction))  ][5]
            lin_features[onehot_map[direction]][6]  = feature_buffer[ Direction.flipud(direction)                   ][6]
            lin_features[onehot_map[direction]][7]  = feature_buffer[ Direction.flipud(Direction.rot90(direction))  ][7]

    # swap the axes of the features to match array shape of agents memory
    conv_features = np.swapaxes(np.array(conv_features), 0, 1)
    lin_features = np.swapaxes(np.array(lin_features), 0, 1)

    return conv_features, lin_features

def state_to_features(game_state: dict) -> np.array:
    """
    Extract useful features from a game state

    Features are split between convolutional and linear features

    The distance_map() function is used to scan the arena and generate maps containing distance and direction information,
    these maps are then used to calculate information rich features

    :param game_state:  A dictionary describing the current game board.
    :return:            Dictionary with both 'conv_features' and 'lin_features'
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    conv_features = []
    lin_features = []

    # basic game_state information
    _, _, bomb_available, (ax, ay) = game_state['self']
    others = [(x, y) for _, _, _, (x, y) in game_state['others']]
    arena = game_state['field']

    # more complex maps containing different information
    distance, direction, crates = distance_map(ax, ay, arena)
    coins = coin_map(game_state['coins'], arena)
    players = player_map(ax, ay, others, arena)
    danger = danger_map(distance, game_state['bombs'], game_state['explosion_map'])

    # adding features
    conv_features.append(distance)
    conv_features.extend(onehot_encode_direction_map(direction))
    conv_features.append(crates)
    conv_features.append(coins)
    conv_features.append(players)
    conv_features.append(danger)

    lin_features.append(int(bomb_available))
    lin_features.extend(onehot_encode_direction(Direction.UP))
    #lin_features.extend(last_k_actions())

    return { 'conv_features': conv_features, 'lin_features': lin_features }


def distance_map(ax, ay, arena):
    """
    Use Dijkstra to calculate distance to every position and corresponding gradient

    distance:       Map containing the distance to the player using the shortest path
    direction:  Map containing the direction the player would have to move to reach a position
    crates:     Map containing the amount of crates between a position and a player

    :param ax:      Players x coordinate
    :param ay:      Players y coordinate
    :param arena:   Full arena data
    :return:        Distance, Gradient, Direction and Crate maps
    """
    distance = np.full_like(arena, Distance.INFINITE)
    scanned = np.full_like(arena, False)

    crates = np.full_like(arena, Distance.INFINITE)

    direction = np.full_like(arena, Direction.NONE)
    direction[ax, ay + 1] = Direction.DOWN
    direction[ax, ay - 1] = Direction.UP
    direction[ax + 1, ay] = Direction.LEFT
    direction[ax - 1, ay] = Direction.RIGHT

    distance[ax, ay] = 0
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
                if d + 1 < distance[x+dx, y+dy]:
                    distance[x+dx, y+dy] = d + 1
                    direction[x+dx, y+dy] = direction[x, y]
                    crates[x+dx, y+dy] = crates[x, y] + arena[x+dx, y+dy]
                    heappush(pq, (d + 1, (x+dx, y+dy)))

    return distance, direction, crates

def onehot_encode_direction_map(direction_map):
    """
    Onehot encode a direction map

    :param direction_map:   Direction map to be encoded
    :return:                Onehot encoded maps for every direction
    """
    up      = [[int(direction == Direction.UP) for direction in line] for line in direction_map]
    down    = [[int(direction == Direction.DOWN) for direction in line] for line in direction_map]
    left    = [[int(direction == Direction.LEFT) for direction in line] for line in direction_map]
    right   = [[int(direction == Direction.RIGHT) for direction in line] for line in direction_map]

    return up, down, left, right

def onehot_encode_direction(direction):
    """
    Onehot encode a direction

    :param direction:   Direction map to be encoded
    :return:            Onehot encoding for every direction
    """
    up      = int(direction == Direction.UP)
    down    = int(direction == Direction.DOWN)
    left    = int(direction == Direction.LEFT)
    right   = int(direction == Direction.RIGHT)

    return up, down, left, right

def nearest_objects(distance, objects, k=1):
    """
    Get the k objects closest to the player

    :param distance:    Distance map from distance_map() function
    :param objects:     List of object positions being considered
    :param k:           Number of objects to be returned
    :return:            Slice of the original objects list potentially filled up with '-1's if not enough objects
    """
    objects.sort(key=lambda obj: distance[obj[0], obj[1]])
    objects = objects[:k]

    while len(objects) < k:
        objects.append((-1, -1))

    return objects

def danger_map(distance, bombs, explosion_map):
    """
    Map of potentially dangerous positions

    value of 0:     Safe position
    value of 1-2:   Active explosion
    value of 3-6:   Bomb is about to explode

    :param distance:        Distance map from distance_map() function
    :param bombs:           List of bombs with their metadata
    :param explosion_map:   Map of currently ongoing explosions
    :return:                Updated Explosion Map
    """
    explosion_radius = 3  # Bombs affect up to 3 fields in each direction

    for (bx, by), t in bombs: # for every bomb
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # in every direction
            for r in range(explosion_radius + 1): # from 0 to 3
                if distance[bx + r*dx, by + r*dy] == Distance.INFINITE: # explosion blocked by wall
                    break
                explosion_map[bx + r*dx, by + r*dy] = max(t + 3, explosion_map[bx + r*dx, by + r*dy])

    return explosion_map

def player_map(ax, ay, others, arena):
    """
    Map of players

    value  0:       No player
    value  1:       Our agent
    value -1:       Enemy player

    :param ax:      Players x coordinate
    :param ay:      Players y coordinate
    :param others:  List of other players positions
    :param arena:   Full arena data
    :return:        Map of all players
    """
    players = np.full_like(arena, 0)
    players[ax, ay] = 1
    for (x, y) in others:
        players[x, y] = -1

    return players

def coin_map(coin_positions, arena):
    """
    Map of coins

    value 0:                No coin
    value 1:                Coin

    :param coin_positions:  List of coin positions
    :param arena:           Full arena data
    :return:                Map of all coins
    """
    coins = np.full_like(arena, 0)
    for (x, y) in coin_positions:
        coins[x, y] = 1

    return coins


last_actions_deque = deque(maxlen=4)
def last_k_actions():
    return list(last_actions_deque) + [4] * (last_actions_deque.maxlen - len(last_actions_deque))  # padded with WAIT
