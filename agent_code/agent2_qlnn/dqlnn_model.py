import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import numpy as np
from collections import deque
import random
from heapq import heapify, heappop, heappush
from .utils import Direction, Distance, Action


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
                 batch_size=64, mem_size=100000):
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
        :param mem_size:    Limit of memorized transitions
        """
        self.logger = logger
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.mem_size = mem_size
        self.batch_size = batch_size

        # initialize model
        feature_dims = { 'channels': 9, 'height': 17, 'width': 17, 'linear': 1, 'out': 6}
        self.Q_eval = DeepQNetwork(feature_dims)

        # initialize memory
        self.state_memory = {
            'conv_features': np.zeros((self.mem_size, feature_dims['channels'], feature_dims['height'], feature_dims['width']), dtype=np.float32),
            'lin_features': np.zeros((self.mem_size, feature_dims['linear']), dtype=np.float32),
        }
        self.new_state_memory = {
            'conv_features': np.zeros((self.mem_size, feature_dims['channels'], feature_dims['height'], feature_dims['width']), dtype=np.float32),
            'lin_features': np.zeros((self.mem_size, feature_dims['linear']), dtype=np.float32),
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
        if new_state is None:
            new_state = state

        index = self.mem_cntr % self.mem_size

        self.state_memory['conv_features', index]       = state['conv_features']
        self.state_memory['lin_features',  index]       = state['lin_features' ]
        self.new_state_memory['conv_features', index]   = new_state['conv_features']
        self.new_state_memory['lin_features',  index]   = new_state['lin_features' ]
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

        last_actions_deque.append(action)
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

        # get corresponding state batch
        state_batch     = { 'conv_features': self.state_memory['conv_features'][batch],     'lin_features': self.state_memory['lin_features'][batch]        }
        new_state_batch = { 'conv_features': self.new_state_memory['conv_features'][batch], 'lin_features': self.new_state_memory['lin_features'][batch]    }
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        # apply symmetry
        state_batch, new_state_batch, action_batch = state_symmetries(state_batch, new_state_batch, action_batch)

        # transfer states to GPU
        state_batch = {
            'conv_features': torch.tensor(np.array(state_batch['conv_features'])).to(self.Q_eval.device),
            'lin_features': torch.tensor(np.array(state_batch['lin_features'])).to(self.Q_eval.device),
        }
        new_state_batch = {
            'conv_features': torch.tensor(np.array(new_state_batch['conv_features'])).to(self.Q_eval.device),
            'lin_features': torch.tensor(np.array(new_state_batch['lin_features'])).to(self.Q_eval.device),
        }
        reward_batch = torch.tensor(reward_batch).to(self.Q_eval.device)
        terminal_batch = torch.tensor(terminal_batch).to(self.Q_eval.device)

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


def state_symmetries(state_batch, new_state_batch, action_batch):
    """
    Transform a batch of transitions into a batch of symmetric transitions for data augmentation
    Note that the same symmetry is applied to every state to ensure continuity between a series of states

    Reward is not given as an argument because a symmetric transitions will always give the same reward

    Symmetries can be added by adding new functions with the same signature to the symmetries list

    2D features are automatically transformed, but for convolutional or linear features encoding direction as an
    onehot encoding the directions have to be mapped in the corresponding onehot_maps list to the index of the feature

    Implemented are the 8 square symmetries, see https://www2.math.upenn.edu/~kazdan/202F13/notes/symmetries-square.pdf

    :param state_batch:     Batch of states before action was taken
    :param new_state_batch: Batch of states after action was taken
    :param action_batch:    Batch of actions taken by the Agent
    :return:                Transformed versions of the parameters
    """

    # -> define map onehot encoded directions of convolutional features here
    conv_onehot_maps = [
        {Direction.UP: 0, Direction.DOWN: 1, Direction.LEFT: 2, Direction.RIGHT: 3},
    ]

    # -> define map onehot encoded directions of linear features here
    lin_onehot_maps = [
        {Direction.UP:  5,  Direction.DOWN:  6, Direction.LEFT:  7, Direction.RIGHT:  8 },
        {Direction.UP:  9,  Direction.DOWN: 10, Direction.LEFT: 11, Direction.RIGHT: 12 },
        {Direction.UP: 13,  Direction.DOWN: 14, Direction.LEFT: 15, Direction.RIGHT: 16 },
        {Direction.UP: 17,  Direction.DOWN: 18, Direction.LEFT: 19, Direction.RIGHT: 20 },
        {Direction.UP: 21+Action.UP, Direction.DOWN: 21+Action.DOWN, Direction.LEFT: 21+Action.LEFT, Direction.RIGHT:  21+Action.RIGHT },
        {Direction.UP: 29,  Direction.DOWN: 30, Direction.LEFT: 31, Direction.RIGHT: 32},
        {Direction.UP: 33,  Direction.DOWN: 34, Direction.LEFT: 35, Direction.RIGHT: 36},
        {Direction.UP: 37,  Direction.DOWN: 38, Direction.LEFT: 39, Direction.RIGHT: 40},
        {Direction.UP: 42+Action.UP, Direction.DOWN: 42+Action.DOWN, Direction.LEFT: 42+Action.LEFT, Direction.RIGHT: 42+Action.RIGHT},
        {Direction.UP: 48,  Direction.DOWN: 49, Direction.LEFT: 50, Direction.RIGHT: 51},
        {Direction.UP: 53,  Direction.DOWN: 54, Direction.LEFT: 55, Direction.RIGHT: 56},
    ]

    # define symmetries
    def I(x, is_conv=False):
        return x

    def R(x, is_conv=False):
        if is_conv: return np.rot90(x, k=1)
        else:       return Direction.rot90(x, k=1)

    def R2(x, is_conv=False):
        if is_conv: return np.rot90(x, k=2)
        else:       return Direction.rot90(x, k=2)

    def R3(x, is_conv=False):
        if is_conv: return np.rot90(x, k=-1)
        else:       return Direction.rot90(x, k=-1)

    def S(x, is_conv=False):
        if is_conv: return np.fliplr(x)
        else:       return Direction.fliplr(x)

    def SR(x, is_conv=False):
        if is_conv: return np.fliplr(np.rot90(x, k=1))
        else:       return Direction.fliplr(Direction.rot90(x, k=1))

    def SR2(x, is_conv=False):
        if is_conv: return np.flipud(x)
        else:       return Direction.flipud(x)

    def SR3(x, is_conv=False):
        if is_conv: return np.flipud(np.rot90(x, k=1))
        else:       return Direction.flipud(Direction.rot90(x, k=1))

    # pick random symmetry
    symmetries = [ I, R, R2, R3, S, SR, SR2, SR3 ]
    sym = symmetries[np.random.choice(len(symmetries))]

    for batch in (state_batch, new_state_batch):
        # apply symmetry to convolutional features
        batch['conv_features'] = [[sym(feature, True) for feature in state] for state in batch['conv_features']]

        # swap features according to symmetry transformation
        for onehot_map in conv_onehot_maps:
            for features in batch['conv_features']:
                feature_buffer = {
                    Direction.UP:       features[onehot_map[Direction.UP    ]],
                    Direction.DOWN:     features[onehot_map[Direction.DOWN  ]],
                    Direction.LEFT:     features[onehot_map[Direction.LEFT  ]],
                    Direction.RIGHT:    features[onehot_map[Direction.RIGHT ]],
                }

                for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
                    features[onehot_map[direction]] = feature_buffer[sym(direction)]

        # do the same thing but for linear features
        for onehot_map in lin_onehot_maps:
            for features in batch['lin_features']:
                feature_buffer = {
                    Direction.UP:       features[onehot_map[Direction.UP    ]],
                    Direction.DOWN:     features[onehot_map[Direction.DOWN  ]],
                    Direction.LEFT:     features[onehot_map[Direction.LEFT  ]],
                    Direction.RIGHT:    features[onehot_map[Direction.RIGHT ]],
                }

                for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
                    features[onehot_map[direction]] = feature_buffer[sym(direction)]

    # apply symmetry to action if it is directional
    action_batch = np.array([
        action  if action == Action.WAIT or action == Action.BOMB
                else Direction.to_action(sym(Direction.from_action(action)))
        for action in action_batch
    ])

    return state_batch, new_state_batch, action_batch


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
    enemies = [(x, y) for _, _, _, (x, y) in game_state['others']]
    field = game_state['field']
    crates = [tuple(pos) for pos in np.argwhere(field == 1)]
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    danger_threshold = 1

    coordinate_history.append((ax, ay))

    # more complex maps containing different information
    distance_map, direction_map = get_distance_map(ax, ay, field, bombs, enemies)
    crate_map   = get_generic_map(crates, field)
    coin_map    = get_generic_map(coins, field)
    player_map  = get_player_map(ax, ay, enemies, field)
    danger_map  = get_danger_map(distance_map, bombs, explosion_map)
    dead_end_map, dead_end_list = get_dead_end_map(field, enemies)

    safety_direction = get_direction_to_safety(distance_map, direction_map, danger_map, danger_threshold)
    disallowed_actions = get_disallowed_actions(ax, ay, bomb_available, distance_map, direction_map, danger_map, danger_threshold, safety_direction)

    # adding features
    lin_features.append(bomb_available)                                                                     # 1
    lin_features.append(number_of_crates_reachable(ax, ay, field))                                          # 2
    lin_features.append(danger_map[ax, ay] > danger_threshold)                                              # 3
    lin_features.append(is_repeating_actions())                                                             # 4
    lin_features.extend(nearest_coins_feature(coins, distance_map, direction_map))                          # 5-8
    lin_features.extend(nearest_crates_feature(crates, distance_map, direction_map))                        # 9-12
    lin_features.extend(get_free_distances(ax, ay, field))                                                  # 13-16
    lin_features.extend(onehot_encode_direction(safety_direction))                                          # 17-20
    lin_features.extend(disallowed_actions)                                                                 # 21-26
    lin_features.append(float(len(crates) / 135))                                                           # 27
    lin_features.append(float(game_state['round'] * len(enemies)) / 400)                                    # 28
    lin_features.extend(enemy_directions_feature(distance_map, direction_map, enemies))                     # 29-40
    lin_features.append(is_at_crossing(ax, ay))                                                             # 41
    lin_features.extend(get_last_actions(k=1))                                                              # 42-47
    lin_features.extend(get_direction_to_trapped_enemy(dead_end_list, distance_map, direction_map))         # 48-51
    lin_features.append(is_next_to_enemy(ax, ay, enemies))                                                  # 52
    lin_features.extend(do_not_enter_dead_end(ax, ay, dead_end_map, dead_end_list, enemies, distance_map))  # 53-56

    conv_features.extend(onehot_encode_direction_map(direction_map))                                        # 1-4
    conv_features.append(distance_map)                                                                      # 5
    conv_features.append(crate_map)                                                                         # 6
    conv_features.append(coin_map)                                                                          # 7
    conv_features.append(player_map)                                                                        # 8
    conv_features.append(danger_map)                                                                        # 9

    # zoom maps to an area around the agent
    conv_features = [focus_map(ax, ay, full_map, r=-1) for full_map in conv_features]

    return { 'conv_features': conv_features, 'lin_features': lin_features }

last_actions_deque = deque(maxlen=100)
coordinate_history = deque([], maxlen=20)

def get_distance_map(ax, ay, field, bombs=None, enemies=None):
    """
    Use Dijkstra to calculate distance and direction to every position

    distance:       Map containing the distance to the player using the shortest path
    direction:      Map containing the direction the player would have to move to reach a position

    :param ax:      Players x coordinate
    :param ay:      Players y coordinate
    :param field:   Full field data
    :param bombs:   List of bomb positions
    :param enemies: List of enemy positions
    :return:        Distance, Gradient, Direction and Crate maps
    """

    blocked = [[row == -1 for value in row] for row in field]

    if bombs is not None:
        for bomb in bombs:
            blocked[bomb] = True

    if enemies is not None:
        for enemy in enemies:
            blocked[enemy] = True

    distance = np.full_like(field, Distance.INFINITE)
    scanned = np.full_like(field, False)

    direction = np.full_like(field, Direction.NONE)
    direction[ax, ay + 1] = Direction.DOWN  if not blocked[ax, ay + 1] else Direction.NONE
    direction[ax, ay - 1] = Direction.UP    if not blocked[ax, ay - 1] else Direction.NONE
    direction[ax + 1, ay] = Direction.LEFT  if not blocked[ax + 1, ay] else Direction.NONE
    direction[ax - 1, ay] = Direction.RIGHT if not blocked[ax - 1, ay] else Direction.NONE

    distance[ax, ay] = 0

    pq = [(0, (ax, ay))]
    heapify(pq)

    while pq:
        d, (x, y) = heappop(pq)

        if scanned[x, y]:
            continue
        scanned[x, y] = True

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if not blocked[x + dx, y + dy]:
                # relax node
                if d + 1 < distance[x+dx, y+dy]:
                    distance[x+dx, y+dy] = d + 1
                    if x != ax or y != ay:
                        direction[x+dx, y+dy] = direction[x, y]
                    heappush(pq, (d + 1, (x+dx, y+dy)))

    return distance, direction

def focus_map(ax, ay, full_map, r):
    """
    todo: Fix out of bounds
    Focus a map to a smaller area around the Agent

    :param ax:          Agents x position
    :param ay:          Agents y position
    :param full_map:    Full 17x17 Map
    :param r:           Radius of the final map
    :return:            Zoomed map with dimensions 2r + 1
    """
    if r == -1:
        return full_map

    return [row[ay-r : ay+r+1] for row in full_map[ax-r : ax+r+1]]

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

    return [up, down, left, right]

def onehot_encode_direction(direction, scaling=1):
    """
    Onehot encode a direction

    :param direction:   Direction map to be encoded
    :param scaling:     Factor that encoding is multiplied by
    :return:            Onehot encoding for every direction
    """
    up      = int(direction == Direction.UP)    * scaling
    down    = int(direction == Direction.DOWN)  * scaling
    left    = int(direction == Direction.LEFT)  * scaling
    right   = int(direction == Direction.RIGHT) * scaling

    return [up, down, left, right]

def get_nearest_objects(distance_map, objects, k=1):
    """
    Get the k objects closest to the player

    :param distance_map:    Distance map from distance_map() function
    :param objects:         List of object positions being considered
    :param k:               Number of objects to be returned
    :return:                Slice of the original objects list potentially filled up with '-1's if not enough objects
    """
    objects.sort(key=lambda obj: distance_map[obj[0], obj[1]])
    objects = objects[:k]

    while len(objects) < k:
        objects.append((-1, -1))

    return objects

def get_direction(obj, direction_map):
    """
    Always get a valid direction to an object

    :param obj:             (x, y) of object
    :param direction_map:   Direction map from get_distance_map() function
    :return:                Direction to object or Direction.NONE if invalid object
    """
    if 0 <= obj[0] < 17 and 0 <= obj[1] < 17:
        return direction_map[obj]
    else:
        return Direction.NONE

def get_generic_map(objects, field):
    """
    Map of objects

    value 0:        No object
    value 1:        Yes object

    :param objects: List of object positions
    :param field:   Full field data
    :return:        Map of all objects
    """
    object_map = np.full_like(field, 0)
    for (x, y) in objects:
        object_map[x, y] = 1

    return object_map

def get_player_map(ax, ay, others, field):
    """
    Map of players

    value  0:       No player
    value  1:       Our agent
    value -1:       Enemy player

    :param ax:      Players x coordinate
    :param ay:      Players y coordinate
    :param others:  List of other players positions
    :param field:   Full field data
    :return:        Map of all players
    """
    player_map = np.full_like(field, 0)
    player_map[ax, ay] = 1
    for (x, y) in others:
        player_map[x, y] = -1

    return player_map

def get_danger_map(distance_map, bombs, explosion_map):
    """
    Map of potentially dangerous positions

    value of 0:     Safe position
    value of 1-3:   Bomb is about to explode
    value of 4-5:   Active explosion

    :param distance_map:    Distance map from get_distance_map() function
    :param bombs:           List of bombs with their metadata
    :param explosion_map:   Map of currently ongoing explosions
    :return:                Updated Explosion Map
    """
    danger_map = explosion_map.copy()

    for (bx, by), t in bombs: # for every bomb
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # in every direction
            for r in range(0, 4): # from 0 to 3
                if distance_map[bx + r * dx, by + r * dy] == Distance.INFINITE and r != 0: # explosion blocked by wall
                    break
                danger_map[bx + r*dx, by + r*dy] = max(t + 2, danger_map[bx + r*dx, by + r*dy])

    # invert danger to be more meaningful
    danger_map = [[6 - danger if danger > 0 else 0 for danger in row] for row in danger_map]

    return danger_map

def nearest_coins_feature(coins, distance_map, direction_map, k=1):
    """
    List of onehot encoded positions of the k nearest coins

    :param coins:           List of positions of coins
    :param distance_map:    Distance map from get_distance_map() function
    :param direction_map:   Direction map from get_distance_map() function
    :param k:               Number of coins
    :returns                List of the features
    """

    features = []
    for coin_direction in [
        onehot_encode_direction(get_direction(coin, direction_map))
        for coin in get_nearest_objects(distance_map, coins, k)
    ]:
        features.extend(coin_direction)

    return features

def nearest_crates_feature(crates, distance_map, direction_map, k=1):
    """
    List of onehot encoded positions of free spaces of the k nearest crates

    :param crates:          List of positions of crates
    :param distance_map:    Distance map from get_distance_map() function
    :param direction_map:   Direction map from get_distance_map() function
    :param k:               Number of crates
    :returns                List of the features
    """

    # convert crates into locations next to crates
    locations = [
        (x + dx, y + dy)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for x, y in crates
    ]
    locations = list(set(locations)) # convert to set and back to remove duplicates

    features = []
    for crate_direction in [
        onehot_encode_direction(get_direction(crate, direction_map))
        for crate in get_nearest_objects(distance_map, locations, k)
    ]:
        features.extend(crate_direction)

    return features

def enemy_directions_feature(distance_map, direction_map, enemies, k=3):
    """
    :returns the direction to the nearest enemies as one-hot, an array of length 12.
    """

    # Max distance value to be used in the features, the high INF value for the distances "confuses" the model
    max_dist = np.sqrt(30)

    features = []
    for enemy_direction in [
        onehot_encode_direction(
            get_direction(enemy, direction_map),
            scaling=min(np.sqrt(distance_map[enemy]), max_dist)
        )
        for enemy in get_nearest_objects(distance_map, enemies, k)
    ]:
        features.extend(enemy_direction)

    return features

def get_last_actions(k=1):
    """
    returns the last k actions as a one-hot
    """
    one_hot = [0] * 6 * k
    for i, action in enumerate(reversed(last_actions_deque)):
        if i >= k:
            break

        one_hot[6*i + action] = 1

    return one_hot

def is_repeating_actions(k=4):
    """
    returns True when the agent is repeating actions such that it is stuck in one location,
    e.g. 4 times wait, left-right-left-right, up-down-up-down
    """
    # If fewer than k actions have been taken, it cannot be stuck
    if len(last_actions_deque) < k:
        return 0

    # Convert actions to a list for easier processing
    actions = list(last_actions_deque)[-k:]

    # Check if all actions are WAIT
    if actions.count(Action.WAIT) == k:
        return True

    # Check for patterns indicating the agent is stuck
    if actions[0] == actions[2] and actions[1] == actions[3] and actions[0] != actions[1]:
        return True

    return False

def is_repeating_positions(min_unique_positions=5):
    """
    Determines if the agent is likely stuck based on its movement history.
    :return: True if the agent is likely stuck, False otherwise.
    Not called as a feature, but to make a stuck agent drop a bomb to force it to move.
    """
    # Not enough history to make a decision
    if len(coordinate_history) < 20:
        return False

    # Calculate the number of unique positions visited in the last 20 steps
    unique_positions = len(set(coordinate_history))

    if unique_positions < min_unique_positions:
        return True
    else:
        return False

def is_at_crossing(ax, ay):
    """
    Idea: If bombs are not dropped at a crossing, two directions are blocked which lessens the impact.
    This feature enables us to punish that.
    """
    return int((ax % 2 == 1) and (ay % 2 == 1))

def number_of_crates_reachable(ax, ay, field):
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

def get_free_distances(ax, ay, field):
    """
    :return: in all four directions the sqrt of the distance the agent can "see", e.g. if there is a wall next to the agent
    zero in that direction. The distance can be at most 14 when the agent is at the edge of the field and a whole
    row is free
    """
    distances = [0, 0, 0, 0]  # Distances for up, down, left, and right

    # Up direction
    for y in range(ay - 1, -1, -1):
        if field[ax, y] != 0:
            break
        distances[3] += 1

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

    # Right direction
    for x in range(ax + 1, field.shape[0]):
        if field[x, ay] != 0:
            break
        distances[0] += 1

    return np.sqrt(distances)

def get_direction_to_safety(distance_map, direction_map, danger_map, danger_threshold, time_limit=4):
    """Idea: when an agent finds itself in the explosion radius of a bomb, this should point the agent
    in the direction of the nearest safe tile, especially useful for avoiding placing a bomb and then
    walking in a dead end waiting for the bomb to blow up. Should also work for escaping enemy bombs.
    """
    positions = [[(x, y) for y in range(17)] for x in range(17)]
    positions.sort(key=lambda pos: distance_map[pos[0], pos[1]])

    for position in positions:
        if distance_map[position] > time_limit:
            return Direction.NONE
        if danger_map[position] < danger_threshold:
            return direction_map[position]

def get_disallowed_actions(ax, ay, bomb_available, distance_map, direction_map, danger_map, danger_threshold, safety_direction):
    """
    can be used as feature or to block those actions in the choose_action function
    that would lead to certain death
    """
    disallowed = [False] * 6

    # Cannot bomb if it has no bomb. I have never seen the agent try to do that though
    disallowed[Action.BOMB] = not bomb_available

    if safety_direction == Direction.NONE:
        # if not in danger, do not walk into danger or wall
        for direction, (dx, dy) in Direction.deltas():
            if danger_map[ax + dx, ay + dy] >= danger_threshold or distance_map[ax + dx, ay + dy] == Distance.INFINITE:
                # Danger or obstacle ahead
                disallowed[Direction.to_action(direction)] = True
    else:
        # if in danger, walk out of it as fast as possible
        for direction in [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]:
            if direction != safety_direction:
                disallowed[Direction.to_action(direction)] = True

    # disallow bomb placement if it would lead to certain death
    if bomb_available:
        # Predict the danger level after placing a bomb
        future_danger_map = get_danger_map(distance_map, [((ax, ay), 3)], danger_map)

        # calculate the future game state
        future_nearest_safe = get_direction_to_safety(distance_map, direction_map, future_danger_map, 1)

        # check whether the agent could escape the bomb
        if future_nearest_safe == Direction.NONE:
            disallowed[Action.BOMB] = True

    return disallowed

def get_dead_end_map(field, enemies):
    """
    todo: fix
    Returns map of all dead ends.
    The idea is to teach the agent to avoid these when enemies are nearby and to follow enemy agents inside
    and then placing a bomb behind them when they make the mistake of entering one.
    """
    dead_end_map = np.zeros_like(field)
    dead_end_list = []

    def is_dead_end(x, y):
        """Check if a tile is part of a dead end and trace the full dead end path."""
        dead_end_path = [(x, y)]  # Start from this tile
        current_x, current_y = x, y
        prev_x, prev_y = -1, -1

        while True:
            open_neighbors = []
            for _, (dx, dy) in Direction.deltas():
                nx, ny = current_x + dx, current_y + dy
                if field[nx, ny] == 0 and (nx, ny) != (prev_x, prev_y):  # Only count open tiles, avoid the previous one
                    open_neighbors.append((nx, ny))

            if len(open_neighbors) == 1:  # Dead-end continues if only one neighbor is open
                prev_x, prev_y = current_x, current_y
                current_x, current_y = open_neighbors[0]
                dead_end_path.append((current_x, current_y))
            else:
                break  # If more than one neighbor is open, it's not part of a dead end

        # A valid dead end must have length of at least 2 tiles
        if len(dead_end_path) > 2:
            for px, py in dead_end_path[:-1]:
                dead_end_map[px, py] = 1

            # Check if there is an enemy in this dead end
            enemy_in_dead_end = None
            for enemy in enemies:
                if enemy in dead_end_path:  # If enemy is in the dead end path
                    enemy_in_dead_end = enemy
                    break

            dead_end_list.append({
                'closed_end': dead_end_path[0],
                'open_end': dead_end_path[-2],
                'enemy': enemy_in_dead_end
            })

    # Iterate over every cell in the field
    for x in range(1, 16):
        for y in range(1, 16):
            if field[x, y] == 0:  # Free tile, check for dead end
                open_neighbors = sum(1 for _, (dx, dy) in Direction.deltas() if field[x + dx, y + dy] == 0)
                if open_neighbors == 1:  # Only one open neighbor indicates a potential dead end entrance
                    is_dead_end(x, y)

    return dead_end_map, dead_end_list

def get_direction_to_trapped_enemy(dead_end_list, distance_map, direction_map):
    """
    todo: something isn't right here
    If there is the open end of a dead end containing an enemy closer to our agent than the manhattan distance of the
    enemy to the open end, return one-hot directions to the enemy.
    Also return whether the agent is less than four steps away
    from the closed end of the dead end, because if it drops a bomb then,
    the enemy is dead (except when another agent blows up a crate that belonged to the closed end).
    This can then easily be rewarded, enabling the agent to learn this behavior without the need for artificial constraints.
    """
    direction = Direction.NONE

    # Iterate over each dead end to find one that contains an enemy
    for dead_end in dead_end_list:
        enemy = dead_end['enemy']
        if enemy is None:
            continue  # No enemy in this dead end, skip

        open_end = dead_end['open_end']
        closed_end = dead_end['closed_end']

        # Check if the agent is closer to the open end than the enemy is
        agent_to_open_end_dist = distance_map[open_end]
        enemy_to_open_end_dist = abs(enemy[0] - open_end[0]) + abs(
            enemy[1] - open_end[1])  # Manhattan distance (ignores curved dead ends)

        if agent_to_open_end_dist <= enemy_to_open_end_dist:
            # If agent is closer to the open end, find the direction to the enemy
            direction = get_direction(enemy, direction_map)

            # Check if the agent can trap the enemy with a bomb (agent is near the closed end)
            #agent_to_closed_end_dist = distance_map[closed_end]
            #if agent_to_closed_end_dist <= 3 and \
            #        (closed_end[0] == ex == ax or closed_end[1] == ey == ay):  # it has to be a straight dead end
            #    can_trap_enemy = 1

            break  # Stop after finding the first valid enemy in a dead end

    return onehot_encode_direction(direction)

def do_not_enter_dead_end(ax, ay, dead_end_map, dead_end_list, enemies, distance_map):
    """
    Do not enter a dead end if an enemy is nearby (within 4 tiles) and not inside the same dead end.
    :return: One-hot direction to the dead-end entrance if an enemy is nearby, else a zero-vector.
    """
    # Iterate through all adjacent tiles around the agent
    if dead_end_map[ax, ay] == 0:  # Not already in dead end
        for direction, (dx, dy) in Direction.deltas():
            # Check if this tile is an open end of a dead end
            for dead_end in dead_end_list:
                if dead_end['open_end'] == (ax + dx, ay + dy):
                    # Check if any enemy is nearby
                    for enemy in enemies:
                        if distance_map[enemy] < 5:  # Chose 'within 5 tiles' as nearby
                            return onehot_encode_direction(direction)

def is_next_to_enemy(ax, ay, others, d=1):
    """
    Checks if the agent is adjacent (d=1) or close to any enemy.
    Can be used to reward attacking enemy agents when they are close.
    """
    for ex, ey in others:
        if abs(ax - ex) + abs(ay - ey) == d:
            return True

    return False

# todo today: achieve immortality
#  - sometimes runs into enemy explosions while fleeing from its own explosion - flaw in disallowed_actions
#  - can get trapped between two bombs/surrounded by other agents - add function/features that prevents this
#  - make disallowed_actions the "function of absolute immortality" - the agent should not be able to die at all
#  - the agent does not seem motivated to move toward and blow up crates in the end
#  - experiment: add many layers and see what happens
#  - check whether all rewards always work as they are supposed to and do not somehow cause the agent to learn
#    unwanted stuff such as the oscillation behavior
#  - Why exactly does performance decrease after some amount of training steps? Why the periodic ups and downs?


# changes:
# - Directions, Actions, Distance to utils classes
# - Usage of maps
# - get_generic_map
# - danger_map improved
# - is_stuck -> is_repeating_actions
# - is_stuck2 -> is_repeating_positions
# - neighbouring -> focus_map
# - relative_bomb_positions removed
# - nearest_coins_feature, nearest_crate_feature improved
# - last_action -> last_actions(k)
# - direction_to_safety improved
# - disallowed_actions improved
# - enemies_distances_and_directions -> enemy_directions_feature
# - modified direction_to_enemy_in_dead_end a bit