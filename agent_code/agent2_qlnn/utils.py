
class Action:
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    WAIT = 4
    BOMB = 5

    @classmethod
    def from_str(a, str):
        action_map = {
            'RIGHT':    a.RIGHT,
            'DOWN':     a.DOWN,
            'LEFT':     a.LEFT,
            'UP':       a.UP,
            'WAIT':     a.WAIT,
            'BOMB':     a.BOMB
        }
        return action_map[str]

    @classmethod
    def to_str(a, action):
        action_map = {
            a.RIGHT:    'RIGHT',
            a.DOWN:     'DOWN',
            a.LEFT:     'LEFT',
            a.UP:       'UP',
            a.WAIT:     'WAIT',
            a.BOMB:     'BOMB'
        }
        return action_map[action]


class Direction:
    UP = -2
    LEFT = -1
    NONE = 0
    RIGHT = 1
    DOWN = 2

    @classmethod
    def deltas(d):
        return {d.LEFT: (-1, 0), d.RIGHT: (1, 0), d.UP: (0, -1), d.DOWN: (0, 1)}

    @classmethod
    def from_action(d, action):
        direction_map = {
            Action.RIGHT:   d.RIGHT,
            Action.DOWN:    d.DOWN,
            Action.LEFT:    d.LEFT,
            Action.UP:      d.UP,
            Action.WAIT:    d.NONE,
            Action.BOMB:    d.NONE
        }
        return direction_map[action]

    @classmethod
    def to_action(d, direction):
        direction_map = {
            d.RIGHT: Action.RIGHT,
            d.DOWN: Action.DOWN,
            d.LEFT: Action.LEFT,
            d.UP: Action.UP,
        }
        return direction_map[direction]


    @classmethod
    def rot90(d, direction, k=1):
        for i in range(k % 4):
            if direction == d.UP:
                direction = d.RIGHT
            elif direction == d.RIGHT:
                direction = d.DOWN
            elif direction == d.DOWN:
                direction = d.LEFT
            elif direction == d.LEFT:
                direction = d.UP
            else:
                return d.NONE

        return direction

    @classmethod
    def fliplr(d, direction):
        if direction == d.LEFT:
            return d.RIGHT
        elif direction == d.RIGHT:
            return d.LEFT
        else:
            return direction

    @classmethod
    def flipud(d, direction):
        if direction == d.UP:
            return d.DOWN
        elif direction == d.DOWN:
            return d.UP
        else:
            return direction

class Distance:
    INFINITE = 999