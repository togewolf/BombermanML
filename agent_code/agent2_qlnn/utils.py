
class Direction:
    UP = -2
    LEFT = -1
    NONE = 0
    RIGHT = 1
    DOWN = 2

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