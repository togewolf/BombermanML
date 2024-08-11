
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self): ...


def act(self, game_state: dict) -> str:
    """ game state dict: instructions p 6
     return max p over all actions, exclude invalid actions
     """
