import events as e


# events listed in instructions p7
# from .callbacks import something if necessary

def setup_training(self): ...


def game_events_occurred(self, old_game_state, self_action, new_game_state, events): ...


def end_of_round(self, last_game_state, last_action, events): ...
# learn here

# use events to define rewards and penalties
