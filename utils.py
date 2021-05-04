import numpy as np


def action2ind(action: int):
    """Converts action 0-9 into a tuple of form [0-512, 0-512] showing center of action to draw on"""
    assert type(action) is int

    action_arr = np.zeros(9)
    action_arr[action] = 1
    action_arr = action_arr.reshape((3, 3)).T
    action_ind = (np.array(np.argwhere(action_arr)[0]) / 2 * 172) + 85

    return action_ind.astype(int)


def ind2mtrx(action_ind: np.ndarray([int, int])):
    """Converts action index from [0-512, 0-512] to Matrix co-ord form [-1 - 1, -1 - 1]"""
    assert type(action_ind[0]) is int

    return ((np.array(action_ind)) / 256) - 1  # Flip to go to form x, y instead of y, x?
