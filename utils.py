from typing import Tuple

import numpy as np
from matplotlib import colors
from nOmicron.microscope import xy_scanner

rabanimap = colors.ListedColormap(["black", "white", "orange"])


def get_scan():
    print("Acquiring scan")
    return xy_scanner.get_xy_scan("Z", "Forward-Backward", "Up")


def action2ind(action: int):
    """Converts action 0-9 into a tuple of form [0-512, 0-512] showing center of action to draw on"""
    # assert type(action) is int
    #
    # action_arr = np.zeros(9)
    # action_arr[action] = 1
    # action_arr = np.flipud(np.fliplr((action_arr.reshape((3, 3)).T)))   # I don't understand why this works, it should do nothing?????
    # action_ind = np.array(np.argwhere(action_arr)[0])

    if action == 0:
        action_ind = [392, 392]
    elif action == 1:
        action_ind = [220, 392]
    elif action == 2:
        action_ind = [50, 392]
    elif action == 3:
        action_ind = [392, 220]
    elif action == 4:
        action_ind = [220, 220]
    elif action == 5:
        action_ind = [50, 220]
    elif action == 6:
        action_ind = [392, 50]
    elif action == 7:
        action_ind = [220, 50]
    elif action == 8:
        action_ind = [50, 50]
    else:
        raise ValueError

    return action_ind


def ind2mtrx(action_ind: Tuple[int, int]):
    """Converts action index from [0-512, 0-512] to Matrix co-ord form [-1 - 1, -1 - 1]"""
    # assert type(action_ind[0]) is int

    # Invert y because of different origin
    mtrx_inds = ((np.array(action_ind)) / 256) - 1
    mtrx_inds[0] = -mtrx_inds[0]

    return mtrx_inds
