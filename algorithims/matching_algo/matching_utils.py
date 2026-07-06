import numpy as np


def inv_matching(matching):
    """
    Get an inverse matching obj
    """
    inv_match = {}
    for player in matching.keys():
        arm = matching[player]
        inv_match[arm] = player
    return inv_match


def get_rank_function(preferences):
    pref = np.array(preferences)
    rank = np.zeros(pref.shape)
    for player in range(pref.shape[0]):
        for position, arm in enumerate(pref[player, :]):
            rank[player, arm] = position
    return rank
