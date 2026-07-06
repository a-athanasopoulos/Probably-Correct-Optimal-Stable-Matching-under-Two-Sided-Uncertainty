import numpy as np


def update_reward(reward, num_plays, old_rewards):
    """Update expected rewards"""
    new_rewards = ((old_rewards * num_plays) + reward) / (num_plays + 1)
    return new_rewards


def rewards_to_preferences(rewards):
    """
    convert rewards to preferences array
    """
    return np.argsort(-rewards)


def tuple_to_matching(tmp_matching_t, N_p):
    """Conver a tuple to matching list"""
    matching = np.array([None] * N_p)
    for (p, a) in tmp_matching_t:
        matching[p] = a
    return matching

def fix_psm(d):
    """
    Convert a dict like {'p_0': 'a_0', 'p_1': 'a_1'}
    into a list [0, 1], where p_i is matched to a_j.
    """
    matching = []
    for key in sorted(d.keys(), key=lambda x: int(x.split('_')[1])):
        # extract the number after 'a_'
        j = int(d[key].split('_')[1])
        matching.append(j)
    return matching
