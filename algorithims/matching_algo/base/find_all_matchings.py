import itertools


def find_all_matchings(num_of_players):
    """
    produce all matching of two-sided matching with n agents in each side
    """
    player_permutations = list(itertools.permutations(range(num_of_players)))
    return player_permutations

