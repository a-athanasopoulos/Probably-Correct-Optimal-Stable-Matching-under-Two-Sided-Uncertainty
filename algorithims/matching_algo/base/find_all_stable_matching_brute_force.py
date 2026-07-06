from algorithims.matching_algo.base.find_all_matchings import find_all_matchings
from algorithims.matching_algo.base.is_stable import is_stable


def all_stable_matching_brute_force(players_ranking, arms_rankings, num_players):
    matchings = find_all_matchings(num_of_players=num_players)
    list_of_stable_matchings = []
    for matching in matchings:
        if is_stable(matching, arms_rankings, players_ranking, num_players):
            list_of_stable_matchings += [matching]
    return list_of_stable_matchings