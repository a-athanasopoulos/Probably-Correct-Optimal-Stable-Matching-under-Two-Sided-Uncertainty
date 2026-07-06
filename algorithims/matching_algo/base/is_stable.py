import numpy as np

from algorithims.matching_algo.base.find_all_matchings import find_all_matchings


def is_stable(arm_matching, players_ranking, arms_rankings, num_players):
    return not is_unstable(arm_matching, players_ranking, arms_rankings, num_players)


def is_unstable(arm_matching, players_ranking, arms_rankings, num_players):
    # arm_matching: [0,1,-1]
    # arm 0 matches player 0; arm 1 matches player 1; arm 2 matches nothing

    # if arm unmatched -> unstable
    if -1 in arm_matching:
        return 1

    # create player matching
    player_matching = np.ones(num_players) * (-1)
    for p_idx in range(num_players):
        if p_idx in arm_matching:
            player_matching[p_idx] = arm_matching.index(p_idx)

    # if player unmatched -> unstable
    if -1 in player_matching:
        return 1

    # find blocking pair
    for p_idx in range(num_players):
        for possible_arm_rank in range(players_ranking[p_idx].index(int(player_matching[p_idx]))):
            arm = players_ranking[p_idx][possible_arm_rank]
            for possible_player_rank in range(arms_rankings[arm].index(arm_matching[arm])):
                if arms_rankings[arm][possible_player_rank] == p_idx:
                    return 1
    return 0


if __name__ == '__main__':
    players_ranking_test = [[0, 1], [0, 1]]
    arm_ranking_test = [[0, 1], [0, 1]]
    num_players_test = 2

    arm_matching_test = [1, 0]
    res = is_unstable(arm_matching=arm_matching_test,
                      players_ranking=players_ranking_test,
                      arms_rankings=arm_ranking_test,
                      num_players=num_players_test)
    token = "unstable" if res else "stable"
    print("The matching is " + str(token))

    matchings = find_all_matchings(num_of_players=2)
    for matching in matchings:
        matching = list(matching)
        res = is_unstable(arm_matching=matching,
                          players_ranking=players_ranking_test,
                          arms_rankings=arm_ranking_test,
                          num_players=num_players_test)
        print(res)
        token = "unstable" if res else "stable"
        print(f"The matching {matching} is " + str(token))
