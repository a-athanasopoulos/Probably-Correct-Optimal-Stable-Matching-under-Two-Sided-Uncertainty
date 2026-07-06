import numpy as np

from algorithims.matching_algo.matching_utils import inv_matching

import copy


def GS(pref_s1, pref_s2):
    """
    GS Algorithim.
    pref_s1 dict of list
    pref_s2 dict of list
    The list is not complete, and when can be none if agent prefer to remain unmatched.
    """

    pref_p = copy.deepcopy(pref_s1)
    pref_a = copy.deepcopy(pref_s2)
    N_p = len(pref_p)
    N_a = len(pref_a)

    set_players = set(range(0, N_p))
    match = {a: None for a in range(0, N_a)}
    done_players = set()

    t = 0
    while len(done_players) < N_p:
        t += 1
        for player in set_players.difference(done_players):
            if len(pref_p[player]) == 0:
                done_players.update({player})
            else:
                proposal = pref_p[player].pop(0)
                if match[proposal] is None:
                    if player in pref_a[proposal]:
                        match[proposal] = player
                        done_players.update({player})
                elif pref_a[proposal].index(player) < pref_a[proposal].index(match[proposal]):
                    done_players.remove(match[proposal])
                    match[proposal] = player
                    done_players.update({player})

    # match to tuple of agent
    matching = [(match[a], a) for a in match.keys() if match[a] is not None]
    return matching


def gale_shapley_algo(arms_rankings, player_ranking, num_players, num_arms):
    # propose_order records the order players should follow while proposing
    init_propose_order = np.zeros(num_players, int)
    propose_order = init_propose_order

    # matched record whether a specific player is matched or not
    matched = np.zeros(num_players, bool)

    # matching records the choice of a player for a specific arm
    matching = [[] for _ in range(num_arms)]

    # Terminates if all matched
    while np.sum(matched) != num_players:

        # players propose at the same time
        for p_idx in range(num_players):
            if not matched[p_idx]:
                # p_proposal is the index of an arm
                # propose_order is the vector, p_o[i] is the order of player i's next proposal
                p_proposal = player_ranking[p_idx][propose_order[p_idx]]
                matching[p_proposal].append(p_idx)

        # arms choose its player
        for a_idx in range(num_arms):
            a_choices = matching[a_idx]

            if len(a_choices) != 0:
                # each arm chooses the its most preferable one
                a_choice = next((x for x in arms_rankings[a_idx] if x in matching[a_idx]), None)
                # update arm's choice where there should only be one left
                matching[a_idx] = [a_choice]
                # update player's state of matched
                for p_idx in a_choices:
                    matched[p_idx] = (p_idx == a_choice)
                    propose_order[p_idx] += (1 - (p_idx == a_choice))

    return np.squeeze(matching)


def gale_shapley_algo2(arms_rankings, player_ranking, num_players, num_arms):
    matched = np.zeros(num_players, bool)
    proposal_order = np.zeros(num_players, int)
    arm_matching = [[] for _ in range(num_arms)]

    i = 0
    while np.sum(matched) != num_arms:
        for player in range(num_players):
            if not matched[player]:
                player_proposal = player_ranking[player][proposal_order[player]]
                proposal_order[player] += 1

                if arm_matching[player_proposal] == []:
                    arm_matching[player_proposal] = player
                    matched[player] = True
                elif arms_rankings[player_proposal].index(arm_matching[player_proposal]) > arms_rankings[
                    player_proposal].index(player):
                    matched[arm_matching[player_proposal]] = False
                    matched[player] = True
                    arm_matching[player_proposal] = player
        i += 1
    return inv_matching(arm_matching)  # return player matching


def gale_shapley_algo3(arms_rankings, player_ranking, num_players, num_arms):
    matched = np.zeros(num_players, dtype=bool)
    proposal_order = np.zeros(num_players, dtype=int)
    arm_matching = [None] * num_arms  # Using None instead of [] for better performance

    num_matched_arms = 0  # Track the number of matched arms

    while num_matched_arms < num_arms:
        for player in range(num_players):
            if not matched[player]:
                player_proposal = player_ranking[player][proposal_order[player]]
                proposal_order[player] += 1

                if arm_matching[player_proposal] is None:
                    arm_matching[player_proposal] = player
                    matched[player] = True
                    num_matched_arms += 1  # Increment matched arms count
                elif arms_rankings[player_proposal].index(arm_matching[player_proposal]) > arms_rankings[
                    player_proposal].index(player):
                    matched[arm_matching[player_proposal]] = False
                    matched[player] = True
                    arm_matching[player_proposal] = player

    return inv_matching(arm_matching)
