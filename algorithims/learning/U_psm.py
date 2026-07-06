from algorithims.learning.BasePac import Base_Elimination_pref

from algorithims.matching_algo.base.gale_shapley import GS
from algorithims.matching_algo.super_stable.has_pervasive_stable_matching import pervasive_stable_matching
from algorithims.learning.extra_utils.utils import rewards_to_preferences, tuple_to_matching, fix_psm

from algorithims.matching_algo.coloring.minimum_edge_coloring import get_matchings_edge_coloring

import numpy as np
import networkx as nx


def matrix_to_dag(matrix, prefix="a"):
    """
    Convert a pairwise preference matrix to a DAG.

    Parameters
    ----------
    matrix : array-like (n x n)
        matrix[i][j] == 1 means prefix_{i+1} > prefix_{j+1}
    prefix : str
        Node name prefix (default: 'a')

    Returns
    -------
    G : nx.DiGraph
        Directed acyclic graph representing the dominance relation
    """
    M = np.asarray(matrix)
    n = M.shape[0]

    G = nx.DiGraph()
    nodes = [f"{prefix}{i}" for i in range(n)]
    G.add_nodes_from(nodes)

    for i in range(n):
        for j in range(n):
            if M[i, j] == 1:
                G.add_edge(nodes[i], nodes[j])

    return G


class U_PSM_pac(Base_Elimination_pref):
    """
    this is a version of uniform sampling with not know min delta.
    At every round sample all the pairs.
    It stops when all pairs are eliminated.
    """

    @staticmethod
    def check_pervasive_stable_matching(partial_pref_p_a, partial_pref_a_p, N_p, N_a):
        prefix_1 = "p"
        prefix_2 = "a"
        employers = [prefix_1 + "_" + str(a) for a in range(N_p)]
        applicants = [prefix_2 + "_" + str(a) for a in range(N_a)]

        dags_p = {}
        for p in range(N_p):
            agent = prefix_1 + "_" + str(p)
            dags_p[agent] = matrix_to_dag(partial_pref_p_a[p], prefix=prefix_2 + "_")

        dags_a = {}
        for a in range(N_a):
            agent = prefix_2 + "_" + str(a)
            dags_a[agent] = matrix_to_dag(partial_pref_a_p[a], prefix=prefix_1 + "_")

        sm, psm = pervasive_stable_matching(pref_e_dag=dags_p,
                                            pref_a_dag=dags_a,
                                            employers=employers,
                                            applicants=applicants)
        if psm is not False:
            has_psm = True
            has_super_stable = True
        else:
            has_psm = False
            has_super_stable = True if sm else False
        return sm, has_psm, has_super_stable

    def pac_loop(self):
        t = 0

        self.active_pairs = np.logical_or(self.active_p_a, self.active_a_p.T) * 1
        has_psm = False
        psm = False
        print((has_psm == False) or (self.active_pairs.sum() == 0))
        while (not has_psm) and (self.active_pairs.sum() != 0):
            # this sample all pairs every round and stops when psm is found
            self.active_p_a = np.ones((self.N_p, self.N_a), dtype=int)
            self.active_a_p = np.ones((self.N_a, self.N_p), dtype=int)
            self.active_pairs = np.logical_or(self.active_p_a, self.active_a_p.T) * 1
            matching_cover = get_matchings_edge_coloring(num_players=self.N_p,
                                                         num_arms=self.N_a,
                                                         adjacency_matrix=self.active_pairs,
                                                         complete_flag=False)
            for tmp_matching in matching_cover:
                t += 1
                if t % 5000 == 0:
                    print(f"Step:{t}")
                for player, arm in tmp_matching:
                    # Sample rewards from true distributions
                    reward_player = self.sample_rewards(self.mean_p[player, arm])
                    reward_arm = self.sample_rewards(self.mean_a[arm, player])

                    # Update running mean estimates for preferences
                    if self.active_p_a[player, arm] == 1:
                        self.mean_est_p[player, arm] = self.update_reward(
                            reward=reward_player,
                            num_plays=self.num_plays[player, arm],
                            old_rewards=self.mean_est_p[player, arm]
                        )
                    if self.active_a_p[arm, player] == 1:
                        self.mean_est_a[arm, player] = self.update_reward(
                            reward=reward_arm,
                            num_plays=self.num_plays[player, arm],
                            old_rewards=self.mean_est_a[arm, player]
                        )

                    # Increment play counter
                    self.num_plays[player, arm] += 1

                self.step_metrics()
            self.update_confidence_interval()
            self.update_partial_pref(partial_pref=self.partial_pref_p_a,
                                     active_pairs=self.active_p_a,
                                     confidence_intervals=self.confidence_intervals_p,
                                     elim_time=self.elim_time_p_a,
                                     N=self.N_p,
                                     t=t)
            self.update_active_pairs_from_pref(partial_pref=self.partial_pref_p_a, active_pairs=self.active_p_a,
                                               N=self.N_p)

            self.update_partial_pref(partial_pref=self.partial_pref_a_p,
                                     active_pairs=self.active_a_p,
                                     confidence_intervals=self.confidence_intervals_a,
                                     elim_time=self.elim_time_a_p,
                                     N=self.N_a,
                                     t=t)
            self.update_active_pairs_from_pref(partial_pref=self.partial_pref_a_p, active_pairs=self.active_a_p,
                                               N=self.N_a)

            self.active_pairs = np.logical_or(self.active_p_a, self.active_a_p.T) * 1

            psm, has_psm, has_sm = self.check_pervasive_stable_matching(self.partial_pref_p_a,
                                                                        self.partial_pref_a_p,
                                                                        self.N_p,
                                                                        self.N_a)
            self.active_pairs = np.logical_or(self.active_p_a, self.active_a_p.T) * 1

        # Commit Phase: Construct estimated strict preferences
        est_pref_p = rewards_to_preferences(self.mean_est_p)
        est_pref_a = rewards_to_preferences(self.mean_est_a)

        # Convert to dictionary format for the GS algorithm
        tmp_pref_p = {p: est_pref_p[p].tolist() for p in range(self.N_p)}
        tmp_pref_a = {a: est_pref_a[a].tolist() for a in range(self.N_a)}

        # Find the employer-optimal matching based on estimates
        tuple_matching = GS(pref_s1=tmp_pref_p, pref_s2=tmp_pref_a)

        # Convert tuple format back to a standard matching list
        self.out_matching = tuple_to_matching(tuple_matching, self.N_p).tolist()

        # Calculate final performance metrics
        self.correct, self.sample_complexity = self.final_metrics(t=t, matching=self.out_matching)
        correct = False
        if psm:
            fixed_psm = fix_psm(psm)
            correct = self.optimal_stable_matching == fixed_psm

        return self.out_matching, correct, self.sample_complexity
