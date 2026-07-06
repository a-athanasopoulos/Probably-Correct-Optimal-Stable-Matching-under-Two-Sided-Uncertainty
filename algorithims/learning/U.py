import numpy as np
from algorithims.learning.BasePac import Base_Elimination

from algorithims.matching_algo.base.gale_shapley import GS
from algorithims.learning.extra_utils.utils import rewards_to_preferences, tuple_to_matching

from algorithims.matching_algo.coloring.minimum_edge_coloring import get_matchings_edge_coloring


class U(Base_Elimination):

    def pac_loop(self):
        t = 0
        self.active_pairs = np.logical_or(self.active_p_a, self.active_a_p.T) * 1
        while self.active_pairs.sum() > 0:
            self.active_p_a = np.ones((self.N_p, self.N_a), dtype=int)
            self.active_a_p = np.ones((self.N_a, self.N_p), dtype=int)
            self.active_pairs = np.logical_or(self.active_p_a, self.active_a_p.T) * 1
            matching_cover = get_matchings_edge_coloring(num_players=self.N_p,
                                                         num_arms=self.N_a,
                                                         adjacency_matrix=self.active_pairs,
                                                         complete_flag=True)
            for tmp_matching in matching_cover:
                t += 1
                if t % 10000 == 0:
                    print(f"Step:{t}")
                for player, arm in tmp_matching:
                    # Sample rewards from true distributions
                    reward_player = self.sample_rewards(self.mean_p[player, arm])
                    reward_arm = self.sample_rewards(self.mean_a[arm, player])

                    # Update running mean estimates for preferences
                    self.mean_est_p[player, arm] = self.update_reward(
                        reward=reward_player,
                        num_plays=self.num_plays[player, arm],
                        old_rewards=self.mean_est_p[player, arm]
                    )

                    self.mean_est_a[arm, player] = self.update_reward(
                        reward=reward_arm,
                        num_plays=self.num_plays[player, arm],
                        old_rewards=self.mean_est_a[arm, player]
                    )

                    # Increment play counter
                    self.num_plays[player, arm] += 1

                self.step_metrics()
            self.update_confidence_interval()
            self.active_p_a = self.update_active_pairs(active_pairs=self.active_p_a,
                                                       N=self.N_p,
                                                       confidence_intervals=self.confidence_intervals_p)
            self.active_a_p = self.update_active_pairs(active_pairs=self.active_a_p,
                                                       N=self.N_a,
                                                       confidence_intervals=self.confidence_intervals_a)
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

        return self.out_matching, self.correct, self.sample_complexity
