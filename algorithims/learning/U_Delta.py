import numpy as np
from algorithims.learning.BasePac import BasePac

from algorithims.learning.utils import get_rr_matching
from algorithims.matching_algo.base.gale_shapley import GS
from algorithims.learning.extra_utils.utils import rewards_to_preferences, tuple_to_matching


class U_Delta(BasePac):
    def __init__(self, mean_p, mean_a, N_p, N_a, delta, min_Delta):
        # Correctly initialize the base class
        super().__init__(mean_p, mean_a, N_p, N_a, delta)
        self.min_Delta = min_Delta
        self.h = int(np.ceil(8 * np.log(2 * self.N_p * self.N_a / self.delta) / (self.min_Delta ** 2)))

    def pac_loop(self):
        # h: Number of samples per pair required for PAC guarantees

        # Total iterations to ensure every pair is sampled h times
        # Using a Round Robin approach (get_rr_matching)
        total_steps = max(self.N_p, self.N_a) * self.h

        t = 0
        for t in range(total_steps):
            # get_rr_matching ensures uniform coverage of all possible pairs
            tmp_matching = get_rr_matching(t, N_p=self.N_p)

            for player, arm in enumerate(tmp_matching):
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
