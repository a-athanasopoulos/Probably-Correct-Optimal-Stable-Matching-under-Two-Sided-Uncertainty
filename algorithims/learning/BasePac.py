import numpy as np

from algorithims.matching_algo.base.gale_shapley import GS
from algorithims.learning.extra_utils.utils import rewards_to_preferences, tuple_to_matching


class BasePac(object):
    def __init__(self, mean_p, mean_a, N_p, N_a, delta):
        self.N_p = N_p
        self.N_a = N_a
        self.mean_p = mean_p
        self.mean_a = mean_a

        # get preferences from rewards
        true_pref_p = rewards_to_preferences(mean_p)
        true_pref_a = rewards_to_preferences(mean_a)
        self.true_pref_p = {p: true_pref_p[p].tolist() for p in range(self.N_p)}
        self.true_pref_a = {a: true_pref_a[a].tolist() for a in range(self.N_a)}

        # get optimal stable matching
        self.optimal_stable_matching = self.get_optimal_stable_matching()

        # init
        self.num_plays = np.zeros((self.N_p, self.N_a))
        self.mean_est_p = np.zeros((self.N_p, self.N_a))
        self.mean_est_a = np.zeros((self.N_a, self.N_p))

        # Algorithim parameters
        self.delta = delta

        # output
        self.out_matching = None
        self.matching_t = None
        # metrics
        self.correct = False
        self.sample_complexity = 0
        self.correct_t = []

    def get_optimal_stable_matching(self):
        tuple_matching = GS(pref_s1=self.true_pref_p, pref_s2=self.true_pref_a)
        matching = tuple_to_matching(tuple_matching, self.N_p).tolist()
        return matching

    @staticmethod
    def update_reward(reward, num_plays, old_rewards):
        new_rewards = ((old_rewards * num_plays) + reward) / (num_plays + 1)
        return new_rewards

    @staticmethod
    def sample_rewards(true_mean, scale=1):
        return np.random.normal(loc=true_mean, scale=scale)

    def step_metrics(self):
        est_pref_p = rewards_to_preferences(self.mean_est_p)
        est_pref_a = rewards_to_preferences(self.mean_est_a)

        # Convert to dictionary format for the GS algorithm
        tmp_pref_p = {p: est_pref_p[p].tolist() for p in range(self.N_p)}
        tmp_pref_a = {a: est_pref_a[a].tolist() for a in range(self.N_a)}

        # Find the employer-optimal matching based on estimates
        tuple_matching = GS(pref_s1=tmp_pref_p, pref_s2=tmp_pref_a)
        matching_t = tuple_to_matching(tuple_matching, self.N_p).tolist()
        correct = matching_t == self.optimal_stable_matching
        self.correct_t += [correct]
        self.matching_t = matching_t

    def final_metrics(self, t, matching):
        correct = matching == self.optimal_stable_matching
        sample_complexity = t
        return correct, sample_complexity

    def get_metrics_dict(self):
        metrics = {"correct": self.correct,
                   "sample_complexity": self.sample_complexity,
                   "correct_t": self.correct_t}
        return metrics


class Base_Elimination(BasePac):
    """
    this is a version of uniform sampling with not know min delta.
    At every round sample all the pairs.
    It stops when all pairs are eliminated.
    """

    def __init__(self, mean_p, mean_a, N_p, N_a, delta, min_Delta=None):
        # Correctly initialize the base class
        super().__init__(mean_p, mean_a, N_p, N_a, delta)

        self.confidence_intervals_p = np.ones((self.N_p, self.N_a, 2)) * np.inf
        self.confidence_intervals_p[:, :, 0] = - np.inf

        self.confidence_intervals_a = np.ones((self.N_a, self.N_p, 2)) * np.inf
        self.confidence_intervals_a[:, :, 0] = - np.inf

        self.active_p_a = np.ones((self.N_p, self.N_a), dtype=int)
        self.active_a_p = np.ones((self.N_a, self.N_p), dtype=int)
        self.active_pairs = None

    @staticmethod
    def overlapping_agents(UCB, LCB):
        n_agents = len(UCB)
        eliminate = np.zeros(n_agents, dtype=bool)
        for i in range(n_agents):
            overlap = False
            for j in range(n_agents):
                if i == j:
                    continue
                # intervals [LCB[i], UCB[i]] and [LCB[j], UCB[j]] overlap if:
                if not (UCB[i] < LCB[j] or UCB[j] < LCB[i]):
                    overlap = True
                    break
            if not overlap:
                eliminate[i] = True
        return eliminate

    @staticmethod
    def update_active_pairs(active_pairs, N, confidence_intervals):
        """
        N_p: Number of primary agents (e.g., employers).
        confidence_intervals: Shape (N_p, N_a, 2) where index 0 is LCB and 1 is UCB.
        """
        for agent in range(N):
            # Only process agents who still have multiple active potential matches
            if active_pairs[agent].sum() > 1:
                # Extract LCB and UCB for all candidates for this specific agent
                # We use only the candidates currently marked as active
                active_indices = np.where(active_pairs[agent] == 1)[0]

                if len(active_indices) <= 1:
                    continue

                UCB = confidence_intervals[agent, active_indices, 1]
                LCB = confidence_intervals[agent, active_indices, 0]

                # Use the static method without 'self'
                to_eliminate_mask = Base_Elimination.overlapping_agents(UCB, LCB)

                # Map the mask back to the original candidate indices
                for idx, should_eliminate in enumerate(to_eliminate_mask):
                    if should_eliminate:
                        actual_candidate_idx = active_indices[idx]
                        active_pairs[agent, actual_candidate_idx] = 0

        return active_pairs

    @staticmethod
    def update_ci(N_p, N_a, delta, active_p_a, active_a_p, mean_est_p, mean_est_a, confidence_intervals_p,
                  confidence_intervals_a, num_plays):
        for player in range(N_p):
            for arm in np.nonzero(active_p_a[player])[0]:
                t = num_plays[player, arm]
                if t == 0:
                    print("Error")
                    continue
                ucb_factor = np.sqrt(2 * np.log(8 * N_p * N_a * (t ** 2) / delta) / (t))
                confidence_intervals_p[player, arm, 0] = mean_est_p[player, arm] - ucb_factor
                confidence_intervals_p[player, arm, 1] = mean_est_p[player, arm] + ucb_factor

        for arm in range(N_a):
            for player in np.nonzero(active_a_p[arm])[0]:
                t = num_plays[player, arm]
                if t == 0:
                    print("Error")
                    continue
                ucb_factor = np.sqrt(2 * np.log(8 * N_p * N_a * (t ** 2) / delta) / (t))
                confidence_intervals_a[arm, player, 0] = mean_est_a[arm, player] - ucb_factor
                confidence_intervals_a[arm, player, 1] = mean_est_a[arm, player] + ucb_factor

    def update_confidence_interval(self):
        self.update_ci(N_p=self.N_p,
                       N_a=self.N_a,
                       delta=self.delta,
                       active_p_a=self.active_p_a,
                       active_a_p=self.active_a_p,
                       mean_est_p=self.mean_est_p,
                       mean_est_a=self.mean_est_a,
                       confidence_intervals_p=self.confidence_intervals_p,
                       confidence_intervals_a=self.confidence_intervals_a,
                       num_plays=self.num_plays)


class Base_Elimination_pref(Base_Elimination):
    """
    this is a version of uniform sampling with not know min delta.
    At every round sample all the pairs.
    It stops when all pairs are eliminated.
    """

    def __init__(self, mean_p, mean_a, N_p, N_a, delta, min_Delta=None):
        # Correctly initialize the base class
        super().__init__(mean_p, mean_a, N_p, N_a, delta)

        self.confidence_intervals_p = np.ones((self.N_p, self.N_a, 2)) * np.inf
        self.confidence_intervals_p[:, :, 0] = - np.inf

        self.confidence_intervals_a = np.ones((self.N_a, self.N_p, 2)) * np.inf
        self.confidence_intervals_a[:, :, 0] = - np.inf

        self.active_p_a = np.ones((self.N_p, self.N_a), dtype=int)
        self.active_a_p = np.ones((self.N_a, self.N_p), dtype=int)
        self.active_pairs = None

        # partial info
        self.partial_pref_p_a = np.zeros((self.N_p, self.N_a, self.N_a), dtype=int)
        self.partial_pref_a_p = np.zeros((self.N_a, self.N_p, self.N_p), dtype=int)

        # Elimination time
        self.elim_time_p_a = np.zeros((self.N_p, self.N_a, self.N_a), dtype=int)
        self.elim_time_a_p = np.zeros((self.N_a, self.N_p, self.N_p), dtype=int)

        # DAGs
        self.pref_dag_p = []
        self.pref_dag_p = []

    @staticmethod
    def overlapping_agents(UCB, LCB):
        n_agents = len(UCB)
        eliminate = np.zeros(n_agents, dtype=bool)
        for i in range(n_agents):
            overlap = False
            for j in range(n_agents):
                if i == j:
                    continue
                # intervals [LCB[i], UCB[i]] and [LCB[j], UCB[j]] overlap if:
                if not (UCB[i] < LCB[j] or UCB[j] < LCB[i]):
                    overlap = True
                    break
            if not overlap:
                eliminate[i] = True
        return eliminate

    @staticmethod
    def get_active_agents_from_partial_pref(partial_pref, active_agents, N):
        for agent in range(N):
            sub_array = np.delete(partial_pref[agent], agent)
            not_done = np.any(sub_array == 0)
            active_agents[agent] = not_done

    @staticmethod
    def get_partial_pref(partial_pref_p, confidence_interval, elimination_time, t):
        N_agents = confidence_interval.shape[0]
        for agent in range(N_agents):
            for agent_2 in range(N_agents):
                if agent == agent_2:
                    continue

                # if partial_pref_p[agent, agent_2] != 0:
                #     continue

                lcb_a = confidence_interval[agent, 0]
                ucb_a = confidence_interval[agent, 1]

                lcb_a2 = confidence_interval[agent_2, 0]
                ucb_a2 = confidence_interval[agent_2, 1]

                if lcb_a > ucb_a2:
                    partial_pref_p[agent, agent_2] = 1
                    partial_pref_p[agent_2, agent] = -1
                    elimination_time[agent, agent_2] = t
                    elimination_time[agent_2, agent] = t

                elif lcb_a2 > ucb_a:
                    partial_pref_p[agent_2, agent] = 1
                    partial_pref_p[agent, agent_2] = -1
                    elimination_time[agent, agent_2] = t
                    elimination_time[agent_2, agent] = t

    @staticmethod
    def update_partial_pref(partial_pref, active_pairs, confidence_intervals, elim_time, N, t):
        for agent in range(N):
            Base_Elimination_pref.get_partial_pref(partial_pref_p=partial_pref[agent],
                                                   confidence_interval=confidence_intervals[agent],
                                                   elimination_time=elim_time[agent],
                                                   t=t)

    @staticmethod
    def update_active_pairs_from_pref(partial_pref, active_pairs, N):
        for agent in range(N):
            Base_Elimination_pref.get_active_agents_from_partial_pref(partial_pref=partial_pref[agent],
                                                                      active_agents=active_pairs[agent],
                                                                      N=N)
