import pulp

from algorithims.matching_algo.linear_programing.BaseLinearModel import BaseLinearModel


class LpStableMatching(BaseLinearModel):
    name = "LpStableMatching"

    def __init__(self, player_preference, worker_preference, N):
        BaseLinearModel.__init__(self,
                                 player_preference=player_preference,
                                 worker_preference=worker_preference,
                                 N=N,
                                 name="LpStableMatching")

    def set_objective(self, **kwargs):
        self.prob += 0, "Arbitrary Objective Function"

    def set_constraints(self, **kwargs):
        self.set_base_constrains()
        self.set_stability_constraints()


class PlayerOptimalStableMatching(BaseLinearModel):
    name = "PlayerOptimalStableMatching"

    def __init__(self, player_preference, worker_preference, N):

        BaseLinearModel.__init__(self,
                                 player_preference,
                                 worker_preference,
                                 N,
                                 name="PlayerOptimalStableMatching")

    def set_objective(self, **kwargs):
        total_worker_rankings = []
        for player in range(self.N):
            for worker in range(self.N):
                total_worker_rankings += [self.x[player][worker] * self.ranking_player[player][worker]]

        self.prob += pulp.lpSum(total_worker_rankings), "Worker Optimal Objective Function"

    def set_constraints(self, **kwargs):
        self.set_base_constrains()
        self.set_stability_constraints()

    def set_additional_solution_constrains(self, run, **kwargs):
        total_player_rankings = []
        current_ranking = []
        for player in range(self.N):
            for worker in range(self.N):
                total_player_rankings += [self.x[player][worker] * self.ranking_player[player][worker]]
                current_ranking += [self.x[player][worker].varValue * self.ranking_player[player][worker]]
        self.prob += pulp.lpSum(total_player_rankings) <= sum(current_ranking), f"Player optimal constrains {run}"


class WorkerOptimalStableMatching(BaseLinearModel):
    name = "WorkerOptimalStableMatching"

    def __init__(self, player_preference, worker_preference, N):

        BaseLinearModel.__init__(self,
                                 player_preference,
                                 worker_preference,
                                 N,
                                 name="WorkerOptimalStableMatching")

    def set_objective(self, **kwargs):
        total_worker_rankings = []
        for player in range(self.N):
            for worker in range(self.N):
                total_worker_rankings += [self.x[player][worker] * self.ranking_worker[worker][player]]

        self.prob += pulp.lpSum(total_worker_rankings), "Worker Optimal Objective Function"

    def set_constraints(self, **kwargs):
        self.set_base_constrains()
        self.set_stability_constraints()

    def set_additional_solution_constrains(self, run, **kwargs):
        total_player_rankings = []
        current_ranking = []
        for player in range(self.N):
            for worker in range(self.N):
                total_player_rankings += [self.x[player][worker] * self.ranking_worker[worker][player]]
                current_ranking += [self.x[player][worker].varValue * self.ranking_player[worker][player]]
        self.prob += pulp.lpSum(total_player_rankings) <= sum(current_ranking), f"Worker optimal constrains {run}"


class EgalitarianStableMatch(BaseLinearModel):
    name = "EgalitarianStableMatch"

    def __init__(self, player_preference, worker_preference, N):

        BaseLinearModel.__init__(self,
                                 player_preference,
                                 worker_preference,
                                 N,
                                 name="EgalitarianStableMatch")

    def set_objective(self, **kwargs):
        total_rank = []
        for player in range(self.N):
            for worker in range(self.N):
                total_rank += [
                    self.x[player][worker] * self.ranking_player[player][worker] +
                    self.x[player][worker] * self.ranking_worker[worker][player]
                ]
        self.prob += pulp.lpSum(total_rank), "Egalitarian stable match"

    def set_constraints(self, **kwargs):
        self.set_base_constrains()
        self.set_stability_constraints()

    def set_additional_solution_constrains(self, run, **kwargs):
        rank_diff = []
        current_ranking = []
        for player in range(self.N):
            for worker in range(self.N):
                rank_diff += [self.x[player][worker] * self.ranking_player[player][worker] +
                              self.x[player][worker] * self.ranking_worker[worker][player]]
                current_ranking += [self.x[player][worker].varValue * self.ranking_player[player][worker] +
                                    self.x[player][worker].varValue * self.ranking_worker[worker][player]]
        self.prob += pulp.lpSum(rank_diff) <= sum(current_ranking), f"Eq optimal constrains {run}"


class SexEqualStableMatching(BaseLinearModel):
    name = "SexEqualStableMatching"

    def __init__(self, player_preference, worker_preference, N, delta):
        self.delta = delta
        self.t = pulp.LpVariable("dummy",
                                 lowBound=0,
                                 cat=pulp.LpInteger)
        BaseLinearModel.__init__(self,
                                 player_preference,
                                 worker_preference,
                                 N,
                                 name="SexEqualStableMatching")

    def set_objective(self, **kwargs):
        self.prob += self.t, "Sex-Equal stable match"

    def set_abs_constrains(self):
        rank_diff = []
        for player in range(self.N):
            for worker in range(self.N):
                rank_diff += [
                    self.x[player][worker] * self.ranking_player[player][worker] -
                    self.x[player][worker] * self.ranking_worker[worker][player]]

        self.prob += self.t >= (pulp.lpSum(rank_diff) + self.delta), "abs_constrains_1"
        self.prob += self.t >= - (pulp.lpSum(rank_diff) + self.delta), "abs_constrains_2"

    def set_constraints(self, **kwargs):
        self.set_base_constrains()
        self.set_stability_constraints()
        self.set_abs_constrains()

    def set_additional_solution_constrains(self, run, **kwargs):
        rank_diff = []
        current_diff = []
        for player in range(self.N):
            for worker in range(self.N):
                rank_diff += [
                    self.x[player][worker] * self.ranking_player[player][worker] -
                    self.x[player][worker] * self.ranking_worker[worker][player]]
                current_diff += [self.x[player][worker].varValue * self.ranking_player[player][worker] -
                                 self.x[player][worker].varValue * self.ranking_worker[worker][player]]
        self.prob += pulp.lpSum(rank_diff) + self.delta <= abs(sum(current_diff) + self.delta), f"Sex-equal constrains {run}"
        self.prob += -(pulp.lpSum(rank_diff) + self.delta) <= abs(sum(current_diff) + self.delta), f"Sex-equal constrains 2 {run}"
        # print("total rank diff:", sum(current_diff))
        # print("total abs rank diff:", abs(sum(current_diff) + self.delta))