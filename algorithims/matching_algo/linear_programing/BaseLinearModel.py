import pulp

from algorithims.matching_algo.matching_utils import get_rank_function


class BaseLinearModel(object):
    def __init__(self, player_preference, worker_preference, N, name):
        self.name = name
        self.N = N
        # preferences
        self.player_preference = player_preference
        self.worker_preference = worker_preference
        self.ranking_player = get_rank_function(player_preference)
        self.ranking_worker = get_rank_function(worker_preference)

        # define allocation matrix variable
        type_V = pulp.LpInteger  # LpContinuous
        self.x = pulp.LpVariable.dicts(name="AllocationMatrix",
                                       indices=(range(N), range(N)),
                                       lowBound=0,
                                       upBound=1,
                                       cat=type_V)

        self.prob = pulp.LpProblem(name, pulp.LpMinimize)
        self.set_objective()
        self.set_constraints()

    def set_objective(self, **kwargs):
        raise NotImplementedError("Please implement your platform method")

    def run_one(self, **kwargs):
        raise NotImplementedError("Please implement your platform method")

    def set_base_constrains(self):
        for player in range(self.N):
            c_name = "player_limit_{}".format(player)
            self.prob += pulp.lpSum([self.x[player][worker] for worker in range(self.N)]) <= 1, c_name

        for worker in range(self.N):
            c_name = "worker_limit_{}".format(worker)
            self.prob += pulp.lpSum([self.x[player][worker] for player in range(self.N)]) <= 1, c_name

    def set_stability_constraints(self):
        # add blocking constrains
        for player in range(self.N):
            for worker in range(self.N):
                # player is match to a more preferred worker
                index_worker = self.player_preference[player].index(worker)  # index on the preference list of player
                player_block = pulp.lpSum(
                    [self.x[player][self.player_preference[player][more_preferred]]
                     for more_preferred in range(index_worker)]
                )

                # worker is match to a more preferred player
                index_player = self.worker_preference[worker].index(player)
                worker_block = pulp.lpSum(
                    [self.x[self.worker_preference[worker][more_preferred]][worker]
                     for more_preferred in range(index_player)]
                )

                # is matched
                matched = self.x[player][worker]

                # set constrain
                name = "blocking_{}_{}".format(player, worker)
                self.prob += pulp.lpSum([player_block, worker_block, matched]) >= 1, name

    def set_constraints(self, **kwargs):
        raise NotImplementedError("Please implement your platform method")

    def set_exclude_solution_constrains(self, run):
        constrain_list = []
        for player in range(self.N):
            for worker in range(self.N):
                if self.x[player][worker].varValue >= 1:
                    constrain_list += [self.x[player][worker]]
        self.prob += pulp.lpSum(constrain_list) <= self.N - 1, f"solution_constrain_{run}"

    def set_additional_solution_constrains(self, run, **kwargs):
        pass

    def get_match(self):
        match = []
        for player in range(self.N):
            for worker in range(self.N):
                value = self.x[player][worker].varValue
                if value > 0:
                    match.append(worker)
        return match

    def get_evaluation(self):
        return 0

    def run(self, runs=1, additional_exlution_constrains=1, **kwargs):
        status = 1
        matchings = []
        evaluations = []
        run = 1
        while status == 1:
            self.prob.solve(pulp.PULP_CBC_CMD(msg=0))  # se can add some time limits

            status = self.prob.status

            if status == 1:
                # get matching
                match = self.get_match()
                evaluation = self.get_evaluation()
                matchings.append(match)
                evaluations.append(evaluation)

                # add additional constrain to remove solution for all stable matching
                self.set_exclude_solution_constrains(run)
                if additional_exlution_constrains:  # otherwise get all evaluation for the stable matchings
                    self.set_additional_solution_constrains(run)

            if runs == 1:
                status = 0

            run += 1

        return matchings, evaluations
