import pickle


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Aggregator:
    def __init__(self, save_path=None, verbose=True):
        """
        save_fn: optional callable like save_fn(metrics) to persist results.
        verbose: print per-trial info if True.
        """
        self.save_path = save_path
        self.verbose = verbose

    def run_single_instance(self, algorithm_cls, algorithm_parameters, instance, trials):
        """
        algorithm_cls: class (or callable) to instantiate, e.g. Elim_RL_pac
        algorithm_parameters: tuple (delta, min_Delta)
        instance: tuple (mean_p, mean_a, N_p, N_a)
        trials: int
        """
        mean_p, mean_a, N_p, N_a = instance
        delta, min_Delta = algorithm_parameters

        self.c_list = []
        self.metrics = []
        if self.verbose:
            print("START trials ***************************")
        for t in range(trials):
            if self.verbose:
                print(f"RUN trial:{t}")
            algo = algorithm_cls(
                mean_p=mean_p,
                mean_a=mean_a,
                N_p=N_p,
                N_a=N_a,
                delta=delta,
                min_Delta=min_Delta,
            )

            m, c, s = algo.pac_loop()  # keeping your unpacking
            self.metrics.append(algo.get_metrics_dict())
            self.c_list.append(c)

            if self.verbose and 'sample_complexity' in self.metrics[-1]:
                print(self.metrics[-1]['sample_complexity'])

        if self.save_path is not None:
            save_pickle(self.metrics,  self.save_path)

        return self.metrics, self.c_list


    def run_instances(self, algorithm_cls, algorithm_parameters, instances):
        """
        algorithm_cls: class (or callable) to instantiate, e.g. Elim_RL_pac
        algorithm_parameters: tuple (delta, min_Delta)
        instance: tuple (mean_p, mean_a, N_p, N_a)
        trials: int
        """
        delta, min_Delta = algorithm_parameters
        if self.verbose:
            print("START trials ***************************")
        self.c_list = []
        self.metrics = []
        for i, instance in enumerate(instances):
            if self.verbose:
                print(f"RUN trial:{i}")
            mean_p, mean_a, N_p, N_a, min_delta_instance = instance
            algo = algorithm_cls(
                mean_p=mean_p,
                mean_a=mean_a,
                N_p=N_p,
                N_a=N_a,
                delta=delta,
                min_Delta=min_delta_instance,
            )
            m, c, s = algo.pac_loop()  # keeping your unpacking
            self.metrics.append(algo.get_metrics_dict())
            self.c_list.append(c)

        if self.save_path is not None:
            save_pickle(self.metrics,  self.save_path)

        return self.metrics, self.c_list
