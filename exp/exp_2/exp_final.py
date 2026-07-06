from algorithims.learning.agregator import Aggregator, load_pickle
from algorithims.learning.utils import aggregate_metrics


def run_instances(instances_loaded, delta, base_path):
    #
    print("RUN Uniform no Delta")
    from algorithims.learning.U import U
    agg1 = Aggregator(
        save_path=base_path + f"/U_{N}_D_{Delta_min}_delta_{delta}.pkl",
        verbose=True)

    metrics1, c_list = agg1.run_instances(algorithm_cls=U,
                                          algorithm_parameters=(delta, None),
                                          instances=instances_loaded)
    metrics_elim_uniform = aggregate_metrics(metrics1)
    print(metrics_elim_uniform)

    print("RUN Elimination no RS")
    from algorithims.learning.E_NS import E_NS
    agg2 = Aggregator(save_path=base_path + f"/E_NS_N_{N}_D_{Delta_min}_delta_{delta}.pkl",
                      verbose=True)

    metric2, c_list = agg2.run_instances(algorithm_cls=E_NS,
                                         algorithm_parameters=(delta, None),
                                         instances=instances_loaded)
    metrics_elim_ns = aggregate_metrics(metric2)
    print(metrics_elim_ns)

    print("RUN Elimination PSM")
    from algorithims.learning.E_psm import E_PSM

    agg3 = Aggregator(save_path=base_path + f"/E_PSM_N_{N}_D_{Delta_min}_delta_{delta}.pkl",
                      verbose=True)

    metrics3, c_list = agg3.run_instances(algorithm_cls=E_PSM,
                                          algorithm_parameters=(delta, None),
                                          instances=instances_loaded)
    metrics_elim_psm = aggregate_metrics(metrics3)
    print(metrics_elim_psm)

    print("RUN Extended Elimination PSM")

    from algorithims.learning.EE_psm import EE_PSM

    agg4 = Aggregator(
        save_path=base_path + f"/EE_PSM_N_{N}_D_{Delta_min}_delta_{delta}.pkl",
        verbose=True)

    metrics4, c_list = agg4.run_instances(algorithm_cls=EE_PSM,
                                          algorithm_parameters=(delta, None),
                                          instances=instances_loaded)
    metrics_elim_psm_rs = aggregate_metrics(metrics4)
    print(metrics_elim_psm_rs)

    print("RUN Uniform PSM")
    from algorithims.learning.U_psm import U_PSM_pac

    agg5 = Aggregator(
        save_path=base_path + f"/U_PSM_N_{N}_D_{Delta_min}_delta_{delta}.pkl",
        verbose=True)

    metrics5, c_list = agg5.run_instances(algorithm_cls=U_PSM_pac,
                                          algorithm_parameters=(delta, None),
                                          instances=instances_loaded)
    metrics_uniform_psm_rs = aggregate_metrics(metrics5)
    print(metrics_uniform_psm_rs)

    return


if __name__ == '__main__':
    delta = 0.1
    for Delta_min in [0.2]:
        for N in [2, 3, 5, 10, 20]:
            print(f"RUN N = {N}")
            base_path = "./instances"
            save_path = "./results"
            instances_loaded = load_pickle(base_path + f"/N_{N}_Delta_min_{Delta_min}.pkl")
            run_instances(instances_loaded, delta=delta, base_path=save_path)
