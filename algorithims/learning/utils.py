import numpy as np
def aggregate_metrics(metrics):
    # aggregate metrics
    correct_list = []
    sample_complexity = []
    correct_t_list = []
    for exp_n in range(len(metrics)):
        correct_list += [metrics[exp_n]["correct"]]
        sample_complexity += [metrics[exp_n]["sample_complexity"]]
        correct_t_list += [metrics[exp_n]["correct_t"]]

    ragged_array = np.array(correct_t_list, dtype=object)
    max_len = max(len(row) for row in ragged_array)
    padded = np.array([
        np.pad(np.array(row, dtype=float), (0, max_len - len(row)), constant_values=np.nan)
        for row in ragged_array
    ])

    final_mean = np.nanmean(padded, axis=0)
    return sample_complexity, correct_list, final_mean


def get_rr_matching(t, N_p):
    """Round Robin Matching"""
    tmp_match = [(t + player) % N_p for player in range(N_p)]
    return tmp_match
