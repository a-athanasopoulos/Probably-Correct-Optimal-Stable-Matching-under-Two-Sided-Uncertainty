import copy

from algorithims.matching_algo.super_stable.super_stable_final import super_smp_final, is_pervasive_final


def pervasive_stable_matching(pref_e_dag, pref_a_dag, employers, applicants):
    employer_dags_copy = copy.deepcopy(pref_e_dag)
    applicant_dags_copy = copy.deepcopy(pref_a_dag)
    super_sm = super_smp_final(employers=employers,
                               applicants=applicants,
                               employer_dags=employer_dags_copy,
                               applicant_dags=applicant_dags_copy)
    if super_sm:
        # check if super stable matching is pervasive
        is_pervasive = is_pervasive_final(Z=super_sm,
                                          employers=employers,
                                          applicants=applicants,
                                          employer_dags=pref_e_dag,
                                          applicant_dags=pref_a_dag)
        if is_pervasive:
            return super_sm, True  # the super stable matching is pervasive
        else:
            return super_sm, False  # is not  pervasive_stable_matching
    else:
        return False, False
        # is not  pervasive_stable_matching
