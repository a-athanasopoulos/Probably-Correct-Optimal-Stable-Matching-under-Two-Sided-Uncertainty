import networkx as nx
from collections import defaultdict
import numpy as np


def create_preference_dag_final(agent_name, other_agents, partial_order):
    """
    Creates a DAG representing an agent's partial preferences.

    Args:
        agent_name: String name of the agent (e.g., 'e1') [cite: 77]
        other_agents: List of candidate names on the opposite side [cite: 72]
        partial_order: A list of sets, where each set is an equivalence class.
                      Lower index = higher preference. [cite: 79, 81]
    """
    dag = nx.DiGraph(name=f"Preferences for {agent_name}")

    # Pre-populate the graph with all potential candidates as nodes [cite: 72, 87]
    dag.add_nodes_from(other_agents)

    # Add edges between equivalence classes to represent strict partial ordering [cite: 76, 86]
    for i in range(len(partial_order) - 1):
        current_tier = partial_order[i]
        next_tier = partial_order[i + 1]

        for higher in current_tier:
            for lower in next_tier:
                # An edge from 'higher' to 'lower' means higher is preferred [cite: 90]
                dag.add_edge(higher, lower)

    return dag


def get_head(dag):
    heads = [n for n, d in dag.in_degree() if d == 0]
    return heads


def incomparable_nodes(G, v):
    ancestors = nx.ancestors(G, v)
    descendants = nx.descendants(G, v)
    return set(G.nodes()) - ancestors - descendants - {v}


def check_stoping_cond(employer, dag, engaged):
    heads = get_head(dag)
    cond = (engaged == set(heads)) or (len(heads) == 0)
    return cond


def get_condition_all(employers, employer_dags, engaged):
    conds = []
    for emp in employers:
        cond = check_stoping_cond(employer=emp, dag=employer_dags[emp], engaged=engaged[emp])
        conds += [cond]

    done = np.all(conds)
    return done


def invert_e_to_a(D):
    inv = defaultdict(set)
    for e, A in D.items():
        for a in A:
            inv[a].add(e)
    return dict(inv)


def bypass_node(G, v):
    preds = list(G.predecessors(v))
    succs = list(G.successors(v))

    # connect predecessors directly to successors
    for p in preds:
        for s in succs:
            if p != s:
                G.add_edge(p, s)

    G.remove_node(v)


def delete_pair(e, a, dag_e, dag_a):
    dag_a.remove_node(e)
    bypass_node(dag_e, a)


def super_smp_final(employers, applicants, employer_dags, applicant_dags):
    engaged = {e: set() for e in employers}
    proposed = {a: False for a in applicants}

    i = 0
    while not get_condition_all(employers, employer_dags, engaged):
        i += 1
        # print("round i:", i)
        for emp in employers:
            heads = get_head(employer_dags[emp])
            if set(heads) != engaged[emp]:
                # print("AAA")
                for a in heads:
                    proposed[a] = True
                    engaged[emp].add(a)
                    succesors = list(nx.descendants(applicant_dags[a], emp))
                    for e_prim in succesors:
                        if a in engaged[e_prim]:
                            engaged[e_prim].remove(a)

                        delete_pair(e=e_prim,
                                    a=a,
                                    dag_e=employer_dags[e_prim],
                                    dag_a=applicant_dags[a])
        # print("engaged,", engaged)
        inv_engaged = invert_e_to_a(engaged)
        for a in inv_engaged.keys():
            if len(inv_engaged[a]) > 1:
                e_list = inv_engaged[a]

                deletions = []
                for e in e_list:
                    engaged[e].remove(a)
                    if e not in applicant_dags[a].nodes():
                        continue

                    descendants = list(nx.descendants(applicant_dags[a], e))
                    for d_e in descendants:
                        deletions += [(d_e, a)]

                    incompatible = incomparable_nodes(applicant_dags[a], e)
                    for i_e in incompatible:
                        # print(i_e, a)
                        deletions += [(i_e, a)]

                for (tmp_e, tmp_a) in set(deletions):
                    delete_pair(e=tmp_e,
                                a=tmp_a,
                                dag_e=employer_dags[tmp_e],
                                dag_a=applicant_dags[tmp_a])
    # 3. Maximum Cardinality Matching 
    G = nx.Graph()
    G.add_nodes_from(employers + applicants)
    for a, fiancees in engaged.items():
        for e in fiancees:
            G.add_edge(e, a)

    # Use Hopcroft-Karp to find the matching
    full_matching = nx.bipartite.matching.hopcroft_karp_matching(G, top_nodes=employers)

    # 4. Mandatory Last Check [cite: 216, 223, 261]
    # If any applicant received a proposal but is unmatched, no super-stable matching exists
    for a in applicants:
        if proposed[a] and a not in full_matching:
            return False

    # Return only the Employer -> Applicant pairings for clarity
    m = {e: full_matching[e] for e in employers if e in full_matching}
    return m


def is_pervasive_final(employers, applicants, employer_dags, applicant_dags, Z):
    """
    Check pervasiveness of a (super-stable) matching Z using the paper's G(I) acyclicity test.

    Inputs
    ------
    employers : list[str]
    applicants : list[str]
    employer_dags : dict[e -> nx.DiGraph]   # nodes are applicants
    applicant_dags: dict[a -> nx.DiGraph]   # nodes are employers
    Z : dict[e -> a]                        # matching (each e matched to one a)

    Assumption: edges in each DAG go from better -> worse.
    """

    def strictly_prefers(P: nx.DiGraph, x, y) -> bool:
        # x ≻ y
        if x == y:
            return False
        if not (P.has_node(x) and P.has_node(y)):
            return False
        return nx.has_path(P, x, y)

    def weakly_prefers(P: nx.DiGraph, x, y) -> bool:
        # x ⪰ y in the paper’s sense ("can be above in some refinement"):
        # equivalent to NOT(y ≻ x) in a strict partial order
        if x == y:
            return True
        if not (P.has_node(x) and P.has_node(y)):
            return False
        return not nx.has_path(P, y, x)

    # invert Z: applicant -> employer
    Z_inv = {}
    for e, a in Z.items():
        if a is None:
            continue
        if a in Z_inv:
            raise ValueError(f"Applicant {a} matched twice: {Z_inv[a]} and {e}")
        Z_inv[a] = e

    # Build G(I) on applicant vertices
    G_I = nx.DiGraph()
    G_I.add_nodes_from(applicants)

    # For each applicant a, compute M(a) and add edges (a -> Z(e)) for e in M(a)
    for a in applicants:
        if a not in Z_inv:  # unmatched => no edges out
            continue

        Za = Z_inv[a]  # employer matched to a under Z
        Pa = applicant_dags[a]  # a's DAG over employers

        for e in employers:
            if e == Za:
                continue
            if e not in Z:
                continue
            Ze = Z[e]
            if Ze is None:
                continue

            # (1) Z(a) ≻_a e
            if not strictly_prefers(Pa, Za, e):
                continue

            # (2) a ⪰_e Z(e)
            Pe = employer_dags[e]  # e's DAG over applicants
            if not weakly_prefers(Pe, a, Ze):
                continue

            # (3) no e0 strictly between Za and e s.t. e0 strictly prefers a to Z(e0)
            blocked = False
            for e0 in employers:
                if e0 == Za or e0 == e:
                    continue
                if e0 not in Z or Z[e0] is None:
                    continue

                if strictly_prefers(Pa, Za, e0) and strictly_prefers(Pa, e0, e):
                    Pe0 = employer_dags[e0]
                    if strictly_prefers(Pe0, a, Z[e0]):
                        blocked = True
                        break

            if blocked:
                continue

            # e ∈ M(a) => add edge a -> Z(e)
            G_I.add_edge(a, Ze)

    return nx.is_directed_acyclic_graph(G_I)
