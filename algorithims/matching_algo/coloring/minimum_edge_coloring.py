# edge coloring using recoloring algorithm described in the introduction of
# https://link.springer.com/content/pdf/10.1007/BF00998632.pdf
# recoloring path using augmenting path
import itertools
from collections import defaultdict


def fixed_matching(matching, num_players):
    matching.sort(key=lambda x: x[0])
    fixed_matching = []
    for (p, a) in matching:
        fixed_matching += [(p,a - num_players)]
    return fixed_matching


def transform_to_complete_matching(matching, num_players, num_arms):
    max_degree = min(num_players, num_arms)
    matching_degree = len(matching)
    complete_m = matching.copy()
    if matching_degree < max_degree:
        # All agents on both sides
        left_agents = set(range(num_players))
        right_agents = set(range(num_players, num_players + num_arms))

        # Extract matched agents from the current matching
        matched_left = set(p for p, _ in matching)
        matched_right = set(a for _, a in matching)

        # Find unmatched agents
        unmatched_left = list(left_agents - matched_left)
        unmatched_right = list(right_agents - matched_right)

        # Create a complete matching
        while matching_degree < max_degree:
            # Copy existing matching
            a = unmatched_left.pop()
            b = unmatched_right.pop()
            complete_m.append((a, b))
            matching_degree = len(complete_m)

    return complete_m



def matching_from_edge_colors(num_players, num_arms, edge_colors, possible_colors):
    matchings = {c: [] for c in possible_colors}
    for i in range(num_players):
        for j in range(num_arms):
            if (i, j + num_players) in edge_colors:
                c = edge_colors[i, j + num_players]
                matchings[c] += [(i, j + num_players)]
    return matchings


def recoloring(edge_colors, vertex_colors, augmenting_path, color_a, color_b):
    """ fix this remove add color"""
    i = 0
    for (v1, v2) in augmenting_path:
        vertex_colors[v1].discard(color_a)
        vertex_colors[v2].discard(color_a)
        vertex_colors[v1].discard(color_b)
        vertex_colors[v2].discard(color_b)

    for (v1, v2) in augmenting_path:
        if i % 2 == 0:
            c1 = color_b
        else:
            c1 = color_a
        edge_colors[(v1, v2)] = c1
        edge_colors[(v2, v1)] = c1
        vertex_colors[v1].add(c1)
        vertex_colors[v2].add(c1)
        i += 1
    return edge_colors, vertex_colors


def find_augmenting_path(v1, a, b, adjacency_list, edge_colors):
    augmenting_path = []
    i = 0
    flag = True
    while flag:
        i += 1
        if i % 2 == 0:
            c = b
        else:
            c = a
        flag = False
        for v2 in adjacency_list[v1]:
            if (v1, v2) in edge_colors:
                if edge_colors[(v1, v2)] == c:
                    augmenting_path += [(v1, v2)]
                    v1 = v2
                    flag = True
                    break
                else:
                    flag = False
    return augmenting_path


def edge_coloring(num_players, num_arms, adjacency_list, max_degree):
    possible_colors = set(range(max_degree))
    # Dictionary to store the color of each edge
    edge_colors = {}

    # Dictionary to store used colors for each vertex
    vertex_colors = defaultdict(set)

    # Step 1: Iterate over each edge in the bipartite graph
    for u in range(num_players):
        for w in adjacency_list[u]:
            if (u, w) not in edge_colors:
                # Step 2: Find the smallest available color for the edge (u, v)
                a = possible_colors.difference(vertex_colors[u])
                b = possible_colors.difference(vertex_colors[w])

                available_colors = a.intersection(b)
                if len(available_colors) >= 1:  # if available color
                    # Find the smallest color that is not used by adjacent edges
                    color = min(available_colors)

                    # Step 3: Assign the color to the edge (u, v)
                    edge_colors[(u, w)] = color
                    edge_colors[(w, u)] = color
                    vertex_colors[u].add(color)
                    vertex_colors[w].add(color)
                else:  # if not available color we need to recolor by contrating augemneting path
                    a = min(a)
                    b = min(b)
                    aug_path = find_augmenting_path(v1=w,
                                                    a=a,
                                                    b=b,
                                                    adjacency_list=adjacency_list,
                                                    edge_colors=edge_colors)
                    edge_colors, vertex_colors = recoloring(edge_colors=edge_colors, vertex_colors=vertex_colors,
                                                            augmenting_path=aug_path, color_a=a, color_b=b)
                    edge_colors[(u, w)] = a
                    edge_colors[(w, u)] = a
                    vertex_colors[u].add(a)
                    vertex_colors[w].add(a)
    matchings = matching_from_edge_colors(num_players, num_arms, edge_colors, possible_colors)
    return matchings


def get_matchings_edge_coloring(num_players, num_arms, adjacency_matrix, complete_flag=False):
    #  1. find degree
    degree_players = max(adjacency_matrix.sum(axis=1))
    degree_arms = max(adjacency_matrix.sum(axis=0))
    max_degree = int(max(degree_players, degree_arms))

    #  2. create adjacency_list
    adjacency_list = defaultdict(list)

    for p in range(num_players):
        for a in range(num_arms):
            if adjacency_matrix[p][a] == 1:
                adjacency_list[p] += [a + num_players]
                adjacency_list[a + num_players] += [p]

    matchings = edge_coloring(num_players=num_players,num_arms=num_arms, adjacency_list=adjacency_list, max_degree=max_degree)
    complete_matchings = []
    for matching_id in matchings:
        if complete_flag:
            complete_matching = transform_to_complete_matching(matching=matchings[matching_id],
                                                               num_players=num_players,
                                                               num_arms=num_arms)
            fixed_complete_matching = fixed_matching(complete_matching, num_players)
            assert len(fixed_complete_matching) == min(num_arms, num_players)
        else:
            complete_matching = matchings[matching_id]
            fixed_complete_matching = fixed_matching(complete_matching, num_players)

        complete_matchings += [fixed_complete_matching]
    return complete_matchings


if __name__ == '__main__':
    import numpy as np
    import time

    MATRIX = 1 - np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
    matchings_to_sample = get_matchings_edge_coloring(num_players=5,
                                                      adjacency_matrix=MATRIX)


    def generate_all_possible_constraints(K):
        all_constraints = []
        for binary_tuple in itertools.product([0, 1], repeat=K * K):
            constraint_matrix = np.array(binary_tuple).reshape(K, K)
            all_constraints.append(constraint_matrix)
        return all_constraints


    num_agents = 3
    all_constraints = generate_all_possible_constraints(num_agents)

    start = time.time()
    for idx, constraint_matrix in enumerate(all_constraints):
        max_player = (1 - constraint_matrix).sum(axis=1).max()
        max_arms = (1 - constraint_matrix).sum(axis=0).max()
        max_matchings = max(max_arms, max_player)
        matchings_to_sample = get_matchings_edge_coloring(num_players=num_agents,
                                                          adjacency_matrix=1 - constraint_matrix)
        if len(matchings_to_sample) > max_matchings:
            print("=======================================================")
            print(f"\nConstraint Matrix {idx + 1}:")
            print(f"constraint_matrix:\n", constraint_matrix)
            print(max_matchings, " \n")
            print(max_matchings, len(matchings_to_sample))
            print("=======================================================")
    end = time.time()
    print(end - start)
