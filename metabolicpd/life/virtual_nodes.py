import pprint
from collections import defaultdict

import numpy as np
import pandas as pd

# By Prop 7 in LIFE Approach by Nathaniel J. Merrill
# If an equilibrium for a flux vector exists for nonzero positive edge weights,
# For every vertex v for which there exists a path from Intake to v,
# there exists a path from v to Outtake(Excretion)

# By Prop 12 in same paper, for all v adj to sources, there exists some path
# from v to a sink.

# All nodes need to be contained in a path from a source to a sink

# read graph/network from clean file

pp = pprint.PrettyPrinter(indent=4)


def add_nodes(file):
    """Adds Sources and Sinks to construct a network with a steady state solution."""
    edge_list = pd.read_excel(file)
    # TODO: Ask Chris if nodes which are only incident to uberedges exist.
    # NOTE: Currently only adding sinks
    edge_list_cells = np.unique(
        edge_list[["tail", "head", "uberPos", "uberNeg"]].values
    )
    node_names = []
    for entry in edge_list_cells:
        elements = entry.split(", ")
        for ele in elements:
            if ele != "none":
                node_names.append(ele)
    node_names = np.unique(node_names)
    num_nodes = node_names.size
    node_idx = dict(zip(node_names, range(num_nodes)))

    for i in range(num_nodes):
        print(node_names[i])
    print(num_nodes)

    # NOTE: Using a weighted incidence matrix to store info
    # Each integer value corresponds to an edge,
    # Not listing uberedges as it does not impact source/sink locations
    # TODO: is line above true

    G = defaultdict(list)

    for row in edge_list[["tail", "head"]].itertuples():
        t = [node_idx[i] for i in row.tail.split(", ")]
        h = [node_idx[i] for i in row.head.split(", ")]
        for i in t:
            G[i].extend(h)

    # Build graph dictionary
    pp.pprint(G)
    all_paths = DFS(G, 0)
    max_len = max(len(p) for p in all_paths)
    max_paths = [p for p in all_paths if len(p) == max_len]

    print("All Paths:")
    pp.pprint(all_paths)
    print("Longest Paths:")
    for p in max_paths:
        print("  ", p)
    print("Longest Path Length:")
    print(max_len)


def DFS(G, v, seen=None, path=None):
    """DFS longest path implementation."""
    if seen is None:
        seen = []
    if path is None:
        path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            if G.get(t) is None:
                paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths


if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000)
    add_nodes("data/simple_pd_network_no_sinks.xlsx")
