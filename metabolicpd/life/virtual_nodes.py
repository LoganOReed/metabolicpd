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
    edge_list = pd.read_excel(file + ".xlsx")
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
        print(f"{i}: {node_names[i]}")

    # NOTE: Using a weighted incidence matrix to store info
    # Each integer value corresponds to an edge,
    # Not listing uberedges as it does not impact source/sink locations
    G = defaultdict(list)
    GI = defaultdict(list)

    for row in edge_list[["tail", "head"]].itertuples():
        t = [node_idx[i] for i in row.tail.split(", ")]
        h = [node_idx[i] for i in row.head.split(", ")]
        for i in t:
            G[i].extend(h)
        for i in h:
            GI[i].extend(t)

    paths = []
    for i in range(num_nodes):
        paths = paths + longest_path(G, GI, i)
    sources = set()
    sinks = set()
    for p in paths:
        sources.add(p[0])
        sinks.add(p[-1])
    sources = list(sources)
    sinks = list(sinks)
    print(sources)
    print(sinks)

    for i in range(len(sinks)):
        e_name = "e" + str(i)
        print(f"{e_name} for {node_names[sinks[i]]}")
        new_row = pd.Series(
            {
                "tail": node_names[sinks[i]],
                "head": e_name,
                "uberPos": "none",
                "uberNeg": "none",
            }
        )
        edge_list = pd.concat([edge_list, new_row.to_frame().T], ignore_index=True)
        print(edge_list)

    edge_list.to_excel(file + "_virtual.xlsx")


def longest_path(G, GI, v):
    """DFS longest path combining both directions."""
    if GI.get(v):
        rev_paths = longest_path_head(GI.copy(), v)
        max_len = max(len(p) for p in rev_paths)
        max_paths = [p for p in rev_paths if len(p) == max_len]
        if not G.get(v):
            for i in range(len(max_paths)):
                max_paths[i].append(v)
            paths = max_paths
        else:
            paths = longest_path_tail(G.copy(), v, seen=max_paths[0], path=max_paths[0])
    else:
        paths = longest_path_tail(G.copy(), v)
    return paths


def longest_path_tail(G, v, seen=None, path=None):
    """DFS longest path implementation with v as tail."""
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
                paths.append(list(t_path))
            paths.extend(longest_path_tail(G, t, seen[:], t_path))
    return paths


def longest_path_head(G, v, seen=None, path=None):
    """DFS longest path implementation with v as head."""
    if seen is None:
        seen = []
    if path is None:
        path = []

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = [t] + path
            if G.get(t) is None:
                paths.append(list(t_path))
            paths.extend(longest_path_head(G, t, seen[:], t_path))
    return paths


if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000)
    add_nodes("data/simple_pd_network_no_sinks")
