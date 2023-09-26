import pprint

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
    # node_idx = dict(zip(node_names, range(num_nodes)))
    node_adj = {n: [] for n in node_names}

    for i in range(num_nodes):
        print(node_names[i])
    print(num_nodes)

    # NOTE: Using a weighted incidence matrix to store info
    # Each integer value corresponds to an edge,
    # Not listing uberedges as it does not impact source/sink locations
    # TODO: is line above true

    # Get list of nodes which are adj to given node
    for row in edge_list[["tail", "head"]].itertuples():
        t = [i for i in row.tail.split(", ")]
        h = [i for i in row.head.split(", ")]
        for i in t:
            node_adj[i] = node_adj[i] + h

        pp.pprint(node_adj)
        print(h)
        print(t)
        print(row.Index)
        print("######")


def longest_path_tail(adj, n, visited):
    """Given a node, find the longest path from that node."""
    visited[n] = True
    for i in range(adj.shape[0]):
        if not visited[i]:
            pass


if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000)
    add_nodes("data/simple_pd_network_no_sinks.xlsx")
