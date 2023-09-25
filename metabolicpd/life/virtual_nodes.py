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

    num_nodes = node_names.size
    for i in range(num_nodes):
        print(node_names[i])
    print(num_nodes)


if __name__ == "__main__":
    add_nodes("data/simple_pd_network_no_sinks.xlsx")
