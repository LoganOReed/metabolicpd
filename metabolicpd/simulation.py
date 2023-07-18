import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as scp

# from scipy.integrate import odeint


class LIFE_Network:
    """Implements the pipeline outlined in the related paper."""

    # TODO: add reasonable file checking and exceptions
    # TODO: add constructor that allows user to pick initial flux/mass
    # TODO: Decide if I should keep curr_mass in class or extract it
    def __init__(self, file):
        # read graph/network from clean file
        self.network = pd.read_excel(file)
        unique_entries = np.unique(self.network[["tail", "head"]].values)

        metabolites = []
        for entry in unique_entries:
            elements = entry.split(", ")
            for ele in elements:
                metabolites.append(ele)
        new_unique_metabolites = list(dict.fromkeys(metabolites))

        metabolite_types = []
        for ele in new_unique_metabolites:
            # Using regular expression strings to identify source or sink terms in metabolite dictionary keys
            sink_res = re.search("^[e]+[0-9]$", ele)  # starts with e, ends with #...
            source_res = re.search("^[s]+[0-9]$", ele)  # starts with s, end with #...
            if (
                sink_res is not None
            ):  # check if we found something in the entry that matches the format of 'e###'
                metabolite_types.append("sink")
            elif (
                source_res is not None
            ):  # chick if we found something that matches format of source term
                metabolite_types.append("source")
            else:  # if we didn't find a source or sink term, then it must be an actual metabolite!
                metabolite_types.append("actual")

        temp_dict = {
            "index": [i for i in range(0, len(new_unique_metabolites))],
            "name": new_unique_metabolites,
            "type": metabolite_types,
        }
        self.df = pd.DataFrame(temp_dict)
        # initialize random masses for metabolites
        np.random.seed(seed=12)
        self.curr_mass = np.random.rand(len(self.df.index))
        self.df = self.df.assign(mass=self.curr_mass)
        self.flux = np.ones(self.network.shape[0])

    # NOTE: I've set this up so ideally it will only be called by the "simulation" function once written
    def create_S_matrix(self, mass):
        """Create the 'S' matrix, representing the dynamics for the network x' = S(x) * f.

        The form of the 'S' matrix comes follows LIFE dynamics, with edges in the network corresponding to columns in the
        matrix and metabolites in the network corresponding to the rows of the matrix.

        Args:
            mass : Numpy array of masses to construct the S matrix based off of.

        Returns:
            A numpy array representing the S matrix for the current metabolite masses.
        """
        s_matrix = []
        edge_columns_df = self.network[
            ["tail", "head", "uber"]
        ]  # get a dataframe for just the edges (future-proof for uberedges)
        for row in edge_columns_df.itertuples():
            # iterate through each of the edges
            # there could be more than one substrate
            # or product! (for instance, in a hyperedge)
            substrates = row.tail.split(", ")
            products = row.head.split(", ")
            uber_modulators = row.uber.split(", ")

            col = np.zeros(self.df.shape[0])

            # build the uber term for each expression, default if no uber edges is 1
            uber_term = 1.0
            # TODO Ask Chris if we mean to modify the initial uber_term=1 by multiple metabolites
            #      It seems like it's accounting for the various weights of each + and - on an edge
            #      But I still don't fully understand the definition of the S matrix we are using
            for uber in uber_modulators:
                # there should probably be some checking that happens so we don't duplicate the code in the if statements
                if uber[-1] == "+":  # check the last term in the entry, enhancer
                    # TODO Warning this feels very clunky and potentially slow
                    idx = self.df.loc[self.df["name"] == uber[:-2], "index"].to_numpy()[
                        0
                    ]
                    # note uber_term will always be greater than one
                    uber_term = uber_term * np.exp(mass[idx] / (mass[idx] + 1))
                elif uber[-1] == "-":  # check the last term in the entry, enhancer
                    # TODO Warning this feels very clunky and potentially slow
                    idx = self.df.loc[self.df["name"] == uber[:-2], "index"].to_numpy()[
                        0
                    ]
                    # note uber_term will always be less than one
                    uber_term = uber_term / np.exp(mass[idx] / (mass[idx] + 1))

            # Note that I'm vectorizing as much as possible as the actual dataframe will be massive.
            idxs = self.df.loc[self.df["name"].isin(substrates), "index"]
            idxp = self.df.loc[self.df["name"].isin(products), "index"]
            # Case: Hyperedge
            if len(substrates) > 1 or len(products) > 1:
                # This chunk of code finds min, and sets col values for both substrates and products appropriately
                min_sub = np.min(mass[idxs])
                col[idxs] = -1 * min_sub * uber_term
                col[idxp] = min_sub * uber_term
            # Case: Not Hyperedge
            else:
                if (
                    self.df.loc[self.df["name"] == substrates[0], "type"].item()
                    == "source"
                ):
                    col[idxp] = 1
                elif (
                    self.df.loc[self.df["name"] == products[0], "type"].item() == "sink"
                ):
                    col[idxs] = -1 * mass[idxs] * uber_term
                else:
                    col[idxs] = -1 * mass[idxs] * uber_term
                    col[idxp] = mass[idxs] * uber_term
            s_matrix.append(col)
            # Seems to be the most sane way to check this code
            # print(pd.DataFrame(col).to_markdown())
        return np.array(s_matrix).T

    # NOTE: I think Chris uses metabolite mass for flux value as well, but I'm going to leave it constant for now
    def simulate(self, t_0, t):
        """Runs the simulation."""
        print(self.create_S_matrix(self.curr_mass).shape)
        print(self.flux.shape)
        return scp.solve_ivp(
            lambda t, x: np.matmul(self.create_S_matrix(x), self.flux),
            (t_0, t),
            self.curr_mass,
            t_eval=np.linspace(t_0, t),
        )


if __name__ == "__main__":
    network = LIFE_Network("data/simple_pd_network.xlsx")
    print(network.curr_mass)
    print(network.df.to_markdown())
    result = network.simulate(0, 10)
    print(result.message)
    metas_to_plot = [
        "a_syn_0",
        "a_syn_1",
        "a_syn_proto_0",
        "clearance_0",
        "gba_0",
        "glucosylceramide_0",
        "mis_a_syn_0",
        "mis_a_syn_1",
        "mutant_lrrk2_0",
    ]

    metabolites_to_plot = [0, 1, 3, 6, 17, 22, 23, 25]
    label_idx = 0
    for i in metabolites_to_plot:
        plt.plot(result.t, result.y[i], label=metas_to_plot[label_idx])
        label_idx = label_idx + 1
    plt.ylim([0, 3])
    plt.xlabel("$t$")  # the horizontal axis represents the time
    plt.legend()  # show how the colors correspond to the components of X
    plt.show()
    # plt.plot(t, values_of_interest, label=metabolites_to_plot)
    # plt.ylim([0, 3])
    # plt.legend()
    # plt.show()
