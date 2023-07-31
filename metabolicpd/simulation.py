# -*- coding: utf-8 -*-

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as scp

# import seaborn as sns


class LIFE_Network:
    """Implements the pipeline outlined in the associated paper.

    Attributes:
        network (DataFrame): Stores the graph/network for the system.
        df (DataFrame): Stores an indexed list of metabolites and their properties
        mass (ndarray): Current masses for metabolites, indices matching `df`'s indices
        flux (ndarray): Current fluxes for the network, indices matching `df`'s indices

    """

    # TODO: add reasonable file checking and exceptions
    def __init__(self, file=None, mass=None, flux=None):
        # read graph/network from clean file
        if file is None:
            raise ValueError("A file path must be given.")
        else:
            self.network = pd.read_excel(file)
            unique_entries = np.unique(self.network[["tail", "head"]].values)

        # Gather list of metabolites in network
        metabolites = []
        for entry in unique_entries:
            elements = entry.split(", ")
            for ele in elements:
                metabolites.append(ele)
        new_unique_metabolites = list(dict.fromkeys(metabolites))

        # Use names to determine type of metabolites
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

        # use previous lists to construct dictionary which will be used in DataFrame construction
        temp_dict = {
            "name": new_unique_metabolites,
            "type": metabolite_types,
            "fixed": False,
        }
        self.df = pd.DataFrame(temp_dict)

        if mass is None:
            np.random.default_rng()
            self.mass = np.random.rand(len(self.df.index))
        else:
            self.mass = mass

        if flux is None:
            self.flux = np.ones(self.network.shape[0])
        else:
            self.flux = flux

    # NOTE: I've set this up so ideally it will only be called by the "simulation" function once written
    def create_S_matrix(self, mass):
        """Create the 'S' matrix, representing the dynamics for the network x' = S(x) * f.

        The form of the 'S' matrix comes follows LIFE dynamics, with edges in the network corresponding to columns in the
        matrix and metabolites in the network corresponding to the rows of the matrix.

        Args:
            mass (ndarray): Numpy array of masses to construct the S matrix based off of.

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
            for uber in uber_modulators:
                # there should probably be some checking that happens so we don't duplicate the code in the if statements
                if uber[-1] == "+":  # check the last term in the entry, enhancer
                    idx = self.df[self.df["name"] == uber[:-2]].index.to_numpy()[0]
                    # note uber_term will always be greater than one
                    # TODO: pull the function multiplying the uber edge into a class variable
                    uber_term = uber_term * np.exp(mass[idx] / (mass[idx] + 1))
                elif uber[-1] == "-":  # check the last term in the entry, enhancer
                    idx = self.df[self.df["name"] == uber[:-2]].index.to_numpy()[0]
                    # note uber_term will always be less than one
                    uber_term = uber_term / np.exp(mass[idx] / (mass[idx] + 1))

            # Note that I'm vectorizing as much as possible as the actual dataframe will be massive.
            idxs = self.df[self.df["name"].isin(substrates)].index.to_numpy()
            idxp = self.df[self.df["name"].isin(products)].index.to_numpy()
            # Case: Hyperedge
            if len(substrates) > 1 or len(products) > 1:
                # This chunk of code finds min, and sets col values for both substrates and products appropriately
                # TODO: pull min function into class variable and initialize it
                min_sub = np.min(mass[idxs])
                print(f"Minimum result {min_sub}")
                col[idxs] = -1 * min_sub * uber_term
                col[idxp] = min_sub * uber_term
            # Case: Not Hyperedge
            else:
                if (
                    self.df.loc[self.df["name"] == substrates[0], "type"].item()
                    == "source"
                ):
                    # TODO: get rid of fixed one and add a function which takes index or name and returns weight (maybe between 0 and 1)
                    col[idxp] = 1
                elif (
                    self.df.loc[self.df["name"] == products[0], "type"].item() == "sink"
                ):
                    col[idxs] = -1 * mass[idxs] * uber_term
                else:
                    col[idxs] = -1 * mass[idxs] * uber_term
                    col[idxp] = mass[idxs] * uber_term
            s_matrix.append(col)
        return np.array(s_matrix).T

    def __s_function(self, t, x):
        fixed_idx = self.df[self.df["fixed"]].index.to_numpy()
        der = np.matmul(self.create_S_matrix(x), self.flux)
        # set to zero bevause its the der and we want it constant
        der[fixed_idx] = 0.0
        return der

    def simulate(self, t_0, t):
        """Runs the simulation."""
        sol = scp.solve_ivp(
            self.__s_function,
            (t_0, t),
            self.mass,
            t_eval=np.linspace(t_0, t),
        )
        return sol

    # TODO: Right now the vals need to be ordered by the
    # index not the order the names are entered in
    def fixMetabolites(self, mtbs, vals):
        """Sets fixed flag to true and mass value to val."""
        self.df.loc[self.df["name"].isin(mtbs), ["fixed"]] = True
        idxs = self.df[self.df["name"].isin(mtbs)].index.to_numpy()
        self.mass[idxs] = vals

    def setInitialValue(self, mtbs, vals):
        """Sets mass value to vals."""
        idxs = self.df[self.df["name"].isin(mtbs)].index.to_numpy()
        self.mass[idxs] = vals


if __name__ == "__main__":
    flux = np.random.default_rng().uniform(0.1, 0.8, 28)
    # network = LIFE_Network("data/simple_pd_network.xlsx", mass=None, flux=flux)
    network = LIFE_Network("data/simple_pd_network.xlsx", mass=None, flux=None)

    # To fix clearance_0 at 0.0 for whole runtime
    # the masses need to be in the order of indices not the order of metabolite names
    # network.fixMetabolites(["gba_0", "clearance_0"], [0.0, 2.5])
    # To match old simulation example
    network.fixMetabolites(["gba_0"], [2.5])
    network.setInitialValue(["clearance_0"], [0.0])
    print(network.df.to_markdown())
    result = network.simulate(0, 12)
    print(result.message)
    print(network.df.to_markdown())
    # takes in xlsx, (optional) initial mass/flux, (optional) simulation time
    # gives result of simulation, interfaces for plotting/saving/analysing

    # It makes sense to have another class specifically as an interface for plotting

    # So at the very least I should have one more class for reading in xlsx and cleaning the data

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
