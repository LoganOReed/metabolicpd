# -*- coding: utf-8 -*-

import platform
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as scp
import seaborn as sns

from metabolicpd.life import util


# TODO: Create option to save numpy array to avoid initializing same data over and over
# TODO: Look into designing better data file format to read in from and add conversion to "io"
# TODO: Look into using GraphML as storage
class Metabolic_Graph:
    """Implements the pipeline outlined in the associated paper.

    Attributes:
        network (DataFrame): Stores the graph/network for the system.
        mtb (ndarray): A Structured numpy array which stores name, type, fixed, and index values for meetabolites
        mass (ndarray): Current masses for metabolites, indices matching `df`'s indices
        flux (ndarray): Current fluxes for the network, indices matching `df`'s indices
        ffunc (function): A differentiable, strictly increasing function used in flux computations.
        min_func (function): A function which chooses the weight of edges between real metabolites
        source_weights (ndarray): An array which contains the weights of each source node
        t_0 (float): Initial time value for simulation
        t (float): Final time value for simulation
        num_samples (int): Number of sample points for simulation to return between t_0 and t.

    """

    # TODO: make a list of a couple interesting argument values for test cases
    # TODO: design dataframe for outputting simulation results
    # TODO: Write function that saves resultant data to file
    def __init__(
        self,
        file=None,
        mass=None,
        flux=None,
        ffunc=None,
        min_func=None,
        source_weights=None,
        t_0=0,
        t=10,
        num_samples=50,
    ):
        # setup simulation evaulation points
        self.t_0 = t_0
        self.t = t
        self.num_samples = num_samples
        self.t_eval, self.step_size = np.linspace(
            self.t_0, self.t, self.num_samples, retstep=True
        )

        # read graph/network from clean file
        if file is None:
            raise ValueError("A file path must be given.")
        else:
            try:
                self.network = pd.read_excel(file)
                unique_entries = np.unique(self.network[["tail", "head"]].values)
            except FileNotFoundError as e:
                print("File not found with given name", e)

        # Gather list of metabolites in network
        metabolites = []
        for entry in unique_entries:
            elements = entry.split(", ")
            for ele in elements:
                metabolites.append(ele)
        new_unique_metabolites = list(dict.fromkeys(metabolites))
        self.num_mtb = len(new_unique_metabolites)
        self.num_edges = self.network.shape[0]

        if mass is None:
            np.random.default_rng()
            self.mass = np.random.rand(self.num_mtb)
        else:
            self.mass = mass

        if flux is None:
            self.flux = np.ones(self.network.shape[0])
        else:
            self.flux = flux

        if ffunc is None:
            # self.ffunc = lambda mass, idx: np.exp(mass[idx] / (mass[idx] + 1))
            self.ffunc = lambda mass, idx: util.hill(mass[idx])
        else:
            self.ffunc = ffunc

        if min_func is None:
            self.min_func = util.min
        else:
            self.min_func = min_func

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

        self.mtb = np.zeros(
            self.num_mtb,
            dtype={
                "names": ("name", "type", "fixed", "index"),
                "formats": ("<U32", "<U6", "bool", "<i4"),
            },
        )
        self.mtb["name"] = new_unique_metabolites
        self.mtb["type"] = metabolite_types
        self.mtb["fixed"] = False
        self.mtb["index"] = np.arange(len(new_unique_metabolites))

        # TODO: Come up with input that is easier for user
        if source_weights is None:
            temp_list = []
            for i in range(self.num_mtb):
                temp_list.append(1)
            self.source_weights = np.array(temp_list)
        else:
            self.source_weights = source_weights

        # Create member dict for potential fixed metabolites
        self.fixed_trajectories = {}

        self.hyper_edges = np.zeros((self.network.shape[0], self.num_mtb))
        self.source_edges = np.zeros((self.network.shape[0], self.num_mtb))
        uber_mods = np.zeros((self.network.shape[0], self.num_mtb))

        self.substrates = []
        # Generate lookup tables for s matrix functions
        # TODO: Check with Chris that generating this once will not break edge cases
        # TODO: Check if columns will always be in this order
        for row in self.network[["tail", "head", "uber"]].itertuples():
            row_idx = row.Index
            sub_str = self.mtb[np.isin(self.mtb["name"], row.tail.split(", "))]["index"]
            prod_snk = self.mtb[np.isin(self.mtb["name"], row.head.split(", "))][
                "index"
            ]

            self.substrates.append(sub_str)

            if self.mtb[sub_str[0]]["type"] == "source":
                self.source_edges[row_idx, prod_snk] = self.source_weights[sub_str[0]]
            elif self.mtb[prod_snk[0]]["type"] == "source":
                self.hyper_edges[row_idx, prod_snk] = 0
                self.hyper_edges[row_idx, sub_str] = -1
            else:
                self.hyper_edges[row_idx, sub_str] = -1
                self.hyper_edges[row_idx, prod_snk] = 1

            u_m = row.uber.split(", ")
            for uber in u_m:
                # there should probably be some checking that happens so we don't duplicate the code in the if statements
                if uber[-1] == "+":  # check the last term in the entry, enhancer
                    uber_mods[
                        row.Index, self.mtb[self.mtb["name"] == uber[:-2]]["index"]
                    ] = 1
                elif uber[-1] == "-":  # check the last term in the entry, enhancer
                    uber_mods[
                        row.Index, self.mtb[self.mtb["name"] == uber[:-2]]["index"]
                    ] = -1

        self.uber_enhancers = np.nonzero(uber_mods == 1)
        self.uber_inhibiters = np.nonzero(uber_mods == -1)

    # NOTE: I've set this up so ideally it will only be called by the "simulation" function once written
    # TODO: Write Documentation for new member variables, write example use case
    def create_S_matrix(self, mass):
        """Create the 'S' matrix, representing the dynamics for the network x' = S(x) * f.

        The form of the 'S' matrix comes follows LIFE dynamics, with edges in the network corresponding to columns in the
        matrix and metabolites in the network corresponding to the rows of the matrix.

        Args:
            mass (ndarray): Numpy array of masses to construct the S matrix based off of.

        Returns:
            A numpy array representing the S matrix for the current metabolite masses.
        """
        uber_diag = self.__uber_diagonal(mass)
        mass_diag = self.__mass_diagonal(mass)
        return ((uber_diag @ mass_diag) @ self.hyper_edges + self.source_edges).T

    # TODO: Docstring
    def __mass_diagonal(self, mass):
        mass_diag = np.zeros((self.num_edges, self.num_edges))
        for row in self.network[["tail", "head", "uber"]].itertuples():
            row_index = row.Index
            idxs = self.substrates[row_index]
            min_sub = self.min_func(mass, idxs)
            mass_diag[row_index, row_index] = min_sub
        return mass_diag

    # TODO: Docstring
    def __uber_diagonal(self, mass):
        u_t = np.zeros(self.num_edges)
        u_t.fill(1.0)
        for i in range(len(self.uber_enhancers[0])):
            u_t[self.uber_enhancers[0][i]] = u_t[
                self.uber_enhancers[0][i]
            ] * self.ffunc(mass, self.uber_enhancers[1][i])

        for i in range(len(self.uber_inhibiters[0])):
            u_t[self.uber_inhibiters[0][i]] = u_t[
                self.uber_inhibiters[0][i]
            ] / self.ffunc(mass, self.uber_inhibiters[1][i])

        return np.diagflat(u_t)

    # TODO: Docstring
    def __s_function(self, t, x):
        fixed_idx = self.mtb[self.mtb["fixed"]]["index"]
        der = self.create_S_matrix(x) @ self.flux
        # replaces computed derivative with one which we control
        for i in fixed_idx:
            der[i] = self.fixed_trajectories[i](t)
        return der

    # TODO: Ask Chris if we need to force x being positive
    # TODO: Docstring
    # Allows access to step function which would make setting specific values easier
    # rough comments until I know things work
    def simulate(self):
        """Runs the simulation."""
        ts = []
        xs = []
        sol = scp.RK45(
            self.__s_function, self.t_0, self.mass, self.t, max_step=self.step_size
        )
        # options are 'running' 'finished' or 'failed'

        while sol.status == "running":
            sol.step()
            ts.append(sol.t)
            xs.append(sol.y)

        tt = np.array(ts)
        yy = np.stack(xs)
        res = {"t": tt, "y": yy.T}
        return res

    def fixMetabolite(self, mtb, val, trajectory=None, isDerivative=False):
        """Sets fixed flag to true and mass value to init val, and gives a derivative function for the trajectory."""
        # Need to be careful to have a scalar index instead of an array to view data instead of copy
        idx = self.mtb[self.mtb["name"] == mtb]["index"][0]
        f_mtb = self.mtb[idx]
        f_mtb["fixed"] = True

        self.mass[idx] = val
        if trajectory is None:
            trajectory = np.zeros(self.num_samples)
            isDerivative = True
        else:
            # convert function to ndarray if needed
            if type(trajectory) is not np.ndarray:
                trajectory = trajectory(self.t_eval)

        if not isDerivative:
            trajectory = np.diff(trajectory) / self.step_size

        self.fixed_trajectories[idx] = lambda t: trajectory[
            int(min(np.floor(t / self.step_size), self.num_samples - 2))
        ]

    def setInitialValue(self, mtb, val):
        """Sets mass value to vals."""
        # All of the lines that look like below are temporary hopefully
        # (From Switching to singleton mtb instead of lists)
        idx = self.mtb[self.mtb["name"] == mtb]["index"][0]
        self.mass[idx] = val


# TODO: Maybe move into class or make plot file
def basic_plot(result, network, mtb_to_plot, ylim=[0, 3]):
    """Creates a plot showing the metabolites `mtb_to_plot` using Metabolic_Graph data."""
    # Setup different plt backend for kitty term
    if platform.system() == "Linux":
        plt.switch_backend("module://matplotlib-backend-kitty")
    sns.set_theme()
    sns.set_style("dark")
    sns.color_palette("pastel")
    sns.set_context("talk")

    metabolites_to_plot = mtb_to_plot
    mtb_names = network.mtb[metabolites_to_plot]["name"]
    label_idx = 0
    for i in metabolites_to_plot:
        plt.plot(result["t"], result["y"][i], label=mtb_names[label_idx])
        label_idx = label_idx + 1
    plt.ylim(ylim)
    plt.xlabel("$t$")  # the horizontal axis represents the time
    plt.legend()  # show how the colors correspond to the components of X
    sns.despine(offset=10, trim=True)
    plt.show()


if __name__ == "__main__":
    print("network.py is not intended to be main, use an example or call in a script.")
