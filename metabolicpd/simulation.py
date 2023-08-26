# -*- coding: utf-8 -*-

import logging  # for sending timer logs to a file
import re

import platform
import codetiming as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as scp
import seaborn as sns

is_logging = True
log_to_stream = False

# Setup for always logging to file and logging to stream when level at warning
logger = logging.getLogger("simulation")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("data/time.log")
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)


# def of conditional decorator
# Note: Right now changing logger level between info and warning toggles printing to console
def conditional_timer(timer_name, to_stream=log_to_stream, condition=is_logging):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        if to_stream:
            return ct.Timer(
                name=timer_name,
                text="{name} time: {:.4f} seconds",
                logger=logger.warning,
            )(func)
        else:
            return ct.Timer(
                name=timer_name, text="{name} time: {:.4f} seconds", logger=logger.info
            )(func)

    return decorator


# TODO: Create option to save numpy array to avoid initializing same data over and over
class LIFE_Network:
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

    @conditional_timer("__init__")
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
            self.network = pd.read_excel(file)
            unique_entries = np.unique(self.network[["tail", "head"]].values)

        # Gather list of metabolites in network
        metabolites = []
        for entry in unique_entries:
            elements = entry.split(", ")
            for ele in elements:
                metabolites.append(ele)
        new_unique_metabolites = list(dict.fromkeys(metabolites))
        self.num_mtb = len(new_unique_metabolites)
        self.num_edges = self.network.shape[0]

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
            self.ffunc = lambda mass, idx: np.exp(mass[idx] / (mass[idx] + 1))
        else:
            self.ffunc = ffunc

        if min_func is None:
            self.min_func = lambda mass, idxs: np.min(mass[idxs])
        else:
            self.min_func = min_func

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

    def __mass_diagonal(self, mass):
        mass_diag = np.zeros((self.num_edges, self.num_edges))
        # Temp import from old method for testing
        for row in self.network[["tail", "head", "uber"]].itertuples():
            row_index = row.Index
            idxs = self.substrates[row_index]
            min_sub = self.min_func(mass, idxs)
            mass_diag[row_index, row_index] = min_sub
        return mass_diag

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

    def __s_function(self, t, x):
        fixed_idx = self.mtb[self.mtb["fixed"]]["index"]
        # der = np.matmul(self.create_S_matrix(x), self.flux)
        der = self.create_S_matrix(x) @ self.flux
        # set to zero bevause its the der and we want it constant
        for i in fixed_idx:
            der[i] = self.fixed_trajectories[i](t)
        return der

    # TODO: Look into switching to scipy.integrate.RK45 explicity
    # Allows access to step function which would make setting specific values easier
    # rough comments until I know things work
    # t_0 : start, t: end, t_eval: list of points between t_0,t that the func will evaluate at.
    @conditional_timer("simulate")
    def simulate(self, rtol=1e-4, atol=1e-5):
        """Runs the simulation."""
        sol = scp.solve_ivp(
            self.__s_function,
            (self.t_0, self.t),
            self.mass,
            t_eval=self.t_eval,
            rtol=rtol,
            atol=atol,
        )
        return sol

    @conditional_timer("fixMetabolite")
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


# TODO: Generalize this and give docstring
def basic_graph(result):
    sns.set_theme()
    sns.set_style("dark")
    sns.color_palette("pastel")
    sns.set_context("talk")
    # sns.despine(offset=10, trim=True)
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
    sns.despine(offset=10, trim=True)
    plt.show()


# TODO: Fill this out for general use
def print_timer_stats():
    print(
        "Count: {0}\nTotal: {1}\nMax: {2}\nMin: {3}\nStdev: {4}".format(
            ct.Timer.timers.count("create_S_matrix"),
            ct.Timer.timers.total("create_S_matrix"),
            ct.Timer.timers.max("create_S_matrix"),
            ct.Timer.timers.min("create_S_matrix"),
            ct.Timer.timers.stdev("create_S_matrix"),
        )
    )


if __name__ == "__main__":
    # Setup different plt backend for kitty term
    if platform.system() == "Linux":
        plt.switch_backend("module://matplotlib-backend-kitty")

    # TODO: make a list of a couple interesting argument values for test cases
    # TODO: design dataframe for outputting simulation results
    # TODO: Write function that saves resultant data to file
    # TODO: Rewrite create_S_matrix to vectorize the row operations

    pd.set_option("display.precision", 2)

    network = LIFE_Network(
        file="data/simple_pd_network.xlsx",
        mass=None,  # Makes the masses random via constructor
        flux=np.random.default_rng().uniform(0.1, 0.8, 28),
        ffunc=lambda mass, idx: np.exp(mass[idx] / (mass[idx] + 1)),
        min_func=lambda mass, idxs: np.min(mass[idxs]),
        source_weights=None,
        t_0=0,
        t=15,
        num_samples=500,
    )

    # setting trajectory using ndarray
    # network.fixMetabolite("gba_0", 2.5, -np.sin(network.t_eval), isDerivative=True)
    network.fixMetabolite("gba_0", 2.5)
    # setting trajectory using ufunc
    # network.fixMetabolite("a_syn_0", 1.5)
    # Set initial value without fixing metabolite
    network.setInitialValue("clearance_0", 0.0)

    result = network.simulate()
    logger.warning(result.message)

    # Create and print results to file
    respr = pd.DataFrame(result.y.T, columns=network.mtb["name"])
    respr.insert(0, "time", result.t)
    respr.to_csv("data/results.csv")

    # takes in xlsx, (optional) initial mass/flux, (optional) simulation time
    # gives result of simulation, interfaces for plotting/saving/analysing

    # print(network.mtb)
    basic_graph(result)
