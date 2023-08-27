# -*- coding: utf-8 -*-

import logging  # for sending timer logs to a file
import platform
import re

import codetiming as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as scp
import seaborn as sns

is_logging = True
log_to_stream = True

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

    # TODO: Create local variable that uses t's and num_samples to get max step size for simulation
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
    def simulate(self, rtol=1e-4, atol=1e-5, max_step=np.inf):
        """Runs the simulation."""
        ts = []
        xs = []
        sol = scp.RK45(
            self.__s_function, self.t_0, self.mass, self.t, max_step=max_step
        )
        # options are 'running' 'finished' or 'failed'

        # for i in range(100):
        # sol.step()
        while sol.status == "running":
            sol.step()
            ts.append(sol.t)
            xs.append(sol.y)

        tt = np.array(ts)
        yy = np.stack(xs)
        # print("XSXSXSXS")
        # print(tt)
        # print(tt.shape)
        # print(yy)
        # print(yy.shape)

        res = {"t": tt, "y": yy.T}
        return res

    @conditional_timer("simulate_pos")
    def simulate_prototype_pos_def(
        self, events=None, rtol=1e-4, atol=1e-5, min_step=0.1
    ):
        """Runs the simulation."""
        ts = []
        xs = []
        should_continue = True
        while should_continue:
            sol = scp.solve_ivp(
                self.__s_function,
                (self.t_0, self.t),
                self.mass,
                # t_eval=np.compress(self.t_eval > self.t_0, self.t_eval),
                events=events,
                rtol=rtol,
                atol=atol,
            )
            ts.append(sol.t)
            xs.append(sol.y)
            if sol.status == 1:
                self.t_0 = sol.t[-1]
                # Reset initial state
                self.mass = sol.y[:, -1].copy()
                self.mass = np.clip(self.mass, 0.01, None)
            else:
                should_continue = False
        tt = np.concatenate(ts)
        yy = np.concatenate(xs, axis=1)
        res = {"t": tt, "y": yy}
        return res

    @conditional_timer("fixMetabolite")
    def fixMetabolite(self, mtb, val, trajectory=None, isDerivative=False):
        """Sets fixed flag to true and mass value to init val, and gives a derivative function for the trajectory."""
        # Need to be careful to have a scalar index instead of an array to view data instead of copy
        # TODO: Make exception for out of bounds error
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
def basic_graph(result, network, mtb_to_plot, ylim=[0, 3]):
    sns.set_theme()
    sns.set_style("dark")
    sns.color_palette("pastel")
    sns.set_context("talk")
    # sns.despine(offset=10, trim=True)

    metabolites_to_plot = mtb_to_plot
    mtb_names = network.mtb[metabolites_to_plot]["name"]
    label_idx = 0
    for i in metabolites_to_plot:
        # plt.plot(result.t, result.y[i], label=mtb_names[label_idx])
        plt.plot(result["t"], result["y"][i], label=mtb_names[label_idx])
        label_idx = label_idx + 1
    plt.ylim(ylim)
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


# TODO: Write with numpy so it doesn't take forever
@conditional_timer("Hill")
def hill(x, p=1, k=1.0):
    """x: positive mass of tail node, p: power (int), k some 'dissociation' constant."""
    x_p = pow(x, p)
    return x_p / (k + x_p)


if __name__ == "__main__":
    # Setup different plt backend for kitty term
    if platform.system() == "Linux":
        plt.switch_backend("module://matplotlib-backend-kitty")

    # TODO: make a list of a couple interesting argument values for test cases
    # TODO: design dataframe for outputting simulation results
    # TODO: Write function that saves resultant data to file
    # TODO: Rewrite create_S_matrix to vectorize the row operations

    pd.set_option("display.precision", 2)

    network = Metabolic_Graph(
        file="data/simple_pd_network.xlsx",
        mass=None,  # Makes the masses random via constructor
        flux=np.random.default_rng().uniform(0.1, 0.8, 28),
        ffunc=lambda mass, idx: np.exp(mass[idx] / (mass[idx] + 1)),
        min_func=lambda mass, idxs: np.min(mass[idxs]),
        source_weights=None,
        t_0=0,
        t=15,
        num_samples=50,
    )

    # setting trajectory using ndarray
    # network.fixMetabolite("gba_0", 2.5, -np.sin(network.t_eval), isDerivative=True)
    network.fixMetabolite("gba_0", 2.5)
    # setting trajectory using ufunc
    # network.fixMetabolite("a_syn_0", 1.5)
    # Set initial value without fixing metabolite
    network.setInitialValue("clearance_0", 0.0)

    # result = network.simulate(max_step=0.5)
    result = network.simulate(max_step=0.5)

    # Create and print results to file
    # respr = pd.DataFrame(result.y.T, columns=network.mtb["name"])
    # respr.insert(0, "time", result.t)
    # respr.to_csv("data/results.csv")

    # takes in xlsx, (optional) initial mass/flux, (optional) simulation time
    # gives result of simulation, interfaces for plotting/saving/analysing

    # print(network.mtb)
    basic_graph(result, network, [0, 1, 3, 6, 17, 22, 23, 25])

    # Another Example Case for a network from another paper
    # cctb_network = LIFE_Network(
    #     file="data/central_carbon_tb.xlsx",
    #     mass=None,  # Makes the masses random via constructor
    #     flux=np.random.default_rng().uniform(0.1, 0.8, 20),
    #     ffunc=lambda mass, idx: np.exp(mass[idx] / (mass[idx] + 1)),
    #     min_func=lambda mass, idxs: np.min(mass[idxs]),
    #     source_weights=None,
    #     t_0=0,
    #     t=15,
    #     num_samples=50,
    # )
    # result = cctb_network.simulate()
    # logger.warning(result.message)
    #
    # basic_graph(result, cctb_network, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    # def min_min(mass, idx):
    #     if np.less_equal(mass[idx], 6):
    #         return np.divide(3, np.add(np.power(mass[idx] - 6, 2), 5))[0]
    #     else:
    #         return np.divide(3, 5)
    #
    # def halt_event(t, x):
    #     return np.min(x)
    # halt_event.terminal = True
    #
    # min_network = LIFE_Network(
    #     file="data/minimal_example.xlsx",
    #     mass=np.array([9, 9, 9]),  # Makes the masses random via constructor
    #     flux=np.array([1, 3, 3, 2]),
    #     ffunc=lambda mass, idx: mass[idx],
    #     min_func=min_min,
    #     source_weights=None,
    #     t_0=0,
    #     t=15,
    #     num_samples=50,
    # )
    #
    # result = min_network.simulate_prototype_pos_def(halt_event)
    # # logger.warning(result.message)
    #
    # basic_graph(result, min_network, [0, 1, 2], ylim=[0, 10])
    #
