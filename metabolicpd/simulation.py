# -*- coding: utf-8 -*-

import logging  # for sending timer logs to a file
import re

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
# TODO: Make switching logging styles easier
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


class LIFE_Network:
    """Implements the pipeline outlined in the associated paper.

    Attributes:
        network (DataFrame): Stores the graph/network for the system.
        df (DataFrame): Stores an indexed list of metabolites and their properties
        mass (ndarray): Current masses for metabolites, indices matching `df`'s indices
        flux (ndarray): Current fluxes for the network, indices matching `df`'s indices

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
        self.index_to_name = np.array(new_unique_metabolites)
        self.name_to_index = dict(zip(metabolites, np.arange(len(metabolites))))

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
        self.metabolite_type = np.array(metabolite_types)

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
            for i in range(len(self.df.index)):
                temp_list.append(1)
            self.source_weights = np.array(temp_list)
        else:
            self.source_weights = source_weights

        # Create member dict for potential fixed metabolites
        self.fixed_trajectories = {}

        self.substrates = []
        self.substrates_sources = []
        self.products = []
        self.products_sinks = []
        self.uber_modulators = []
        self.uber_modulator_signs = []
        # Generate lookup tables for s matrix functions
        # TODO: Check with Chris that generating this once will not break edge cases
        # TODO: Check if columns will always be in this order
        for row in self.network[["tail", "head", "uber"]].itertuples():
            sub_str = self.df[self.df["name"].isin(row.tail.split(", "))].index.tolist()
            self.substrates.append(sub_str)
            if len(sub_str) != 1:
                self.substrates_sources.append()
            elif self.df.loc[self.df.index[sub_str[0]], "type"] == "source":
                self.substrates_sources.append(True)
            else:
                self.substrates_sources.append(False)

            prod_snk = self.df[
                self.df["name"].isin(row.head.split(", "))
            ].index.tolist()
            self.products.append(prod_snk)
            if len(prod_snk) != 1:
                self.products_sinks.append(False)
            elif self.df.loc[self.df.index[prod_snk[0]], "type"] == "sink":
                self.products_sinks.append(True)
            else:
                self.products_sinks.append(False)

            self.uber_modulators.append(
                self.df[
                    self.df["name"].isin([s[:-2] for s in row.uber.split(", ")])
                ].index.tolist()
            )
            u_m = row.uber.split(", ")
            u_m_s = []
            for uber in u_m:
                # there should probably be some checking that happens so we don't duplicate the code in the if statements
                if uber[-1] == "+":  # check the last term in the entry, enhancer
                    u_m_s.append(True)
                elif uber[-1] == "-":  # check the last term in the entry, enhancer
                    u_m_s.append(False)
            self.uber_modulator_signs.append(u_m_s)

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
        s_matrix = []
        # TODO: Check if columns will always be in this order
        for row in self.network[["tail", "head", "uber"]].itertuples():
            # iterate through each of the edges
            # there could be more than one substrate
            # or product! (for instance, in a hyperedge)

            row_index = row.Index
            col = np.zeros(self.df.shape[0])

            # build the uber term for each expression, default if no uber edges is 1
            uber_term = 1.0
            for i in range(len(self.uber_modulator_signs[row_index])):
                # there should probably be some checking that happens so we don't duplicate the code in the if statements
                if self.uber_modulator_signs[row_index][
                    i
                ]:  # check the last term in the entry, enhancer
                    uber_term = uber_term * self.ffunc(
                        mass, self.uber_modulators[row_index][i]
                    )
                elif not self.uber_modulator_signs[row_index][
                    i
                ]:  # check the last term in the entry, enhancer
                    # note uber_term will always be less than one
                    uber_term = uber_term / self.ffunc(
                        mass, self.uber_modulators[row_index][i]
                    )

            # Note that I'm vectorizing as much as possible as the actual dataframe will be massive.
            # Also note that I pulled as much as possible out of the function as its called every time simulate() iterates
            # idxs = np.array(self.substrates[row_index])
            # idxp = np.array(self.products[row_index])
            idxs = self.substrates[row_index]
            idxp = self.products[row_index]
            # Case: Hyperedge
            if len(idxs) > 1 or len(idxp) > 1:
                # This chunk of code finds min, and sets col values for both substrates and products appropriately
                min_sub = self.min_func(mass, idxs)
                col[idxs] = -1 * min_sub * uber_term
                col[idxp] = min_sub * uber_term
            # Case: Not Hyperedge
            else:
                if self.substrates_sources[row_index]:
                    # NOTE: Implementation assumes source edges are simple diredges,
                    #       since networks can always be converted to this form
                    col[idxp] = self.source_weights[idxp]
                elif self.products_sinks[row_index]:
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

    # I'm so sorry for writing this horrible code
    # NOTE: I Attempted to write an equivalent function when given a position function
    # but it ran for more than a minute and a half without any results.
    @conditional_timer("fixMetabolite")
    def fixMetabolite(self, mtb, val, trajectory=None, isDerivative=False):
        """Sets fixed flag to true and mass value to init val, and gives a derivative function for the trajectory."""
        self.df.loc[self.df["name"].isin([mtb]), ["fixed"]] = True
        idx = self.df[self.df["name"].isin([mtb])].index.to_numpy()
        self.mass[idx[0]] = val
        if trajectory is None:
            trajectory = self.t_eval.copy()
            if isDerivative:
                trajectory = trajectory.fill(0.0)
            else:
                trajectory = trajectory.fill(val)
        else:
            # convert function to ndarray if needed
            if type(trajectory) is not np.ndarray:
                trajectory = trajectory(self.t_eval)

        if not isDerivative:
            trajectory = np.diff(trajectory) / self.step_size

        self.fixed_trajectories[idx[0]] = lambda t: trajectory[
            int(min(np.floor(t / self.step_size), self.num_samples - 2))
        ]

    def setInitialValue(self, mtb, val):
        """Sets mass value to vals."""
        # All of the lines that look like below are temporary hopefully
        # (From Switching to singleton mtb instead of lists)
        idx = self.df[self.df["name"].isin([mtb])].index.to_numpy()
        self.mass[idx] = val

    # TODO: Clean up this function
    def get_names(self):
        return self.df["name"]


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
    # TODO: make a list of a couple interesting argument values for test cases
    # TODO: design dataframe for outputting simulation results
    # TODO: Write function that saves resultant data to file
    # TODO: Rewrite create_S_matrix to vectorize the row operations

    network = LIFE_Network(
        file="data/simple_pd_network.xlsx",
        mass=None,  # Makes the masses random via constructor
        flux=np.random.default_rng().uniform(0.1, 0.8, 28),
        ffunc=lambda mass, idx: np.exp(mass[idx] / (mass[idx] + 1)),
        min_func=lambda mass, idxs: np.min(mass[idxs]),
        source_weights=None,
        t_0=0,
        t=15,
    )

    # setting trajectory using ndarray
    network.fixMetabolite("gba_0", 2.5, -np.sin(network.t_eval), isDerivative=True)
    # setting trajectory using ufunc
    network.fixMetabolite("a_syn_0", 1.5, np.cos)
    # Set initial value without fixing metabolite
    network.setInitialValue("clearance_0", 0.0)

    result = network.simulate()
    logger.warning(result.message)

    # Create and print results to file
    respr = pd.DataFrame(result.y.T, columns=network.get_names())
    respr.insert(0, "time", result.t)
    respr.to_csv("data/results.csv")

    # takes in xlsx, (optional) initial mass/flux, (optional) simulation time
    # gives result of simulation, interfaces for plotting/saving/analysing

    print(network.df.to_markdown())
    basic_graph(result)
