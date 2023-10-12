# -*- coding: utf-8 -*-

import platform
import re
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.integrate as scp
import seaborn as sns

from metabolicpd.life import util


# TODO: Make flowchart for simulation function calls
# TODO: Chance virtual nodes so we only have one source and one sink
# TODO: Implement algorithm from phd paper to find suitable flux and source weights.
class Metabolic_Graph:
    """Implements the pipeline outlined in the associated paper.

    Attributes:
        network:
          A pandas dataframe containing the structure of the metabolic network.
        mass:
          A list of positive floats of length equal to the number of metabolites in the network.
        flux:
          A list of floats of length equal to the number of edges in the network
        ffunc:
          A list of floats which take the y value of the desired function for the t values in t_eval.
          The sampled function must be differentiable and strictly increasing.
        min_func:
          A function which decides the edge weight between actual (rather than virtual) metabolites.
        source_weights:
          A list of floats which correspond to the weights of each source nodes (equivalently their edges).
        t_0:
          Initial time value used when running simulation.
        t:
          Final time value used when running simulation.
        t_eval:
          A list of floating points in (t_0, t) used as sample points when fixing metabolite trajectories.
    """

    # TODO: Write function that saves resultant data to file
    # TODO: Use pg. 44 in LIFE Approach by Nathaniel Merrill to find basis
    def __init__(
        self,
        file: str = "",
        mass: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
        flux: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
        ffunc: Optional[
            Callable[[np.ndarray[Any, np.dtype[np.float64]], int], float]
        ] = None,
        min_func: Optional[
            Callable[[np.ndarray[Any, np.dtype[np.float64]], int], float]
        ] = None,
        source_weights: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
        t_0: float | int = 0,
        t: float | int = 10,
        num_samples: int = 50,
    ) -> None:
        """Initializes member variables and computes variables needed for the simulation.

        During the initialization, we compute multiple values and store them in private member variables.
        This allows us to use these values as a lookup table during the simulation computation, saving immense time compared to computing the same values every loop.
        The exception to this occurs when the simulation is short, but init will still occur a relatively small time cost.

        Args:
         file:
          A string to the xlsx file containing the structure of the metabolic network.
         mass:
           A list of positive floats of length equal to the number of metabolites in the network.
         flux:
           A list of floats of length equal to the number of edges in the network
         ffunc:
           Either a differentiable, strictly increasing function or a list of floats which take the y value of the desired function for the t values in t_eval
         min_func:
           A function which decides the edge weight between actual (rather than virtual) metabolites.
         source_weights:
           A list of floats which correspond to the weights of each source nodes (equivalently their edges).
         t_0:
           Initial time value used when running simulation.
         t:
           Final time value used when running simulation.
         num_samples:
           An integer which determines the max step size for the simulation and is used to compute t_eval.
        """
        # setup simulation evaulation points
        self.t_0 = t_0
        self.t = t
        self.t_eval, _ = np.linspace(self.t_0, self.t, num_samples, retstep=True)
        self.final_masses = None

        # read graph/network from clean file
        self.network: pd.DataFrame = pd.read_csv(file)  # type = ignore
        unique_entries = np.unique(self.network[["tail", "head"]].values)
        print(unique_entries)

        # Gather list of metabolites in network
        metabolites = []
        for entry in unique_entries:
            elements = entry.split(" ")
            print(elements)
            for ele in elements:
                metabolites.append(ele)
        new_unique_metabolites = list(dict.fromkeys(metabolites))
        print(new_unique_metabolites)
        num_mtb = len(new_unique_metabolites)
        # TODO: remove num_edges since its never used
        self._num_edges = self.network.shape[0]

        if mass is None:
            np.random.default_rng()
            self.mass = np.random.rand(num_mtb)
        else:
            self.mass = mass

        if flux is None:
            # self.flux = np.ones(self.network.shape[0])
            self.flux = np.random.rand(self.network.shape[0])
        else:
            self.flux = flux

        if ffunc is None:
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
            # starts with e, ends with #...
            sink_res = re.search("^e[0-9]+$", ele)
            # starts with s, end with #...
            source_res = re.search("^s[0-9]+$", ele)
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

        self.mtb: npt.ArrayLike = np.zeros(
            num_mtb,
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
            for _ in range(num_mtb):
                # temp_list.append(1)
                temp_list.append(1)
            self.source_weights = np.array(temp_list)
        else:
            self.source_weights = source_weights

        # Create member dict for potential fixed metabolites
        self.fixed_trajectories = {}

        self.hyper_edges = np.zeros((self.network.shape[0], num_mtb))
        self.source_edges = np.zeros((self.network.shape[0], num_mtb))
        uber_mods = np.zeros((self.network.shape[0], num_mtb))

        self.substrates = []
        # Generate lookup tables for s matrix functions
        # TODO: Add internal datastructure definition to README
        for row in self.network[["tail", "head", "uberPos", "uberNeg"]].itertuples():
            row_idx = row.Index
            sub_str = self.mtb[np.isin(self.mtb["name"], row.tail.split(" "))]["index"]
            prod_snk = self.mtb[np.isin(self.mtb["name"], row.head.split(" "))]["index"]

            self.substrates.append(sub_str)

            if self.mtb[sub_str[0]]["type"] == "source":
                self.source_edges[row_idx, prod_snk] = self.source_weights[sub_str[0]]
            elif self.mtb[prod_snk[0]]["type"] == "source":
                self.hyper_edges[row_idx, prod_snk] = 0
                self.hyper_edges[row_idx, sub_str] = -1
            else:
                self.hyper_edges[row_idx, sub_str] = -1
                self.hyper_edges[row_idx, prod_snk] = 1

            # Next two loops read in inhibiters(uNeg) and enhancers(uPos)
            u_p = row.uberPos.split(" ")
            u_n = row.uberNeg.split(" ")
            for uber in u_p:
                uber_mods[row.Index, self.mtb[self.mtb["name"] == uber]["index"]] = 1

            for uber in u_n:
                uber_mods[row.Index, self.mtb[self.mtb["name"] == uber]["index"]] = -1

        # TODO: Condense uber_mods and uber_enhancers/inhibiters
        self.uber_enhancers = np.nonzero(uber_mods == 1)
        self.uber_inhibiters = np.nonzero(uber_mods == -1)

    # NOTE: I've set this up so ideally it will only be called by the "simulation" function once written
    # TODO: Write Documentation for new member variables, write example use case
    # TODO: Create tests for S_matrix computation
    def create_S_matrix(self, mass: np.ndarray[Any, np.dtype[np.float64]]):
        """Create the 'S' matrix, representing the dynamics for the network x' = S(x) * f.

        The form of the 'S' matrix comes follows LIFE dynamics, with edges in the network corresponding to columns in the
        matrix and metabolites in the network corresponding to the rows of the matrix.

        Args:
            mass:
              Numpy array of masses to construct the S matrix based off of.

        Returns:
            A numpy array representing the S matrix for the current metabolite masses.
        """
        # Compute Uber Diagonal
        u_t = np.zeros(self._num_edges)
        u_t.fill(1.0)
        for i in range(len(self.uber_enhancers[0])):
            u_t[self.uber_enhancers[0][i]] = u_t[
                self.uber_enhancers[0][i]
            ] * self.ffunc(mass, self.uber_enhancers[1][i])

        for i in range(len(self.uber_inhibiters[0])):
            u_t[self.uber_inhibiters[0][i]] = u_t[
                self.uber_inhibiters[0][i]
            ] / self.ffunc(mass, self.uber_inhibiters[1][i])
        uber_diag = np.diagflat(u_t)

        # Compute diagonal for the mass
        mass_diag = np.zeros((self._num_edges, self._num_edges))
        for row in self.network[["tail", "head", "uberPos", "uberNeg"]].itertuples():
            row_index = row.Index
            idxs = self.substrates[row_index]
            min_sub = self.min_func(mass, idxs)
            mass_diag[row_index, row_index] = min_sub

        return ((uber_diag @ mass_diag) @ self.hyper_edges + self.source_edges).T

    # TODO: Docstring
    def __s_function(self, t, x):
        fixed_idx = self.mtb[self.mtb["fixed"]]["index"]  # type: ignore
        der = self.create_S_matrix(x) @ self.flux
        # replaces computed derivative with one which we control
        for i in fixed_idx:
            der[i] = self.fixed_trajectories[i](t)
        return der

    # Allows access to step function which would make setting specific values easier
    def simulate(self):
        """Runs the simulation."""
        ts = []
        xs = []
        sol = scp.RK45(
            self.__s_function,
            self.t_0,
            self.mass,
            self.t,
            max_step=(self.t_eval[1] - self.t_eval[0]),
        )
        # options are 'running' 'finished' or 'failed'
        while sol.status == "running":
            sol.step()
            ts.append(sol.t)
            xs.append(sol.y)

        tt = np.array(ts)
        yy = np.stack(xs)
        self.final_masses = yy[-1]
        res = {"t": tt, "y": yy.T}
        return res

    def fixMetabolite(
        self,
        m_name: str,
        val: float,
        trajectory: Optional[
            np.ndarray[Any, np.dtype[np.float64]]
            | Callable[
                [np.ndarray[Any, np.dtype[np.float64]]],
                np.ndarray[Any, np.dtype[np.float64]],
            ]
        ] = None,
        isDerivative=False,
    ):
        """Sets fixed flag to true and mass value to init val, and gives a derivative function for the trajectory."""
        # Need to be careful to have a scalar index instead of an array to view data instead of copy
        idx = self.mtb[self.mtb["name"] == m_name]["index"][0]  # type: ignore
        f_mtb = self.mtb[idx]  # type: ignore
        f_mtb["fixed"] = True  # type: ignore

        self.mass[idx] = val  # type: ignore
        if trajectory is None:
            trajectory = np.zeros(self.t_eval.shape[0])
            isDerivative = True
        else:
            # convert function to ndarray if needed
            if type(trajectory) is not np.ndarray:
                trajectory = trajectory(self.t_eval)  # type: ignore

        if not isDerivative:
            trajectory = np.diff(trajectory) / (
                self.t_eval[1] - self.t_eval[0]
            )  # type: ignore

        self.fixed_trajectories[idx] = lambda t: trajectory[  # type: ignore
            int(
                min(
                    np.floor(t / (self.t_eval[1] - self.t_eval[0])),
                    self.t_eval.shape[0] - 2,
                )
            )
        ]

    def setInitialValue(self, mtb: str, val: float):
        """Sets mass value to vals."""
        # All of the lines that look like below are temporary hopefully
        # (From Switching to singleton mtb instead of lists)
        idx = self.mtb[self.mtb["name"] == mtb]["index"][0]  # type: ignore
        self.mass[idx] = val  # type: ignore

    def getFinalMasses(self):
        pass


# TODO: Maybe move into class or make plot file
def basic_plot(
    result: dict[str, list[float]],
    network: Metabolic_Graph,
    mtb_to_plot: list[int] = [],
    ylim: list[float] = [0, 3],
) -> None:
    """Creates a plot showing the metabolites `mtb_to_plot` using Metabolic_Graph data."""
    # Setup different plt backend for kitty term
    if platform.system() == "Linux":
        plt.switch_backend("module://matplotlib-backend-kitty")
    sns.set_theme()
    sns.set_style("dark")
    sns.color_palette("pastel")
    sns.set_context("talk")

    if mtb_to_plot == []:
        mtb_to_plot = [i for i in range(network.mtb.size)]

    metabolites_to_plot = mtb_to_plot
    mtb_names = network.mtb[metabolites_to_plot]["name"]  # type: ignore
    label_idx = 0
    for i in metabolites_to_plot:
        plt.plot(result["t"], result["y"][i], label=mtb_names[label_idx])
        label_idx = label_idx + 1
    plt.ylim(ylim)
    plt.xlabel("$t$")  # the horizontal axis represents the time
    plt.legend()  # show how the colors correspond to the components of X
    sns.despine(offset=10, trim=True)
    # plt.savefig("latest.png")
    plt.show()


if __name__ == "__main__":
    print("network.py is not intended to be main, use an example or call in a script.")
