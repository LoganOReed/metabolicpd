import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from metabolicpd.life import network


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


if __name__ == "__main__":
    # Setup different plt backend for kitty term
    if platform.system() == "Linux":
        plt.switch_backend("module://matplotlib-backend-kitty")

    # TODO: make a list of a couple interesting argument values for test cases
    # TODO: design dataframe for outputting simulation results
    # TODO: Write function that saves resultant data to file
    # TODO: Rewrite create_S_matrix to vectorize the row operations

    pd.set_option("display.precision", 2)

    s = network.Metabolic_Graph(
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
    s.fixMetabolite("gba_0", 2.5)
    # setting trajectory using ufunc
    # network.fixMetabolite("a_syn_0", 1.5)
    # Set initial value without fixing metabolite
    s.setInitialValue("clearance_0", 0.0)

    # result = network.simulate(max_step=0.5)
    result = s.simulate(max_step=0.5)

    # Create and print results to file
    # respr = pd.DataFrame(result.y.T, columns=network.mtb["name"])
    # respr.insert(0, "time", result.t)
    # respr.to_csv("data/results.csv")

    # takes in xlsx, (optional) initial mass/flux, (optional) simulation time
    # gives result of simulation, interfaces for plotting/saving/analysing

    # print(network.mtb)
    basic_graph(result, s, [0, 1, 3, 6, 17, 22, 23, 25])
