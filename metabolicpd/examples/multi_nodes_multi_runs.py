# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from metabolicpd.life import network
import random

if __name__ == "__main__":
    s = network.Metabolic_Graph(
        file="data/simple_pd_network.xlsx",
        mass=None,  # Makes the masses random via constructor
        flux=None,
        source_weights=None,
        t_0=0,
        t=25,
        num_samples=100,
    )

    # The three ways of directly controlling metabolites
    # 1. fix trajectory using pos or derivative
    # 2. fix initial value and force it to be constant
    # 3. set initial value (can also be done in Metabolic_Graph.mass)
    #    doesn't control trajectory
    # s.setInitialValue("clearance_0", 0.0)
    # s.fixMetabolite('gba_0', 2.5)
    # result = s.simulate()
    # network.basic_plot(result, s, [0, 1, 3, 5, 6, 17, 18, 19, 23, 35])
    metabolites_to_plot = [0, 1, 3, 5, 6, 17, 18, 19, 22, 23, 35]
    metabolites_to_plot = list(range(0, len(s.mtb)))
    initial_meta_levels_dict = {}
    final_meta_levels_dict = {}
    for i in metabolites_to_plot:
        meta_name = s.mtb[i][0]
        initial_meta_levels_dict[meta_name] = []
        final_meta_levels_dict[meta_name] = []

    num_runs = 50
    for i in range(0, num_runs):
        # reset the initial values:
        random_vals = []
        for mtb in s.mtb:
            meta_name = mtb[0]
            rand_val = random.random()
            initial_meta_levels_dict.setdefault(meta_name, [])
            initial_meta_levels_dict[meta_name].append(rand_val)
            random_vals.append(rand_val)
            s.setInitialValue(meta_name, rand_val)

        # run the simulation
        result = s.simulate()

        # mark the final endpoints of the metabolites we care about
        # begin by generating a dictionary to hold metabolite values
        # append the final value to the dictionary
        for ind in metabolites_to_plot:
            meta_name = s.mtb[ind][0]
            final_meta_levels_dict[meta_name].append(result['y'][ind][-1])

    # use subplot to display multiple metabolites at a time:
    mtb_to_display = [0, 1, 3, 4, 5, 19, 21, 22, 23, 24]
    figure, axis = plt.subplots(len(mtb_to_display), 1)

    for i, meta_ind in enumerate(mtb_to_display):
        meta_name = s.mtb[meta_ind][0]
        axis[i].scatter(initial_meta_levels_dict[meta_name], final_meta_levels_dict[meta_name], label=meta_name)
        axis[i].legend()
        axis[i].set_ylim([0, 7])
    figure.suptitle('Multiple Nodes Multiple Simulations')
    plt.show()

