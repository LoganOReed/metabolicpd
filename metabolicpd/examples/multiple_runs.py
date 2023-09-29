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
        t=12,
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
    metabolites_to_plot = [0, 1, 3, 5, 6, 17, 18, 19, 23, 35]
    initial_meta_levels_dict = {}
    final_meta_levels_dict = {}
    for i in metabolites_to_plot:
        meta_name = s.mtb[i][0]
        initial_meta_levels_dict[meta_name] = []
        final_meta_levels_dict[meta_name] = []

    num_runs = 10
    for i in range(0, num_runs):
        # reset the intial values:
        random_vals = []
        for mtb in s.mtb:
            meta_name = mtb[0]
            rand_val = random.uniform(0.1, 2)
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

    # plot a single metabolite initial vs final:
    plt.scatter(initial_meta_levels_dict['a_syn_1'], final_meta_levels_dict['a_syn_1'], label='a_syn_1')
    plt.plot()
    plt.xlabel('initial value')
    plt.ylabel('final value')
    plt.title('Initial vs Final for Multiple Simulations')
    plt.legend()



    #
    # # adjust clearance_0 to be a derivative - index 6
    # clearance_deriv = np.diff(result['y'][6], prepend=[0])
    #
    # metabolites_to_plot = [0, 1, 3, 5, 6, 17, 18, 19, 23, 35]
    # for i in metabolites_to_plot:
    #     if i == 6:
    #         plt.plot(result['t'], clearance_deriv, label=s.mtb[i][0])
    #     else:
    #         plt.plot(result['t'], result['y'][i], label=s.mtb[i][0])
    #
    # plt.ylim([0, 3])
    # plt.xlabel('time')
    # plt.legend()
    # plt.title('Sample Simulation')
    # plt.ylabel('metabolite level')