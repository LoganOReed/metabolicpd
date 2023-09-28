# -*- coding: utf-8 -*-

import numpy as np
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
from metabolicpd.life import network



if __name__ == "__main__":
    s = network.Metabolic_Graph(
        file="data/simple_pd_network.xlsx",
        mass=None,  # Makes the masses random via constructor
        flux=None,
        source_weights=None,
        t_0=0,
        t=15,
        num_samples=100,
    )

    # The three ways of directly controlling metabolites
    # 1. fix trajectory using pos or derivative
    # 2. fix initial value and force it to be constant
    # 3. set initial value (can also be done in Metabolic_Graph.mass)
    #    doesn't control trajectory

    # set the default values for the metabolites
    num_vars = 0
    meta_names = []
    indicies_of_actual = []
    for mtb in s.mtb:
        meta_name = mtb[0]
        meta_class = mtb[1]
        ind = mtb[3]
        if meta_class == 'actual':
            s.setInitialValue(meta_name, 1.0)
            meta_names.append(meta_name)
            num_vars += 1
            indicies_of_actual.append(ind)

    for mtb in s.mtb:
        s.setInitialValue(mtb[0], 1.0)

    # lower bound and upper bounds
    lower_bounds = np.array([0] * num_vars)
    upper_bounds = np.array([10] * num_vars)

    # defining the problem
    problem = {
        'num_vars': num_vars,
        'names': meta_names,
        'bounds': np.column_stack((lower_bounds, upper_bounds))
    }

    # Generate the samples
    vals = saltelli.sample(problem, 2**3)

    # Run the model
    # numerically solve the ODE
    # output is metabolites at the end of the time step
    Y = np.zeros([len(vals), num_vars])  # initialize the output-holidng matrix
    for i in range(len(vals)):

        # update using the values from the saltelli sampling
        for meta, val in zip(meta_names, vals[i]):
            s.setInitialValue(meta, val)

        sim_result = s.simulate()
        y_temp = []
        for ind in indicies_of_actual:
            to_append = sim_result['y'][ind][-1]
            y_temp.append(to_append)
        Y[i][:] = y_temp
    # Sobol Analysis
    sobol_out = []
    for ind, meta in enumerate(meta_names):
        print(f'Sobol output for {meta}')
        sobol_analysis = sobol.analyze(problem, Y[:, ind], print_to_console=True)
        sobol_out.append(sobol_analysis)

