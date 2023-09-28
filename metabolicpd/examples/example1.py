# -*- coding: utf-8 -*-

import numpy as np

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
    s.setInitialValue("clearance_0", 0.0)

    result = s.simulate()
    network.basic_plot(result, s, [0, 1, 3, 6, 17, 22, 23, 25])
