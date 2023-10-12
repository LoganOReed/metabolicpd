# -*- coding: utf-8 -*-

# example2: most basic example, uses default for almost everything
import numpy as np

from metabolicpd.life import network

if __name__ == "__main__":
    # Another Example Case for a network from another paper
    m = np.random.rand(12) * 3
    f = np.random.default_rng().uniform(0.1, 0.8, 20)
    cctb_network = network.Metabolic_Graph(
        file="data/central_carbon_tb.csv",
        mass=m,  # Makes the masses random via constructor
        flux=f,
        source_weights=None,
        t_0=0,
        t=80,
        num_samples=750,
    )
    result = cctb_network.simulate()
    print(cctb_network.final_masses)
    network.basic_plot(result, cctb_network, [0, 1, 2, 3, 4, 5, 6, 7, 8], ylim=[0, 3])
