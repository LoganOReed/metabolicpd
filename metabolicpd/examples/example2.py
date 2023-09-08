# -*- coding: utf-8 -*-

# example2: most basic example, uses default for almost everything
import numpy as np

from metabolicpd.life import network

if __name__ == "__main__":
    # Another Example Case for a network from another paper
    # m = np.random.rand(12) * 3
    # f = np.random.default_rng().uniform(0.1, 0.8, 20)
    m = np.array(
        [
            0.10194245,
            1.12079691,
            2.36864793,
            0.08116026,
            2.04990994,
            0.45485493,
            1.68571228,
            1.46591457,
            2.28409727,
            2.54365606,
            2.85419202,
            0.47321457,
        ]
    )
    f = np.array(
        [
            0.58770277,
            0.65382647,
            0.34918603,
            0.28588688,
            0.24594188,
            0.5563654,
            0.40659335,
            0.37662764,
            0.53239271,
            0.18575152,
            0.30141636,
            0.52224964,
            0.27252841,
            0.70293088,
            0.57637894,
            0.25985019,
            0.45472977,
            0.53655764,
            0.57407442,
            0.23947391,
        ]
    )

    cctb_network = network.Metabolic_Graph(
        file="data/central_carbon_tb.xlsx",
        mass=m,  # Makes the masses random via constructor
        flux=f,
        source_weights=None,
        t_0=0,
        t=80,
        num_samples=750,
    )
    result = cctb_network.simulate()
    print(result["y"].T[-1])
    print(result["y"].T[-1].shape)
    print(cctb_network.mtb["name"].shape)
    network.basic_plot(result, cctb_network, [0, 1, 2, 3, 4, 5, 6, 7, 8], ylim=[0, 3])
