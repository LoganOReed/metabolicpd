import numpy as np

from metabolicpd.life import network

if __name__ == "__main__":
    # Another Example Case for a network from another paper
    cctb_network = network.Metabolic_Graph(
        file="data/central_carbon_tb.xlsx",
        mass=None,  # Makes the masses random via constructor
        flux=np.random.default_rng().uniform(0.1, 0.8, 20),
        ffunc=lambda mass, idx: np.exp(mass[idx] / (mass[idx] + 1)),
        min_func=lambda mass, idxs: np.min(mass[idxs]),
        source_weights=None,
        t_0=0,
        t=15,
        num_samples=50,
    )
    result = cctb_network.simulate()

    network.basic_graph(result, cctb_network, [0, 1, 2, 3, 4, 5, 6, 7, 8])
