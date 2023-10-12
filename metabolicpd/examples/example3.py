# Example3: Very small metabolic graph with no virtual nodes
# has fixed mass, flux and custom min function

import numpy as np

from metabolicpd.life import network

if __name__ == "__main__":

    def min_min(mass, idx):
        if np.less_equal(mass[idx], 6):
            return np.divide(3, np.add(np.power(mass[idx] - 6, 2), 5))[0]
        else:
            return np.divide(3, 5)

    min_network = network.Metabolic_Graph(
        file="data/minimal_example.csv",
        mass=np.array([3, 2, 5]),
        flux=np.array([1, 3, 3, 2]),
        ffunc=lambda mass, idx: mass[idx],  # type: ignore
        min_func=min_min,
        source_weights=None,
        t_0=0,
        t=15,
        num_samples=50,
    )
    result = min_network.simulate()
    network.basic_plot(result, min_network, [0, 1, 2], ylim=[0, 10])
