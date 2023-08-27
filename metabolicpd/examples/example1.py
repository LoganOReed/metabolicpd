import numpy as np

from metabolicpd.life import network

if __name__ == "__main__":
    s = network.Metabolic_Graph(
        file="data/simple_pd_network.xlsx",
        mass=None,  # Makes the masses random via constructor
        flux=np.random.default_rng().uniform(0.1, 0.8, 28),
        min_func=lambda mass, idxs: np.min(mass[idxs]),
        source_weights=None,
        t_0=0,
        t=15,
        num_samples=50,
    )

    # The three ways of directly controlling metabolites
    # 1. fix trajectory using pos or derivative
    # 2. fix initial value and force it to be constant
    # 3. set initial value (can also be done in Metabolic_Graph.mass)
    #    doesn't control trajectory
    s.fixMetabolite("gba_0", 2.5, -np.sin(s.t_eval), isDerivative=True)
    s.fixMetabolite("a_syn_0", 1.5)
    s.setInitialValue("clearance_0", 0.0)

    result = s.simulate(max_step=0.5)

    # TODO: Write member function for this
    # Create and print results to file
    # respr = pd.DataFrame(result.y.T, columns=network.mtb["name"])
    # respr.insert(0, "time", result.t)
    # respr.to_csv("data/results.csv")

    # TODO: Find a better home for this
    network.basic_plot(result, s, [0, 1, 3, 6, 17, 22, 23, 25])
