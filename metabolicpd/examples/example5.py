# -*- coding: utf-8 -*-

# import numpy as np

from metabolicpd.io.kegg_pathway import network_from_KGML
from metabolicpd.life import network
from metabolicpd.life.virtual_nodes import add_nodes

if __name__ == "__main__":
    file = network_from_KGML("mtu01200")
    file = add_nodes(file)

    s = network.Metabolic_Graph(
        file=file,
        mass=None,  # Makes the masses random via constructor
        flux=None,
        source_weights=None,
        t_0=0,
        t=250,
        num_samples=1000,
    )

    # The three ways of directly controlling metabolites
    # 1. fix trajectory using pos or derivative
    # 2. fix initial value and force it to be constant
    # 3. set initial value (can also be done in Metabolic_Graph.mass)
    #    doesn't control trajectory
    print(s.mtb)

    result = s.simulate()
    # NOTE: it looks like mis_a_syn_0 diverges but it actually converges to ~8
    network.basic_plot(
        result,
        s,
        [i for i in range(54)],
        ylim=[0, 40],
    )
    counter = 0
    for i in result["y"]:
        if i[-1] > 30.0 and counter < 55:
            print(f"{str(counter)}: {i[-1]}")
        counter = counter + 1
    print(
        "Note: The reason this doesn't always approach a steady state solution is that we could use fluxes outside of that space"
    )
