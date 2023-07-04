import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sympy import Matrix, Min, exp, symbols


def extract_metabolites(network_df):
    """Extracts the unique names within the network dataframe.

    This function simply return all the unique names from an edge list
    held in a pandas dataframe. If an entry in the df has multiple string (separated
    by a comma and space: ", "), then it splits the entry.

    Args:
        network_df : pandas dataframe that holds and edge list

    Returns:
        the unique strings contained within network_df
    """
    unique_entries = np.unique(network_df[["tail", "head"]].values)

    metabolites = []
    for entry in unique_entries:
        elements = entry.split(", ")
        for ele in elements:
            metabolites.append(ele)
    new_unique_metabolites = list(dict.fromkeys(metabolites))
    return new_unique_metabolites


def create_meta_lookup(unique_metabolite_list):
    """Creates a lookup dictionary that ties unique metabolite names to an index.

    Args:
        unique_metabolite_list : python list holding unique metabolite names

    Returns:
        A dictionary where keys are metabolite names and values are indices
    """
    metabolite_lookup_dict = {}
    index_counter = 0
    for entry in unique_metabolite_list:
        metabolite_lookup_dict[entry] = index_counter
        index_counter += 1

    return metabolite_lookup_dict


def find_source_sink_index(meta_lookup_dict):
    """Searches the metabolite lookup dictionary for source and sink terms.

    Source terms are indicated by 's#' and sink terms are indicated by
    'e#'. 'e' comes from the language: "excretion in a network"

    Args:
        meta_lookup_dict : dictionary holding unique metabolite names and indices

    Returns:
        source_sink_dict : A dictionary where keys are sources and sinks and values are the corresponding index
        actual_meta_list : A pyhton list that holds actual metabolites within the network
    """
    source_sink_dict = {"source": [], "sink": []}
    actual_meta_list = []
    for key in meta_lookup_dict.keys():
        # Using regular expression strings to identify source or sink terms in metabolite dictionary keys
        sink_res = re.search("^[e]+[0-9]$", key)  # starts with e, ends with #...
        source_res = re.search("^[s]+[0-9]$", key)  # starts with s, end with #...
        if (
            sink_res is not None
        ):  # check if we found something in the entry that matches the format of 'e###'
            source_sink_dict["sink"].append(meta_lookup_dict[key])
        elif (
            source_res is not None
        ):  # chick if we found something that matches format of source term
            source_sink_dict["source"].append(meta_lookup_dict[key])
        else:  # if we didn't find a source or sink term, then it must be an actual metabolite!
            actual_meta_list.append(key)

    return source_sink_dict, actual_meta_list


def create_symbolic_meta_array(actual_meta_lookup):
    """Creates a list that holds Sympy symbols with variable names from actual_meta_lookup.

    Args:
        actual_meta_lookup : a python dictionary with keys being actual metabolites within network and values being
         indices of the corresponding key

    Returns:
        symbolic_array : a list of Sympy symbols that come from the keys in actual_meta_lookup
    """
    symbolic_array = []
    for key in actual_meta_lookup.keys():
        symbolic_array.append(symbols(key))

    return symbolic_array


def create_value_array(sym_array):
    """Creates a numeric array based on number of actual metabolites.

    Args:
        sym_array: a list of Sympy symbols that correspond to metabolites in network

    Returns:
        Numpy array of random values between 0, 1 that is in parallel with sym_array
    """
    np.random.seed(seed=12)
    return np.random.rand(len(sym_array))


def create_S_matrix(net_df, actual_meta_lookup, sym_array):
    """Create the 'S' matrix that defines the dynamics of a metabolic network.

    The form of the 'S' matrix comes follows LIFE dynamics, with edges in the network corresponding to columns in the
    matrix and metabolites in the network corresponding to the rows of the matrix.

    The indices within actual_meta_lookup correspond to the indices of sym_array.

    Args:
        net_df: pandas dataframe of an edge list encoding the metabolic network
        actual_meta_lookup: python dictionary with keys as metabolites in network and values of indices
        sym_array: python list of Sympy symbols corresponding to the metabolites in the network

    Returns:
        A list of lists with elements being sympy symbols that encodes the dynamics of the network according to the
            LIFE approach
    """
    # number of non-source/sink metabolites
    num_metabolites = len(sym_array)  # the number of rows in the matrix
    # num_edges = len(
    #     net_df["tail"].values
    # )  # the number of cols in the matrix  -- this is unused

    s_matrix = []  # create an empty list to hold the eventual columns

    edge_columns_df = net_df[
        ["tail", "head", "uber"]
    ]  # get a dataframe for just the edges (future-proof for uberedges)
    for row in edge_columns_df.itertuples():  # iterate through each of the edges
        substrates = row.tail.split(", ")  # there could be more than one substrate
        products = row.head.split(", ")  # or product! (for instance, in a hyperedge)
        uber_modulators = row.uber.split(", ")

        num_subs = len(substrates)
        num_prods = len(products)

        # create a dummy column - we will set the values later
        current_col = [0 for i in range(0, num_metabolites)]
        # current_col = np.array([0 for i in range(0, num_metabolites)])

        # build the uber term for each expression, default if no uber edges is 1
        uber_term = 1
        for uber in uber_modulators:
            # there should probably be some checking that happens so we don't duplicate the code in the if statements
            # uber_meta = uber[:-2]  # get all the characters, except the last two (either _+/-)
            # index_of_uber_meta = actual_meta_lookup[uber_meta]
            # uber_symbol = sym_array[index_of_uber_meta]
            if uber[-1] == "+":  # check the last term in the entry, enhancer
                uber_meta = uber[
                    :-2
                ]  # get all the characters, except the last two (either _+/-)
                index_of_uber_meta = actual_meta_lookup[
                    uber_meta
                ]  # get the index of the uber metabolites
                uber_symbol = sym_array[
                    index_of_uber_meta
                ]  # get the symbol of the uber metabolite
                uber_term = uber_term * exp(
                    (uber_symbol / (uber_symbol + 1))
                )  # always greater than one
            elif uber[-1] == "-":  # if the last term designates an inhibitor
                uber_meta = uber[
                    :-2
                ]  # get all the characters, except the last two (either _+/-)
                index_of_uber_meta = actual_meta_lookup[uber_meta]
                uber_symbol = sym_array[index_of_uber_meta]
                uber_term = (
                    uber_term * 1 / exp((uber_symbol / (uber_symbol + 1)))
                )  # always less than one

        if num_subs > 1 or num_prods > 1:  # hyperedge!
            index_of_subs_in_meta = []
            index_of_prods_in_meta = []
            for sub in substrates:  # get the index for all the substrates in the edge
                index_of_subs_in_meta.append(actual_meta_lookup[sub])
            for prod in products:
                index_of_prods_in_meta.append(actual_meta_lookup[prod])

            # now, we create the terms for the hyperedge
            # current_col[index_of_subs_in_meta] = -1*min(sym_array[index_of_subs_in_meta])
            # current_col[index_of_prods_in_meta] = min(sym_array[index_of_subs_in_meta])

            # build the term for the hyperedge min(substrates in the hyperedge)
            terms_in_min = [sym_array[sub_idx] for sub_idx in index_of_subs_in_meta]
            expression = Min(*terms_in_min)
            for sub_index in index_of_subs_in_meta:
                current_col[sub_index] = -1 * expression * uber_term
                # current_col[sub_index] = -1*min([sym_array[idx] for idx in index_of_subs_in_meta])
            for prod_index in index_of_prods_in_meta:
                current_col[prod_index] = expression * uber_term
                # current_col[prod_index] = min([sym_array[idx] for idx in index_of_subs_in_meta])

        else:  # not a hyperedge!
            # again, we check if not a source or sink
            if re.search("^[s]+[0-9]$", substrates[0]) is not None:  # source term
                index_of_prod_in_meta = actual_meta_lookup[products[0]]
                current_col[
                    index_of_prod_in_meta
                ] = 1  # we don't ever expect to have a source with an uber edge
            elif re.search("^[e]+[0-9]", products[0]) is not None:  # sink term
                index_of_sub_in_meta = actual_meta_lookup[substrates[0]]
                current_col[index_of_sub_in_meta] = (
                    -1 * sym_array[index_of_sub_in_meta] * uber_term
                )
            else:  # not a source or sink
                index_of_prod_in_meta = actual_meta_lookup[products[0]]
                index_of_sub_in_meta = actual_meta_lookup[substrates[0]]

                current_col[index_of_sub_in_meta] = (
                    -1 * sym_array[index_of_sub_in_meta] * uber_term
                )
                current_col[index_of_prod_in_meta] = (
                    sym_array[index_of_sub_in_meta] * uber_term
                )

        s_matrix.append(current_col)
    return s_matrix


def euler_integrate(
    sym_S_matrix, sym_metabolites, value_array, symbolic_flux, tspan, tstep
):
    """Computes the integral of the dynamics based on euler's method.

    Args:
        sym_S_matrix: matrix of sympy symbols that corresponds to the S matrix in LIFE
        sym_metabolites: a list of sympy symbols that appears within sym_S_matrix
        value_array: a list of values that are in parallel to sym_metabolites
        symbolic_flux: a list of sympy symbols for the constant flux vector in LIFE
        tspan: list of two values, starting point in integration and ending point
        tstep: the step in t that we take for each step in Euler's Method


    Returns:
        t_array: a list of floats from tspan[0] to tspan[1] with stepsize tstep
        meta_value_array: the calculated trajectory of mass for each metabolite within the network
    """
    t_current = tspan[0]
    t_final = tspan[1]

    t_array = [t_current]
    meta_values_array = [value_array]

    while t_current < t_final:
        zipped_values = zip(sym_metabolites, value_array)
        integration_step = Matrix(value_array) + tstep * sym_S_matrix.subs(
            zipped_values
        ) * symbolic_flux.subs(zipped_values)
        value_array = integration_step

        meta_values_array.append(*value_array.T.tolist())

        t_current += tstep
        t_array.append(t_current)

    return t_array, meta_values_array


def modified_euler_integrate(
    sym_S_matrix,
    sym_metabolites,
    value_array,
    symbolic_flux,
    tspan,
    tstep,
    idx_of_meta_interest,
    value_meta_interest,
):
    """Computes the integral of the dynamics based on euler's method, while keeping a set of metabolites fixed.

    Args:
        sym_S_matrix: matrix of sympy symbols that corresponds to the S matrix in LIFE
        sym_metabolites: a list of sympy symbols that appears within sym_S_matrix
        value_array: a list of values that are in parallel to sym_metabolites
        symbolic_flux: a list of sympy symbols for the constant flux vector in LIFE
        tspan: list of two values, starting point in integration and ending point
        tstep: the step in t that we take for each step in Euler's Method
        idx_of_meta_interest: a list of indices for the corresponding metabolites of interest
        value_meta_interest: a list of values corresponding to the metabolites of interest in idx_of_meta_interest


    Returns:
        t_array: a list of floats from tspan[0] to tspan[1] with stepsize tstep
        meta_value_array: the calculated trajectory of mass for each metabolite within the network
    """
    t_current = tspan[0]
    t_final = tspan[1]

    t_array = [t_current]
    value_array[idx_of_meta_interest] = value_meta_interest
    for i in range(len(idx_of_meta_interest)):
        idx = idx_of_meta_interest[i]
        val = value_meta_interest[i]
        # update the value array
        value_array[idx] = val
    meta_values_array = [value_array]

    while t_current < t_final:
        zipped_values = zip(sym_metabolites, value_array)
        integration_step = Matrix(value_array) + tstep * sym_S_matrix.subs(
            zipped_values
        ) * symbolic_flux.subs(zipped_values)
        value_array = integration_step

        intermediate = value_array.T.tolist()
        # for each index of interest, set the value in intermediate
        # intermediate[0][idx_of_meta_interest] = value_meta_interest

        for i in range(len(idx_of_meta_interest)):
            idx = idx_of_meta_interest[i]
            val = value_meta_interest[i]
            # adjust the intermediate value to fix the value of the metabolite of interest
            intermediate[0][idx] = val

        # meta_values_array.append(*value_array.T.tolist())
        meta_values_array.append(*intermediate)
        t_current += tstep
        t_array.append(t_current)

    return t_array, meta_values_array


def extract_indices_of_interest(list_meta_of_interest, actual_meta_lookup):
    """Returns a list of indices for the given list of metabolite names.

    Args:
        list_meta_of_interest: a list of metabolite names of interest
        actual_meta_lookup: a dictionary mapping metabolite names to their indices

    Returns:
        indices: a list of indices for the given metabolite names
    """
    indices = []
    for meta in list_meta_of_interest:
        indices.append(actual_meta_lookup[meta])
    return indices


def plot_results_of_interest(t, values, meta_of_interest, actual_lookup):
    """Plots the results of the simulation for a subset of metabolites.

    Args:
        t: a list of timepoints
        values: a list of metabolite levels during simulation (parallel with t)
        meta_of_interest: a list of metabolite names to plot
        actual_lookup: a python dictionary where keys are metabolites, values are indices of metabolites

    Returns:
        None: displays a matplotlib.pyplot plot
    """
    indices_of_interest = []
    for meta in meta_of_interest:
        indices_of_interest.append(actual_lookup[meta])
    values_of_interest = []
    for entry in values:
        vals = [entry[idx] for idx in indices_of_interest]
        values_of_interest.append(vals)

    plt.plot(t, values_of_interest, label=meta_of_interest)
    plt.ylim([0, 3])
    plt.legend()
    plt.show()


def func_to_int(value_array, t, sym_matrix, sym_flux, sym_metabolites):
    """Function to facilitate scipy.odeint runge-kutta integration.

    Args:
        value_array: a list of initial values for metabolites
        t: np.linspace describing the time-step settings
        sym_matrix: sympy matrix containing the S matrix
        sym_flux: sympy matrix containing the fluxes
        sym_metabolites: list of sympy variables for the metabolites

    Returns:
        deriv_array: the derivative for each metabolite

    """
    zipped_vars = zip(sym_metabolites, value_array)
    deriv = sym_matrix.subs(zipped_vars) * sym_flux.subs(zipped_vars)

    deriv_array = [ele for ele in deriv]

    return deriv_array


def specific_metabolites_func_to_int(
    value_array,
    t,
    sym_matrix,
    sym_flux,
    sym_metabolites,
    indices_of_fixed_meta,
    values_of_fixed_meta,
):
    """Function to facilitate scipy.odeint runge-kutta integration with specified metabolite values.

    Args:
        value_array: a list of initial values for metabolites
        t: np.linspace describing the time-step settings
        sym_matrix: sympy matrix containing the S matrix
        sym_flux: sympy matrix containing the fluxes
        sym_metabolites: list of sympy variables for the metabolites
        indices_of_fixed_meta: list of indices of fixed metabolites
        values_of_fixed_meta: list of values for the deriv of the metabolites

    Returns:
        deriv: the derivative for each metabolite

    """
    zipped_vars = zip(sym_metabolites, value_array)
    num_matrix = np.array(sym_matrix.subs(zipped_vars)).astype(np.float64)
    num_flux = np.array(sym_flux.subs(zipped_vars)).astype(np.float64)
    deriv = np.matmul(num_matrix, num_flux)

    for i in range(0, len(indices_of_fixed_meta)):
        deriv[indices_of_fixed_meta[i]] = values_of_fixed_meta[i]

    return deriv.reshape((deriv.shape[0],))


if __name__ == "__main__":
    # file_name = 'simple_net.xlsx'
    # file_name = 'uber_edge_example.xlsx'
    file_name = "data/simple_pd_network.xlsx"
    net_df = pd.read_excel(file_name)

    uni_meta = extract_metabolites(net_df)
    meta_lookup = create_meta_lookup(uni_meta)

    source_sink_lookup, actual_meta = find_source_sink_index(meta_lookup)

    actual_meta_lookup = create_meta_lookup(actual_meta)  # lookup without sources/sinks

    # create the symbolic metabolite array and the value array
    sym_array = create_symbolic_meta_array(actual_meta_lookup)
    val_array = create_value_array(sym_array)
    print("sym_array:", sym_array)
    print("val_array:", val_array)

    # find idx of clearance and set to 0
    # we do this for ease of comparing runs!
    idx_of_clearance = actual_meta_lookup["clearance_0"]
    val_array[idx_of_clearance] = 0

    # create the symbolic matrix using the symbolic metabolite array
    list_S_matrix = create_S_matrix(
        net_df=net_df, actual_meta_lookup=actual_meta_lookup, sym_array=sym_array
    )
    sym_S_matrix = Matrix(list_S_matrix).T  # convert to sympy matrix

    # create the flux
    num_edges = len(net_df["tail"].values)
    num_flux = [1 for i in range(0, num_edges)]
    sym_flux = Matrix(num_flux)

    # create of list of metabolites that we want to fix the values for
    list_meta_of_interest = ["gba_0"]
    indices_meta_of_interest = extract_indices_of_interest(
        list_meta_of_interest, actual_meta_lookup
    )
    print("indices of interest for meta", indices_meta_of_interest)
    values_of_interest = [2.5]  # the values that we want to set the metabolites to
    print("fixed values of interest", values_of_interest)

    # commented out - now using odeint (for faster testing, could use this function)
    # t, val = modified_euler_integrate(sym_S_matrix, sym_array, val_array, sym_flux, [0, 5], 0.05, indices_meta_of_interest, values_of_interest)

    metabolites_to_plot = [
        "glucosylceramide_0",
        "gba_0",
        "a_syn_0",
        "a_syn_1",
        "mis_a_syn_0",
        "mis_a_syn_1",
        "a_syn_proto_0",
        "mutant_lrrk2_0",
        "clearance_0",
    ]
    # plot the trajectories of the specified metabolites
    # plot_results_of_interest(t=t, values=val, meta_of_interest=metabolites_to_plot, actual_lookup=actual_meta_lookup)

    # dummy variables to be clear with odeint testing
    val_0 = val_array

    # we set the value of the metabolites - use if fixing a specific metabolite to a specific value
    derivs_of_interest = (
        []
    )  # need a new variable to hold the derivatives of the fixed metabolites
    for i in range(0, len(indices_meta_of_interest)):
        val_0[indices_meta_of_interest[i]] = values_of_interest[
            i
        ]  # set the initial value of the metabolite
        derivs_of_interest.append(
            0
        )  # add a zero for the derivatives (so the metabolite remains constant)

    # dummy variables to make it clear how we are using scipy.odeint
    arg_a = sym_S_matrix
    arg_b = sym_flux
    arg_c = sym_array
    arg_d = indices_meta_of_interest
    arg_e = derivs_of_interest

    t = np.linspace(
        0, 7, 20
    )  # start, finish, number of steps; specifies time step and range of integration
    sol = odeint(
        specific_metabolites_func_to_int,
        val_0,
        t,
        args=(arg_a, arg_b, arg_c, arg_d, arg_e),
    )
    plot_results_of_interest(
        t, sol, meta_of_interest=metabolites_to_plot, actual_lookup=actual_meta_lookup
    )
