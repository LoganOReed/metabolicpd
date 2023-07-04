import pandas as pd
import numpy as np
import re
from sympy import Matrix, symbols, exp, Min
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class LIFE_Simulator:

    def __init__(self, network_file_name):
        self.network_file = network_file_name
        self.network_df = pd.read_excel(self.network_file)

        self.all_unique_names = self.extract_uni_names(self.network_df)
        self.name_lookup = self.create_lookup(self.all_unique_names)

        source_sink_lookup, meta_list = self.find_source_sink_index(self.name_lookup)
        self.source_sink_lookup = source_sink_lookup
        self.meta_lookup = self.create_lookup(meta_list)

        self.meta_sym_array = self.create_symbolic_meta_array()
        self.val_array = self.create_value_array(rng_seed=12)

        # normalize the clearance 'node' to start at 0
        self.idx_of_clearance = self.meta_lookup['clearance_0']
        self.val_array[self.idx_of_clearance] = 0

        self.list_S_matrix = self.create_S_matrix(net_df=self.network_df, meta_lookup=self.meta_lookup, sym_array=self.meta_sym_array)
        self.symbolic_S_matrix = Matrix(self.list_S_matrix).T

        self.num_flux, self.sym_flux = self.create_flux()

        self.metabolites_to_fix = None
        self.values_of_metabolites_to_fix = None

        self.metabolites_to_plot = None

    def extract_uni_names(self, net_df):
        """ Extracts the unique names within the network dataframe

            This function simply return all the unique names from an edge list
            held in a pandas dataframe. If an entry in the df has multiple string (separated
            by a comma and space: ", "), then it splits the entry.

            Args:
                network_df : pandas dataframe that holds and edge list

            Returns:
                the unique strings contained within network_df
            """
        unique_entries = np.unique(net_df[['tail', 'head']].values)

        metabolites = []
        for entry in unique_entries:
            elements = entry.split(', ')
            for ele in elements:
                metabolites.append(ele)
        new_unique_metabolites = list(dict.fromkeys(metabolites))
        return new_unique_metabolites

    def create_lookup(self, unique_name_list):
        """ Creates a lookup dictionary that ties unique metabolite names to an index.

        Args:
            unique_name_list : python list holding unique metabolite names

        Returns:
            A dictionary where keys are metabolite names and values are indices
        """
        metabolite_lookup_dict = {}
        index_counter = 0
        for entry in unique_name_list:
            metabolite_lookup_dict[entry] = index_counter
            index_counter += 1

        return metabolite_lookup_dict

    def find_source_sink_index(self, name_lookup_dict):
        """ Searches the metabolite lookup dictionary for source and sink terms.

        Source terms are indicated by 's#' and sink terms are indicated by
        'e#'. 'e' comes from the language: "excretion in a network"

        Args:
            meta_lookup_dict : dictionary holding unique metabolite names and indices

        Returns:
            source_sink_dict : A dictionary where keys are sources and sinks and values are the corresponding index
            actual_meta_list : A pyhton list that holds actual metabolites within the network
        """
        source_sink_dict = {'source': [], 'sink': []}
        actual_meta_list = []
        for key in name_lookup_dict.keys():
            # Using regular expression strings to identify source or sink terms in metabolite dictionary keys
            sink_res = re.search('^[e]+[0-9]$', key)  # starts with e, ends with #...
            source_res = re.search('^[s]+[0-9]$', key)  # starts with s, end with #...
            if sink_res is not None:  # check if we found something in the entry that matches the format of 'e###'
                source_sink_dict['sink'].append(name_lookup_dict[key])
            elif source_res is not None:  # chick if we found something that matches format of source term
                source_sink_dict['source'].append(name_lookup_dict[key])
            else:  # if we didn't find a source or sink term, then it must be an actual metabolite!
                actual_meta_list.append(key)

        return source_sink_dict, actual_meta_list

    def create_symbolic_meta_array(self):
        """ Creates a list that holds Sympy symbols with variable names from actual_meta_lookup

        Args:
            actual_meta_lookup : a python dictionary with keys being actual metabolites within network and values being
             indices of the corresponding key

        Returns:
            symbolic_array : a list of Sympy symbols that come from the keys in actual_meta_lookup
        """
        symbolic_array = []
        for key in self.meta_lookup.keys():
            symbolic_array.append(symbols(key))

        return symbolic_array

    def create_value_array(self, rng_seed):
        """ Creates a numeric array based on number of actual metabolites

        Args:
            sym_array: a list of Sympy symbols that correspond to metabolites in network

        Returns:
            Numpy array of random values between 0, 1 that is in parallel with sym_array
        """
        np.random.seed(seed=rng_seed)
        return np.random.rand(len(self.meta_sym_array))

    def create_S_matrix(self, net_df, meta_lookup, sym_array):
        """ Create the 'S' matrix that defines the dynamics of a metabolic network.

        The form of the 'S' matrix comes follows LIFE dynamics, with edges in the network corresponding to columns in the
        matrix and metabolites in the network corresponding to the rows of the matrix.

        The indices within actual_meta_lookup correspond to the indices of sym_array.

        Args:
            net_df: pandas dataframe of an edge list encoding the metabolic network
            meta_lookup: python dictionary with keys as metabolites in network and values of indices
            sym_array: python list of Sympy symbols corresponding to the metabolites in the network

        Returns:
            A list of lists with elements being sympy symbols that encodes the dynamics of the network according to the
                LIFE approach
        """
        # number of non-source/sink metabolites
        num_metabolites = len(sym_array)  # the number of rows in the matrix
        num_edges = len(net_df['tail'].values)  # the number of cols in the matrix  -- this is unused

        s_matrix = []  # create an empty list to hold the eventual columns

        edge_columns_df = net_df[
            ['tail', 'head', 'uber']]  # get a dataframe for just the edges (future-proof for uberedges)
        for row in edge_columns_df.itertuples():  # iterate through each of the edges
            substrates = row.tail.split(', ')  # there could be more than one substrate
            products = row.head.split(', ')  # or product! (for instance, in a hyperedge)
            uber_modulators = row.uber.split(', ')

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
                if uber[-1] == '+':  # check the last term in the entry, enhancer
                    uber_meta = uber[:-2]  # get all the characters, except the last two (either _+/-)
                    index_of_uber_meta = meta_lookup[uber_meta]  # get the index of the uber metabolites
                    uber_symbol = sym_array[index_of_uber_meta]  # get the symbol of the uber metabolite
                    uber_term = uber_term * exp((uber_symbol / (uber_symbol + 1)))  # always greater than one
                elif uber[-1] == '-':  # if the last term designates an inhibitor
                    uber_meta = uber[:-2]  # get all the characters, except the last two (either _+/-)
                    index_of_uber_meta = meta_lookup[uber_meta]
                    uber_symbol = sym_array[index_of_uber_meta]
                    uber_term = uber_term * 1 / exp((uber_symbol / (uber_symbol + 1)))  # always less than one

            if (num_subs > 1 or num_prods > 1):  # hyperedge!
                index_of_subs_in_meta = []
                index_of_prods_in_meta = []
                for sub in substrates:  # get the index for all the substrates in the edge
                    index_of_subs_in_meta.append(meta_lookup[sub])
                for prod in products:
                    index_of_prods_in_meta.append(meta_lookup[prod])

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
                if re.search('^[s]+[0-9]$', substrates[0]) is not None:  # source term
                    index_of_prod_in_meta = meta_lookup[products[0]]
                    current_col[index_of_prod_in_meta] = 1  # we don't ever expect to have a source with an uber edge
                elif re.search('^[e]+[0-9]', products[0]) is not None:  # sink term
                    index_of_sub_in_meta = meta_lookup[substrates[0]]
                    current_col[index_of_sub_in_meta] = -1 * sym_array[index_of_sub_in_meta] * uber_term
                else:  # not a source or sink
                    index_of_prod_in_meta = meta_lookup[products[0]]
                    index_of_sub_in_meta = meta_lookup[substrates[0]]

                    current_col[index_of_sub_in_meta] = -1 * sym_array[index_of_sub_in_meta] * uber_term
                    current_col[index_of_prod_in_meta] = sym_array[index_of_sub_in_meta] * uber_term

            s_matrix.append(current_col)
        return s_matrix

    def create_flux(self):
        """ Creates numeric and symbolic variables for flux according to number of edges in network

        Returns:
            num_flux: list of all 1s
            sym_flux: sympy Matrix of all 1s
        """
        num_edges = len(self.network_df['tail'].values)
        num_flux = [1 for i in range(0, num_edges)]
        sym_flux = Matrix(num_flux)

        return num_flux, sym_flux

    def set_metabolites_to_fix(self, list_of_meta):
        self.metabolites_to_fix = list_of_meta
        self.indices_of_fixed_metabolites = self.extract_indices(self.metabolites_to_fix, self.meta_lookup)

    def set_values_of_metabolites_to_fix(self, list_of_values):
        self.values_of_metabolites_to_fix = list_of_values

    def set_metabolites_to_plot(self, list_metas_to_plot):
        self.metabolites_to_plot = list_metas_to_plot
    def extract_indices(self, list_of_metabolites, meta_lookup):
        """ Returns a list of indices for the given list of metabolite names.

            Args:
                list_of_metabolites: a list of metabolite names of interest
                meta_lookup: a dictionary mapping metabolite names to their indices

            Returns:
                indices: a list of indices for the given metabolite names
            """
        indices = []
        for meta in list_of_metabolites:
            indices.append(meta_lookup[meta])
        return indices

    def update_values(self):
        self.derivs_of_interest = []
        for i in range(0, len(self.indices_of_fixed_metabolites)):
            index = self.indices_of_fixed_metabolites[i]
            value = self.values_of_metabolites_to_fix[i]
            self.val_array[index] = value
            self.derivs_of_interest.append(0)

    def simulate_fixed(self, t):
        self.time = t
        arg_a = self.symbolic_S_matrix
        arg_b = self.sym_flux
        arg_c = self.meta_sym_array
        arg_d = self.indices_of_fixed_metabolites
        arg_e = self.derivs_of_interest

        solution_fixed = odeint(specific_metabolites_func_to_int, self.val_array, t, args=(arg_a, arg_b, arg_c, arg_d, arg_e))
        self.solution_fixed = solution_fixed
        return solution_fixed

    def simulate(self, t):
        self.time = t
        arg_a = self.symbolic_S_matrix
        arg_b = self.sym_flux
        arg_c = self.meta_sym_array

        solution = odeint(metabolites_func_to_int, self.val_array, t, args=(arg_a, arg_b, arg_c))
        self.solution = solution
        return solution

    def plot_results_of_interest(self):
        """ Plots the results of the simulation for a subset of metabolites

        Args:
            t: a list of timepoints
            values: a list of metabolite levels during simulation (parallel with t)
            meta_of_interest: a list of metabolite names to plot
            actual_lookup: a python dictionary where keys are metabolites, values are indices of metabolites

        Returns:
            None: displays a matplotlib.pyplot plot
        """
        indices_of_interest = []
        for meta in self.metabolites_to_plot:
            indices_of_interest.append(self.meta_lookup[meta])
        values_of_interest = []
        for entry in self.solution_fixed:
            vals = [entry[idx] for idx in indices_of_interest]
            values_of_interest.append(vals)

        plt.plot(t, values_of_interest, label=self.metabolites_to_plot)
        plt.ylim([0, 3])
        plt.legend()
        plt.show()

    def save_simulation(self, filename):
        simulation_df = pd.DataFrame(self.solution_fixed)
        time_df = pd.DataFrame(self.time)

        combined_df = pd.concat([time_df, simulation_df], axis=1)
        lookup_df = pd.DataFrame.from_dict(self.meta_lookup, orient='index')

        output_file = filename+'.xlsx'
        with pd.ExcelWriter(output_file) as writer:
            combined_df.to_excel(writer, sheet_name='sim data', index=False)
            lookup_df.to_excel(writer, sheet_name='lookup dict', index=True)


def specific_metabolites_func_to_int(value_array, t, sym_matrix, sym_flux, sym_metabolites, indices_of_fixed_meta, values_of_fixed_meta):
    """ Function to facilitate scipy.odeint runge-kutta integration with specified metabolite values

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

    return deriv.reshape((deriv.shape[0], ))


def metabolites_func_to_int(value_array, t, sym_matrix, sym_flux, sym_metabolites):
    """ Function to facilitate scipy.odeint runge-kutta integration with specified metabolite values

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

    return deriv.reshape((deriv.shape[0], ))


if __name__ == '__main__':
    file_name = 'data/simple_pd_network.xlsx'
    sim_obj = LIFE_Simulator(file_name)

    metas_to_fix = ['gba_0']
    values_of_metas = [2.5]
    # metas_to_plot = ['glucosylceramide_0', 'gba_0', 'a_syn_0', 'a_syn_1', 'mis_a_syn_0', 'mis_a_syn_1', 'a_syn_proto_0', 'mutant_lrrk2_0', 'clearance_0']
    sim_obj.set_metabolites_to_fix(metas_to_fix)
    sim_obj.set_values_of_metabolites_to_fix(values_of_metas)
    # sim_obj.set_metabolites_to_plot(metas_to_plot)
    sim_obj.update_values()

    t = np.linspace(0, 12, 30)
    sol = sim_obj.simulate_fixed(t)

    metas_to_plot = ['glucosylceramide_0', 'gba_0', 'a_syn_0', 'a_syn_1', 'mis_a_syn_0', 'mis_a_syn_1', 'a_syn_proto_0', 'mutant_lrrk2_0', 'clearance_0']
    sim_obj.set_metabolites_to_plot(metas_to_plot)
    sim_obj.plot_results_of_interest()
    sim_obj.save_simulation(filename='temp/high_gba_12h')
