import re

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from scipy.integrate import odeint


def extract_metabolites(data_frame):
    """Extracts the unique names within the network dataframe.

    This function simply return all the unique names from an edge list
    held in a pandas dataframe. If an entry in the df has multiple string (separated
    by a comma and space: ", "), then it splits the entry.

    Args:
        data_frame : pandas dataframe that holds and edge list

    Returns:
        A DataFrame that contains the unique metabolites, their unique index, and their type
    """
    unique_entries = np.unique(data_frame[["tail", "head"]].values)

    metabolites = []
    for entry in unique_entries:
        elements = entry.split(", ")
        for ele in elements:
            metabolites.append(ele)
    new_unique_metabolites = list(dict.fromkeys(metabolites))

    metabolite_types = []
    for ele in new_unique_metabolites:
        # Using regular expression strings to identify source or sink terms in metabolite dictionary keys
        sink_res = re.search("^[e]+[0-9]$", ele)  # starts with e, ends with #...
        source_res = re.search("^[s]+[0-9]$", ele)  # starts with s, end with #...
        if (
            sink_res is not None
        ):  # check if we found something in the entry that matches the format of 'e###'
            metabolite_types.append("sink")
        elif (
            source_res is not None
        ):  # chick if we found something that matches format of source term
            metabolite_types.append("source")
        else:  # if we didn't find a source or sink term, then it must be an actual metabolite!
            metabolite_types.append("actual")

    temp_dict = {
        "index": [i for i in range(0, len(new_unique_metabolites))],
        "name": new_unique_metabolites,
        "type": metabolite_types,
    }

    metabolite_data_frame = pd.DataFrame(temp_dict)

    return metabolite_data_frame


def create_random_values(df):
    """Creates a numeric array based on number of actual metabolites.

    Args:
        metabolite_names: a list of metabolites in network

    Returns:
        Numpy array of random values between 0, 1 that is in parallel with sym_array
    """
    np.random.seed(seed=12)
    return df.assign(vals=np.random.rand(len(df.index)))


if __name__ == "__main__":
    data_frame = pd.read_excel("data/simple_pd_network.xlsx")
    metabolites = extract_metabolites(data_frame)
    print(type(metabolites))
    print(metabolites)
    metabolites = create_random_values(metabolites)
    print(metabolites)
    # print(metabolite_types)
    # print("index of 's6' "+str(unique_metabolites.index("a_syn_0")))
