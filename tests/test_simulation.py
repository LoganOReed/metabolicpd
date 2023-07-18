import pandas as pd

from metabolicpd.simulation import LIFE_Network


def test_df_type():
    """Test DataFrame is Initialized to DataFrame Type"""
    network = LIFE_Network("data/simple_pd_network.xlsx")
    # print(metabolite_types)
    # print("index of 's6' "+str(unique_metabolites.index("a_syn_0")))
    assert isinstance(network.df, pd.DataFrame)


def test_df_data():
    """Test DataFrame Schema"""
    network = LIFE_Network("data/simple_pd_network.xlsx")
    schema_columns = ["index", "name", "type", "vals"]
    # print("index of 's6' "+str(unique_metabolites.index("a_syn_0")))
    assert network.df.columns.tolist() == schema_columns
