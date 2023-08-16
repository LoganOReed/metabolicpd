import pandas as pd

from metabolicpd.simulation import LIFE_Network


def test_df_type():
    """Test DataFrame is Initialized to DataFrame Type"""
    network = LIFE_Network("data/simple_pd_network.xlsx")
    assert isinstance(network.df, pd.DataFrame)


def test_df_data():
    """Test DataFrame Schema"""
    network = LIFE_Network("data/simple_pd_network.xlsx")
    schema_columns = ["name", "type", "fixed"]
    assert network.df.columns.tolist() == schema_columns


def test_df_rows():
    """Test for expected number of rows."""
    network = LIFE_Network("data/simple_pd_network.xlsx")
    assert network.df.shape[0] == 37


def test_network_edges():
    """Test for expected number of edges."""
    network = LIFE_Network("data/simple_pd_network.xlsx")
    num_edges = len(network.network["tail"].values)  # the number of cols in the matrix
    assert num_edges == 28
