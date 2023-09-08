import numpy as np
import pandas as pd

from metabolicpd.life.network import Metabolic_Graph


class Test_Metabolic_Graph:
    mg = Metabolic_Graph("data/simple_pd_network.xlsx")
    mg2 = Metabolic_Graph(
        "data/central_carbon_tb.xlsx",
        mass=np.array(
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
        ),
        flux=np.array(
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
        ),
        source_weights=None,
        t_0=0,
        t=80,
        num_samples=750,
    )

    def test_df_type(self):
        """Test DataFrame is Initialized to DataFrame Type"""
        assert isinstance(self.mg.network, pd.DataFrame)

    def test_mtb_types(self):
        """Test DataFrame Schema"""
        schema = np.zeros(
            37,
            dtype={
                "names": ("name", "type", "fixed", "index"),
                "formats": ("<U32", "<U6", "bool", "<i4"),
            },
        )
        assert type(self.mg.mtb) == type(schema)

    def test_df_rows(self):
        """Test for expected number of rows."""
        assert self.mg.network.shape[0] == 28

    def test_mg_edges(self):
        """Test for expected number of edges."""
        num_edges = len(
            self.mg.network["tail"].values
        )  # the number of cols in the matrix
        assert num_edges == 28

    def test_mg2_mass_result(self):
        """Test results don't change with fixed inputs"""
        computed_result = self.mg2.simulate()
        expected_result = np.array(
            [
                3.64636251e00,
                4.27417772e01,
                2.39749152e-04,
                3.71710853e-04,
                5.43789273e00,
                8.98866591e-01,
                2.12753919e-04,
                1.40284100e-03,
                2.39505604e00,
                2.54365606e00,
                5.34771183e-04,
                4.32250082e-04,
            ]
        )
        assert np.allclose(
            np.sort(np.array(computed_result["y"].T[-1])),
            np.sort(expected_result),
        )
