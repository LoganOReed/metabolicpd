import os
import sys

import Bio.KEGG.REST as bp
import pandas as pd
from Bio.KEGG.KGML import KGML_parser


def get_compound_name(cid):
    """Uses KEGG REST API to convert a kegg compound id (`cpd:C#####` or `C#####`) to a list of names."""
    return bp.kegg_find("compound", cid).read()


def network_from_KGML(kegg_id, out_dir="data/kegg"):
    kgml_path = out_dir + "/" + kegg_id + ".xml"
    cpd_path = out_dir + "/" + kegg_id + "_compounds.csv"
    res_path = out_dir + "/" + kegg_id + "_network.csv"
    if not os.path.isfile(kgml_path):
        print("Grabbing KGML using API...")
        with open(kgml_path, "w") as f:
            f.write(bp.kegg_get(kegg_id, "kgml").read())
    try:
        with open(kgml_path) as f:
            pathway = KGML_parser.read(f)
    except OSError:
        print("Could not open/read file: ", kgml_path)
        sys.exit(1)

    # Used to list inhibitors
    # for g in pathway.relations:
    #     if g.subtypes[0][0] == "inhibition":
    #         print(g)

    # print pathway info

    cpd_table = pd.DataFrame()
    for g in pathway.compounds:
        cpds = ""
        for c in g.name.split():
            cpds = cpds + c[4:] + " "
        new_row = pd.Series(
            {
                "reaction_id": g.id,
                "compound_id": cpds[:-1],
            }
        )
        cpd_table = pd.concat([cpd_table, new_row.to_frame().T], ignore_index=True)
    # print(cpd_table)
    cpd_table.to_csv(cpd_path, index=False)

    edge_list = pd.DataFrame()
    for g in pathway.reactions:
        t = ""
        h = ""
        for i in g.substrates:
            t = t + str(i.id) + " "
        for i in g.products:
            h = h + str(i.id) + " "

        new_row = pd.Series(
            {
                "tail": t[:-1],
                "head": h[:-1],
                "uberPos": "none",
                "uberNeg": "none",
            }
        )
        edge_list = pd.concat([edge_list, new_row.to_frame().T], ignore_index=True)
        if g.type == "reversible":
            new_row = pd.Series(
                {
                    "tail": h[:-1],
                    "head": t[:-1],
                    "uberPos": "none",
                    "uberNeg": "none",
                }
            )
            edge_list = pd.concat([edge_list, new_row.to_frame().T], ignore_index=True)
    # print(edge_list)
    edge_list.to_csv(res_path, index=False)
    return res_path


if __name__ == "__main__":
    # Carbon metabolism - Mycobacterium tuberculosis H37Rv
    kegg_id = "mtu01200"
    # kegg_id = "hsa05165"
    network_from_KGML(kegg_id)
