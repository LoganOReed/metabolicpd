import os
import sys

import Bio.KEGG.REST as bp
from Bio.KEGG.KGML import KGML_parser

# Directory path for kgml files (assumes you run from project root)
out_dir = "data/kegg"
# Carbon metabolism - Mycobacterium tuberculosis H37Rv
kegg_id = "mtu01200"
# kegg_id = "hsa05165"


def get_compound_name(cid):
    """Uses KEGG REST API to convert a kegg compound id (`cpd:C#####` or `C#####`) to a list of names."""
    return bp.kegg_find("compound", cid).read()


if __name__ == "__main__":
    kgml_path = out_dir + "/" + kegg_id + ".xml"
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
    cpd = "cpd:C00085 cpd:C05345"
    for g in pathway.reactions:
        print(g)
        for h in g.products:
            print(h.name)
    print("####################")

    cpds = []
    for g in pathway.compounds:
        cpds = cpds + g.name.split()

    cpd_path = out_dir + "/" + kegg_id + ".compounds"
    cpd_file = open(cpd_path, "w")
    for c in cpds:
        cpd_file.write(c[4:] + "\n")

    print("#######################")
