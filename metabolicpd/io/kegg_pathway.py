import os

import Bio.KEGG.REST as bp
from Bio.KEGG.KGML import KGML_parser

out_dir = "data/kegg"
kegg_id = "mtu01200"

if __name__ == "__main__":
    kgml_path = out_dir + "/" + kegg_id + ".xml"
    if not os.path.isfile(kgml_path):
        print("Grabbing KGML using API...")
        with open(kgml_path, "w") as f:
            f.write(bp.kegg_get(kegg_id, "kgml").read())
    with open(kgml_path) as f:
        pathway = KGML_parser.read(f)
    print(pathway)
