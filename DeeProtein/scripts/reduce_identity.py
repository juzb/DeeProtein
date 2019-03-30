import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from operator import is_not
import pandas as pd
import numpy as np

levels = np.arange(0.05, .55, 0.05)
blast_cols = {"query_name": "str",
              "subject_name": "str",
              "pident": "float",
              "nident": "int",
              "sub_len": "int"}


def pll_maxident(filename):
    df = pd.read_table(filename, comment="#", names=[*blast_cols], dtype=blast_cols)
    if not len(df.index):
        return None
    max_pident = df.pident.max() / 100
    max_nident = df.nident.max()
    name = df.query_name[0]
    return name, max_pident, max_nident


def main():
    blast_dir = sys.argv[1]
    in_ds = sys.argv[2]
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    in_df = pd.read_csv(in_ds, sep=";", names=["name", "seq", "gos"], dtype="str").set_index("name")
    blast_results = [os.path.join(blast_dir, f) for f in os.listdir(blast_dir) if os.path.isfile(os.path.join(blast_dir, f))]
    with Pool(cpu_count()) as pool:
        maxs = pool.map(pll_maxident, blast_results)
    maxs = list(filter(partial(is_not, None), maxs))
    max_df = pd.DataFrame(maxs, columns=["name", "local_ident", "nident"]).set_index("name")
    in_df["seq_len"] = in_df["seq"].map(len)
    jdf = in_df.join(max_df).fillna(0)
    jdf["global_ident"] = jdf["nident"] / jdf["seq_len"]
    for level in levels:
        for mode in ["global", "local"]:
            out_df = jdf[jdf[mode + "_ident"] < level]
            out_path = os.path.join(out_dir, "test_cafa_{}_ident_lt_{:.2f}".format(mode, level))
            out_df[["seq", "gos"]].to_csv(out_path, sep=";", header=False)


if __name__ == "__main__":
    main()
