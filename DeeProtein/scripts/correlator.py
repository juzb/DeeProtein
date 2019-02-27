import os
import sys
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from goatools.obo_parser import GODag


def corr(x, y):
    """
    Calculate Corr for x vs y.
    Robust to NaNs and infs, returns nans if input doesn't contain values
    :param x: input x
    :param y: input y
    :return: r, p, n
    """
    x = np.asfarray(x)
    y = np.asfarray(y)
    idx = np.isfinite(x) & np.isfinite(y)
    x = x[idx]
    y = y[idx]

    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')

    r, p = stats.pearsonr(x, y)
    n = len(x)

    return r, p, n


def compute_go_correlations(path, gos):
    """Reads all per aa-resolution files and calculates all correlations between the GO term sensitivities
    Makes a dict containing all correlations between two GOs as lists of r and p values"""
    data = {}  # ['GO:1']['GO:2'] = {'p': [1, 2, 3, ...], 'r': [4, 5, 6, ...]}
    for go1 in gos:
        data[go1] = {}
        for go2 in gos:
            data[go1][go2] = {'p': [], 'r': []}
            if go1 == go2:
                data[go1][go1]["p"] = [0.0]
                data[go1][go2]["r"] = [1.0]

    valid_files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.txt')]

    for file_name in valid_files:
        try:
            per_aa_data = pd.read_table(file_name)
            if "replicate" in per_aa_data.columns:
                per_aa_data = per_aa_data.groupby("idx").mean()
            gos_entry = [i.split("_")[0] for i in per_aa_data if i.startswith("GO:") and i.endswith("_classic_+")]

            for go1, go2 in itertools.combinations(gos_entry, 2):
                r, p, _ = corr(per_aa_data[go1 + "_classic_+"][1:], per_aa_data[go2 + "_classic_+"][1:])

                data[go1][go2]['p'].append(p)
                data[go1][go2]['r'].append(r)
                data[go2][go1]['p'].append(p)
                data[go2][go1]['r'].append(r)
        except:
            print("Issues parsing " + file_name)
    return data


def compute_ic_correlations(path, gos):
    """Reads all per aa-resolution files and calculates all correlations between the GO term sensitivities and the information contents.
    Makes a dict containing correlations for each GO as lists of r and p values"""
    data_ic = {}  # ['GO'] = {'p': [1, 2, 3, ...], 'r': [4, 5, 6, ...]}
    for go in gos:
        data_ic[go] = {'p': [], 'r': []}

    valid_files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.txt')]

    for file_name in valid_files:
        try:
            per_aa_data = pd.read_table(file_name)
            if "replicate" in per_aa_data.columns:
                per_aa_data = per_aa_data.groupby("idx").mean()
            gos_entry = [i.split("_")[0] for i in per_aa_data if i.startswith("GO:") and i.endswith("_classic_+")]

            for go in gos_entry:
                r, p, _ = corr(per_aa_data[go + "_classic_+"][1:], per_aa_data["ic"][1:])
                data_ic[go]['p'].append(p)
                data_ic[go]['r'].append(r)
        except:
            print("Issues parsing " + file_name)
    return data_ic


def analyse_go_correlations(go_correlations, godag):
    """
    Analyzes correlations between GO terms on the same sequence. Finds out which GO terms are connected by an
    'is a' (i.e. parent-child) relationship, reports their correlation. Makes boxplot of the r values per GO level of the parent.
    :return: nested list: each entry is for a pair of GO terms connected by an 'is a' relationship, each entry features
        both GO terms,
        the level of the parent term,
            the number of points for which the correlation was calculated,
            the mean r value
            the mean p value
            all r values as a list
            all p values as a list
    """
    data = []
    for go1 in go_correlations:
        for go2 in go_correlations:
            go1_obj = godag.query_term(go1)
            if go1_obj.has_child(go2):
                data.append([go1,
                             go2,
                             go1_obj.level,
                             len(go_correlations[go1][go2]['r']),
                             float(np.nanmean(
                                 go_correlations[go1][go2]['r'])) if len(
                                 go_correlations[go1][go2]['r']) != 0 else float('nan'),
                             float(np.nanmean(
                                 go_correlations[go1][go2]['p'])) if len(
                                 go_correlations[go1][go2]['p']) != 0 else float('nan'),
                             go_correlations[go1][go2]['r'],
                             go_correlations[go1][go2]['p']
                             ])
    return data


def main(path_gogo, path_ic, go_dag_file, go_file, outpath):
    with open(go_file, "r") as _go_file:
        gos = [l.split(" ")[-1].split(".")[0] for l in _go_file]
    go_dag = GODag(go_dag_file, optional_attrs=["relationship"])

    go_go = compute_go_correlations(path_gogo, gos)
    go_ic = compute_ic_correlations(path_ic, gos)
    go_lvl_go = analyse_go_correlations(go_go, go_dag)

    df_go_go = pd.DataFrame.from_records(go_go)
    df_go_ic = pd.DataFrame.from_records(go_ic)

    df_go_go_r = df_go_go.applymap(lambda x: x["r"])
    df_go_go_p = df_go_go.applymap(lambda x: x["p"])
    df_go_go_r_mean = df_go_go_r.applymap(lambda x: np.nanmean(x))
    df_go_go_p_mean = df_go_go_p.applymap(lambda x: np.nanmean(x))
    df_go_go_r_median = df_go_go_r.applymap(lambda x: np.nanmedian(x))
    df_go_go_p_median = df_go_go_p.applymap(lambda x: np.nanmedian(x))

    df_go_ic_mean = df_go_ic.applymap(lambda x: np.nanmean(x))
    df_go_ic_median = df_go_ic.applymap(lambda x: np.nanmedian(x))

    df_go_ic.to_csv(os.path.join(outpath, "df_go_ic.tsv"), sep="\t")
    df_go_ic_mean.to_csv(os.path.join(outpath, "df_go_ic_mean.tsv"), sep="\t")
    df_go_ic_median.to_csv(os.path.join(outpath, "df_go_ic_median.tsv"), sep="\t")
    df_go_go_r.to_csv(os.path.join(outpath, "df_go_go_r.tsv"), sep="\t")
    df_go_go_p.to_csv(os.path.join(outpath, "df_go_go_p.tsv"), sep="\t")
    df_go_go_r_mean.to_csv(os.path.join(outpath, "df_go_go_r_mean.tsv"), sep="\t")
    df_go_go_p_mean.to_csv(os.path.join(outpath, "df_go_go_p_mean.tsv"), sep="\t")
    df_go_go_r_median.to_csv(os.path.join(outpath, "df_go_go_r_median.tsv"), sep="\t")
    df_go_go_p_median.to_csv(os.path.join(outpath, "df_go_go_p_median.tsv"), sep="\t")
    with open(os.path.join(outpath, 'go_parent-child_correlations.tsv'), 'w') as ofile:
        ofile.write('{}\n'.format('\t'.join(['Parent', 'Child', 'parent_level', 'n', 'mean_r',
                                             'mean_p', 'comma_joined_r_values', 'comma_joined_p_values'])))
        for line in go_lvl_go:
            ofile.write('{}\n'.format(
                '\t'.join([str(l) for l in line[:6]] + [';'.join(
                    [str(l) for l in line[6]]), ';'.join([str(l) for l in line[7]])])))


if __name__ == "__main__":
    data_path = sys.argv[1]
    ic_path = sys.argv[2]
    go_dag_file = sys.argv[3]
    go_file = sys.argv[4]
    outpath = sys.argv[5]
    main(data_path, ic_path, go_dag_file, go_file, outpath)
