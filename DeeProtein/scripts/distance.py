import sys
import pandas as pd
import numpy as np
import Bio.PDB as pdb
from Bio import pairwise2
import os
import json
import scipy.stats as stats
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

style_path = sys.argv[-1]  # 'path/to/style/directory'

plt.style.use(json.load(open(os.path.join(style_path, 'style.json'), 'r')))
with open(os.path.join(style_path, 'colors.json'), 'r') as pickle_file:
    colors = json.load(pickle_file)
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

three2single = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def assign_sensitivity(structure, md_df, chain, pdb_path, go):
    """
    Changed:
    lookup the sensitivities directly in the df, no dict.
    :param structure:
    :param md_df:
    :param chain:
    :param pdb_path:
    :return:
    """
    seq_pdb = []
    residues = structure[0][chain]
    for res in residues:  # move along the protein chain
        if not pdb.is_aa(res):
            continue
        aa = three2single[res.get_resname()]
        seq_pdb.append(aa)
    # get the sequence:
    aas = ''.join(md_df['AA'].values[1:].tolist())

    # align

    seq_md = ''.join(md_df['AA'][1:])
    aligned_md, aligned_pdb, identity = water(seq_md, seq_pdb)

    gos = [c for c in md_df.columns if c.startswith('GO:')]

    for aa_md, aa_pdb, res, pos in zip(aligned_md, aligned_pdb, residues, range(len(aligned_md))):
        if aa_md == '-' or aa_pdb == '-':
            continue
        res.sensitivity = {go: md_df.loc[pos, go] for go in gos}
    return structure


def separate_values(data, category, radius, prefix):
    """
    TODO docstr
    :param data:        The data to separate as dict with 'category' and 'd-lig' as keys
    :param category:    category to compare for close vs. distant residues
    :return:            the values of 'category' that are close/distant as two lists
    """
    close = []
    distant = []
    zero_distance_idxes = []
    go = [g for g in data.columns if category in g][0]

    for sen, d in zip(data[go], data['d_ligand']):
        if d < radius:
            close.append(sen)
            if d == 0.0:
                zero_distance_idxes.append(len(close) - 1)
        else:
            distant.append(sen)

    return close, distant, zero_distance_idxes


def plot_and_compare(c, d, zero_distance_idxes, plot_path, pdbid, chain, go, name, ligand, percentage=5):
    """
    Plots and performs statistic
    :param c: list of close values
    :param d: list of distant values
    :param t: todos, i.e. information in a dict with the following keys: 'PDB', 'Chain', 'GO', 'Name', 'Ligand'
    :return: fold-change, p-value based on bootstrapping
    """
    all_values = c + d

    percentile = np.percentile(all_values, percentage)

    n_less_close = np.sum(np.less(c, percentile))
    rel_less_close = n_less_close / len(c)

    fold = rel_less_close / (0.01 * percentage)

    statistic = fold

    fig, ax = plt.subplots(figsize=[1.0, 1.5])

    std = 0.1
    x_c = list(np.random.normal(1, std, len(c)))
    x_d = list(np.random.normal(2, std, len(d)))

    x = np.asarray(x_c + x_d)
    y = np.asarray(c + d)

    x_red = x[zero_distance_idxes]
    y_red = y[zero_distance_idxes]

    zero_distance_below_threshold = []
    try:
        for val in y_red:
            zero_distance_below_threshold.append(val < percentile)
    except TypeError:  # if its just one value
        zero_distance_below_threshold.append(y_red < percentile)
    ax.scatter(x, y, s=1, color=colors['blue'], alpha=0.5)

    ax.scatter(x_red, y_red, s=1, color=colors['red'], alpha=1.0)

    ax.plot([0.5, 2.5], [percentile] * 2, color='black', lw=1)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Close\n{}'.format(len(c)), 'Distant\n{}'.format(len(d))])

    plt.savefig(os.path.join(plot_path, '{}_{}_{}_{}-{}.pdf'.format(pdbid, chain, go.replace(':', '-'), name,
                                                                    ligand.replace('/', '-'))))
    plt.savefig(os.path.join(plot_path, '{}_{}_{}_{}-{}.png'.format(pdbid, chain, go.replace(':', '-'), name,
                                                                    ligand.replace('/', '-'))))

    plt.close()

    # bootstrap:
    n_bs = 10000
    folds = []
    for bs_idx in range(n_bs):
        # sample:
        bs_c = random.sample(all_values, len(c))

        bs_n_less_close = np.sum(np.less(bs_c, percentile))
        bs_rel_less_close = bs_n_less_close / len(c)
        bs_fold = bs_rel_less_close / (0.01 * percentage)
        folds.append(bs_fold)

    p = (1 / (n_bs + 1)) * (1 + np.sum(np.greater_equal(folds, fold)))

    return statistic, p, zero_distance_below_threshold


def plot_dist(data, plot_path, suffix='', int_range=False):
    """
    Plot the distribution of P-values.
    :param data:        list of datapoints to plot
    :return:
    """
    if len(suffix) > 0:
        suffix = '_' + suffix

    data = [float(d) for d in data]
    data = sorted(data, key=lambda x: -x)
    fig, ax = plt.subplots()

    if int_range:
        for i in range(len(data)):
            if data[i] > 7:
                data[i] = 7.1
            if data[i] == 0.0:
                data[i] = 0.5
        print('max: {}'.format(int(np.nanmax(data)) + 2))
        locs = range(int(np.nanmax(data)) + 2)
    else:
        locs = None

    if locs:
        borders = ax.hist(data, bins=locs)
        labels = locs
        plt.xticks(locs, labels)
        ax.set_xlabel('Fold enrichment')

    else:
        for i in range(len(data)):
            if data[i] < 0.0001:
                data[i] = 0.0001
            if data[i] >= 1:
                data[i] = 0.9999
        data = np.log10(data)

        borders = ax.hist(data, bins=[-np.inf] + list(np.arange(-4, -0, 0.5)) + [-0.000001])

        print('Histogram borders: {}'.format(', '.join([str(e) for e in np.exp(borders[1])])))
        ax.set_xlim([-4.49, 0.0])
        ax.set_xlabel('log10 P-value')

    locs, labels = plt.yticks()
    n_locs = []
    n_labels = []
    for (loc, label) in zip(locs, labels):
        if int(loc) == float(loc):
            n_locs.append(loc)
            n_labels.append(int(loc))

    plt.yticks(n_locs, n_labels)

    ax.set_ylabel('Count')
    plt.tight_layout()
    fig.set_size_inches(2.36, 2.36, forward=True)
    plt.savefig(os.path.join(plot_path, 'distribution{}.pdf'.format(suffix)), transparent=True)

    plt.savefig(os.path.join(plot_path, 'distribution{}.svg'.format(suffix)), transparent=True)

    plt.savefig(os.path.join(plot_path, 'distribution{}.png'.format(suffix)), dpi=300, transparent=True)
    plt.close()


def one_entry(entry, radius, out_path, sen_path, plot_path, prefix):
    print(entry)
    pdbid = entry.loc['PDB']
    chain = entry.loc['Chain']
    go = entry.loc['GO']
    name = entry.loc['Name']
    ligand = entry.loc['Ligand']

    try:
        lig_ids = entry.loc['LigID'].split(',')
    except AttributeError:
        print('LigID not given for {}_{}'.format(pdbid, chain))
        return '!LigID', entry, None
    # call parser:
    parser = pdb.PDBParser()

    # get data
    sensitivity_file = os.path.join(sen_path, 'masked_{}_{}_1.txt'.format(pdbid, chain))
    print('Reading {}'.format(sensitivity_file))

    try:
        md_df = pd.read_csv(sensitivity_file, sep='\t', index_col=0)

    except FileNotFoundError:
        print('File not found: {}'.format(sensitivity_file))
        return '!Senstivity file', entry, None

    pdb_file = os.path.join(pdb_path, '{}.pdb'.format(pdbid))
    print('Using PDB-File {}'.format(pdb_file))

    try:
        struc = parser.get_structure(id='{}_{}'.format(pdbid, chain), file=pdb_file)
    except FileNotFoundError:
        print('File not found: {}'.format(pdb_file))
        return '!PDB file', entry, None

    # calculate each residues distance to the ligand
    # select ligands
    ligs = []

    for lig_id in lig_ids:
        lig_chain, lig_name, lig_resid = lig_id.split('/')
        try:
            lig = struc[0][lig_chain][' ', int(lig_resid), ' ']
        except KeyError:
            lig_name = 'H_{}'.format(lig_name)
            lig = struc[0][lig_chain][lig_name, int(lig_resid), ' ']

        print(lig)
        ligs.append(lig)

    # Min instead of mean
    seq = ''.join(md_df['AA'][1:])
    distances = []
    seq_matched = ''
    for res in struc[0][chain].get_residues():  # move along the protein chain
        tmp = []
        if not pdb.is_aa(res):
            continue

        for lig in ligs:
            for lig_at in lig.get_atoms():
                for res_at in res.get_atoms():
                    tmp.append(lig_at - res_at)

        try:
            aa = three2single[res.get_resname()]
        except KeyError:
            continue
        seq_matched += aa
        distances.append(min(tmp))

    alignment_obj = pairwise2.align.globalxx(seq, seq_matched, one_alignment_only=True)[0]
    aligned_md, aligned_pdb = alignment_obj[:2]

    aligned_distances = []
    current_pos = 0
    for aa_md, aa_pdb in zip(aligned_md, aligned_pdb):
        if aa_md == '-':
            continue

        if aa_pdb == '-':
            aligned_distances.append(float('nan'))

        else:
            aligned_distances.append(distances[current_pos])

            current_pos += 1

    md_df['d_ligand'] = [0.0] + aligned_distances

    md_df.to_csv(os.path.join(out_path, 'masked_{}_{}_1.txt'.format(pdbid, chain)), sep='\t')
    # this is saved now

    # rest of the calculation
    try:
        c, d, zero_distance_idxes = separate_values(md_df, go, radius, prefix)
    except IndexError:  # (KeyError, IndexError):
        print('GO not in data for {}: {} not in {}'.format(go, md_df.columns, pdbid))
        return '!GO', entry, None

    stat, p, zero_distance_below_threshold = plot_and_compare(c, d, zero_distance_idxes, plot_path, pdbid, chain, go,
                                                              name, ligand)

    print('{} {} - {:.1e}, {:.2f} \t-> {}, zero distance values below percentile: {}'.format(pdbid, chain, p, stat,
                                                                                             ligand,
                                                                                             zero_distance_below_threshold))

    entry.loc['p'] = p
    entry.loc['stat'] = stat
    entry.loc['nc'] = len(c)
    entry.loc['nd'] = len(d)
    org = entry['Organism']
    name = entry['Name']
    lig = entry['Ligand']

    head = '{} {} - {}\n{} {}, {}, p = {:.3f}, t = {:.2f}\n\n'.format(
        org, name, lig, pdbid, chain, go, p, stat)
    return head, entry, zero_distance_below_threshold


def main(masked_dataset_path, pdb_path, ligand_binding_file, out_path, radius, prefix):
    plot_path = os.path.join(out_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    heads = []
    failed = []
    lbcsv = pd.read_csv(ligand_binding_file, sep=";")

    lbcsv['p'] = np.nan
    lbcsv['stat'] = np.nan
    lbcsv['nc'] = np.nan
    lbcsv['nd'] = np.nan
    all_zero_distance_below_threshold = []
    print(lbcsv)
    for entry in lbcsv.index:
        head, modified_entry, zero_distance_below_threshold = one_entry(lbcsv.loc[entry], radius, out_path,
                                                                        masked_dataset_path, plot_path, prefix)
        if not head.startswith('!'):
            heads.append(head)
            lbcsv.loc[entry] = modified_entry
            print(zero_distance_below_threshold)
            all_zero_distance_below_threshold += zero_distance_below_threshold
        else:
            failed.append('{}_{}\t{}\t{}'.format(lbcsv.loc[entry, 'PDB'],
                                                 lbcsv.loc[entry, 'Chain'],
                                                 lbcsv.loc[entry, 'GO'],
                                                 head))

    print(''.join(heads))
    plot_dist(lbcsv['p'], plot_path, 'p')

    plot_dist(lbcsv['stat'], plot_path, 'fold', True)

    lbcsv.to_csv(os.path.join(out_path, 'ligand_binding.csv'), sep=';')

    print('Failed for:\n * {}'.format('\n * '.join(failed)))

    print('Zero-distance residues sensitivity below threshold: {}/{}'.format(
        np.sum(np.asarray(all_zero_distance_below_threshold)), len(all_zero_distance_below_threshold)))

    print('p < 0.05 for {}'.format(np.sum(np.less([p for p in lbcsv['p'] if not np.isnan(p)], 0.05))))


if __name__ == '__main__':
    masked_dataset_path = sys.argv[1]
    pdb_path = sys.argv[2]
    ligand_binding_file = sys.argv[3]
    out_path = sys.argv[4]
    radius = float(sys.argv[5])
    prefix = sys.argv[6]
    main(masked_dataset_path, pdb_path, ligand_binding_file, out_path, radius, prefix)
