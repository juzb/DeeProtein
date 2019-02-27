"""
Script that takes a masked_dataset_dump and calculates the sphere variance.
"""
import sys, os
import warnings
import pandas as pd
import numpy as np
import Bio.PDB as pdb
import scipy.stats as stats
from Bio import pairwise2
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from itertools import repeat
from multiprocessing import Pool

radius = 5.0  # in angstrom
exclude = 3  # number of residues following and preceding the queried residue to exclude from calculation
bs = 1000  # number of permutations of sensitivity scores to test against

three2single = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def calculate_sphere_variance(structure, chain, md_df, mapping):
    """
    Calculates the sphere variance and returns the corresponding statistic
    Works on a single seq/structure
    :param structure: Bio.PDB structure object
    :param chain: Chain ID as str
    :param md_df: Dataframe containing sensitivity (masked_dump file)
    :param mapping: Dict mapping sensitivity coordinates to PDB coordinates
    :return: A data frame containing the statistics per GO
    """
    # get the sequence:
    seq_len = len(md_df['AA'][1:].tolist())
    # get the list of gos:
    gos = {c.split('_')[0] for c in md_df.columns if c.startswith('GO:')}
    allowed = set(mapping.values())

    # get the residiues:
    residues = structure[0][chain]

    sphere_variances_per_go = {go: [] for go in gos}
    mean_variances = {go: 0. for go in gos}
    mean_fakes = {go: np.zeros(bs) for go in gos}
    p_vals = {}
    emp_vals = {}
    bs_mean = {}
    bs_median = {}
    # for residue in protein
    n_clean = 0
    for seq_idx in range(-1, seq_len):
        if seq_idx == -1:
            # set all to nan:
            for go in gos:
                sphere_variances_per_go[go].append(np.nan)
        elif seq_idx not in mapping:
            for go in gos:
                sphere_variances_per_go[go].append(np.nan)
            continue
        else:
            try:
                res = residues[mapping[seq_idx]]
                ca = res['CA']
            except:
                for go in gos:
                    sphere_variances_per_go[go].append(np.nan)
                continue

            # get the neighbors:
            glob_resseq = mapping[seq_idx][1]
            center = ca.get_coord()
            search = pdb.NeighborSearch(atom_list=list(structure.get_atoms()))
            neighbors = search.search(center=center, radius=radius, level="R")

            n_clean_neighbors = 0
            imps = {x: [] for x in gos}
            fakes = {x: [] for x in gos}
            for n in neighbors:
                het_atm, resseq, _ = n.id
                if het_atm.strip() == '' and resseq >= 0:
                    if abs(resseq - glob_resseq) >= exclude and n.id in allowed:
                        for go in gos:
                            try:
                                imps[go].append(n.sensitivity[go])
                                fakes[go].append(n.fake[go])
                                n_clean_neighbors += 1
                            except AttributeError:
                                break

            if n_clean_neighbors == 0:
                for go in gos:
                    sphere_variances_per_go[go].append(np.nan)
                continue

            for go in gos:
                current_var = np.nanvar(imps[go])
                current_fake = np.nanvar(fakes[go], axis=0)
                sphere_variances_per_go[go].append(current_var)
                mean_variances[go] += np.nan_to_num(current_var)
                mean_fakes[go] += np.nan_to_num(current_fake)

            n_clean += 1

    # normalize the variances:
    if n_clean > 1:
        for go in gos:
            mean_variances[go] /= n_clean
            mean_fakes[go] /= n_clean
            # do the test
            _, p_vals[go] = stats.ttest_1samp(mean_fakes[go], popmean=mean_variances[go], nan_policy="omit")
            emp_vals[go] = (1 + np.sum(mean_fakes[go] <= mean_variances[go])) / (1 + bs)
            bs_mean[go] = np.nanmean(mean_fakes[go])
            bs_median[go] = np.nanmedian(mean_fakes[go])

    # now convert the sphere vars to df and append:
    # sphere_variances_df = pd.DataFrame.from_dict(sphere_variances_per_go, orient='columns')
    # sphere_variances_df.columns = ['%s_%s_svar' % (c, sens_to_use) for c in sphere_variances_per_go] # as the keys were go terms
    # md_df_svar = pd.concat([md_df, sphere_variances_df], axis=1)

    record = {"mean_variances": mean_variances, "p_vals": p_vals, "emp_vals": emp_vals,
              "bs_mean": bs_mean, "bs_median": bs_median}
    df_svar = pd.DataFrame.from_records(record)
    return df_svar


def assign_sensitivity(structure, md_df, chain):
    """
    Adds the sensitivity to the structure object and creates a mapping dict from df coordinates to PDB coords
    :param structure: Bio.PDB structure object
    :param md_df: Dataframe containing sensitivity (masked_dump file)
    :param chain: Chain ID as str
    :return: structure object, mapping dict
    """
    try:
        residues = structure[0][chain]
    except KeyError:
        raise AssertionError
    gos = {c.split('_')[0] for c in md_df.columns if c.startswith('GO:')}
    # get the sequence:
    dat_seq = md_df['AA'].values[1:].tolist()
    seq_len = len(dat_seq)
    pdb_seq = [three2single[r.get_resname()] for r in residues if r.id[0] == " "]

    mapping_ids = [r.id for r in residues if r.id[0] == " "]
    # Align with parameters Reward, Mismatch, Opening, Extension
    aln_pdb, aln_dat = pairwise2.align.localms(pdb_seq, dat_seq, 5, -4, -3, -.1,
                                               gap_char=["-"])[0][:2]
    # Create a mapping dict from data frame coordinates to residue ids 
    curr_dat_offset = 0
    curr_pdb_offset = 0
    mapping = {}
    for aa_pdb, aa_dat in zip(aln_pdb, aln_dat):
        if aa_dat == "-":
            curr_pdb_offset += 1
        elif aa_pdb == "-":
            curr_dat_offset += 1
        else:
            mapping[curr_dat_offset] = mapping_ids[curr_pdb_offset]
            curr_pdb_offset += 1
            curr_dat_offset += 1
    # Go through the sequences and assign the sensitivity
    remaps = [(np.random.permutation([*mapping]) + 1) for b in range(bs)]
    for i, seq_idx in enumerate(mapping):
        residues[mapping[seq_idx]].sensitivity = {}
        residues[mapping[seq_idx]].fake = {}
        for go in gos:
            sensitivity_column = '%s_%s' % (go, "classic_+")
            try:
                imp = md_df[sensitivity_column][seq_idx + 1]
                fakes = [md_df[sensitivity_column][remaps[b][i]] for b in range(bs)]
            except IndexError:
                imp = float('nan')
            residues[mapping[seq_idx]].sensitivity[go] = imp
            residues[mapping[seq_idx]].fake[go] = fakes

    return structure, mapping


def worker(file, pdb_path, out_path):
    """
    Runs sphere variance calculation and stores statistics for given file (used for multiprocessing)
    Side effect: writes files to outpath
    :param file: masked dump file with sensitivities
    :param pdb_path: directory with the pdb files (with RCSB identifiers)
    :param out_path: directory to write results to
    :return:
    """
    try:
        split = os.path.basename(file).split('_')
        print(split)
        pdbid = split[1]
        chain = split[2]

        # call parser:
        parser = pdb.PDBParser()
        # get data
        md_df = pd.read_csv(file, sep='\t')
        pdb_file = os.path.join(pdb_path, "{}.pdb".format(pdbid))
        structure = parser.get_structure(id='{}_{}'.format(pdbid, chain), file=pdb_file)
        assert chain in structure[0]
        # assign sensitivity:
        structure, mapping = assign_sensitivity(structure, md_df, chain)
    except FileNotFoundError:
        print("Did not find dataframe or PDB for " + file)
        return
    except AssertionError:
        print("Chain {} not present in {} for {}".format(chain, pdb_file, file))
        return
    except (KeyError, IndexError):
        print("Key- or IndexError during processing of " + file)
        return

    # calculate sphere var:
    df_svar = calculate_sphere_variance(structure, chain, md_df, mapping)
    out_path = os.path.join(out_path, os.path.basename(file))
    df_svar.to_csv(out_path, sep='\t')


def main(in_path, pdb_path, outpath):
    valid_files = [os.path.join(in_path, x) for x in os.listdir(in_path) if x.endswith('.txt')]
    os.makedirs(outpath, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PDBConstructionWarning)
        pool = Pool()
        pool.starmap(worker, zip(valid_files, repeat(pdb_path), repeat(outpath)))


if __name__ == '__main__':
    masked_dataset_path = sys.argv[1]
    pdb_path = sys.argv[2]
    outpath = sys.argv[3]
    try:
        radius = float(sys.argv[4])
        exclude = int(sys.argv[5])
    except:
        pass
    main(masked_dataset_path, pdb_path, outpath)
