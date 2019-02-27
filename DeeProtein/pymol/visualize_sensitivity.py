from pymol import cmd, stored
import pymol
import numpy as np
import os
import urllib
import pandas as pd
from Bio import pairwise2


"""
needs the libraries listed above accessible in PyMOL
In PyMOL:

PyMOL> run path/to/visualize_sensitivity.py

PyMOL> color_sensitivity(file,             # sensitivity value file, name should be '<prefix>_<PDB ID>_<Chain ID>_<suffix>.<ending, usually txt>
                         column=None,      # column to color according to
                         show_hetatm=True, # whether to show hetatm-flagged atoms by default, e.g. ligands
                         show_chains=True, # whether to show other chains, e.g. interacting proteins in the pdb file
                         on_chain=None,    # redirect coloring to other chain the file-name suggests
                         on_pdb=None,      # redirect coloring to other pdb file
                         reload=True,      # whether to reload the 3D structure, can take a few seconds
                         normalize=True,   # whether to normalize the values to color, keeps zero at zero, does not work well, if there are few outliers
                         min_val=-1,       # instead of normalization, the upper and the lower border for coloring can be given here \n
                         max_val=1        # min_val and max_val must have the same absolute value, if a value of 0 shall be colored white
                        )      
Can deal with a few mismatches between the sensitivity file and the sequence in the pdb file, performs a pairwise alignment for this purpose.

"""

try:
    from goatools.obo_parser import GODag
except:
    GODag = None


three2single = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

pdb_path = "pdbs"
out_path = "out"
label_path = "labels"
if not os.path.isdir(pdb_path):
    os.makedirs(pdb_path)
if not os.path.isdir(out_path):
    os.makedirs(out_path)
if not os.path.isdir(label_path):
    os.makedirs(label_path)

stored.pdbid = None
stored.chain = None
stored.df    = None


def get_pdb_file(pdbid):
    """
    Downloads PDB file with specified ID if it is not already present in the specified path. Tries 10 times and gives
    up then.
    :param path:    Path in which the PDB file is first searched, and stored if it is downloaded
    :param pdbid:   PDB id of the file in question
    :return:        True, if successful, false if the file was not found and could not be downloaded
    """

    path = os.path.join(pdb_path, '{}.pdb'.format(pdbid))
    if not os.path.isfile(path):
        print('Downloading PDB file {} to {}'.format(pdbid, path))
        fails = 0
        while True:
            try:
                urllib.urlretrieve('https://files.rcsb.org/download/{}.pdb'.format(pdbid), path)
                break
            except IndexError:
                if fails == 10:
                    print('FAILED')
                    return False
                fails += 1
        if not check_pdb_file(path):
            print('Could not find a PDB file for the given id {}.'.format(pdbid))
            return False
        print('Done.')
    else:
        print('Already had {}.pdb'.format(pdbid))
    return True

def check_pdb_file(path):
    with open(path, 'r') as ifile:
        if '404 Not Found' in ifile.read(): 
            return False
        else:
            return True

def color_sensitivity(file, column=None, show_hetatm=True, show_chains=True, on_chain=None, reload=True, normalize=True, min_val=-1, max_val=1, on_pdb=None):
    _, pdbid, chain, _ = file.split('_')
    if on_chain:
        chain = on_chain 
    if on_pdb:
        pdbid = on_pdb

    if not stored.pdbid == pdbid or reload:
        stored.pdbid = pdbid
        stored.chain = chain

        stored.df = pd.read_csv(os.path.join(label_path, file), sep='\t')
        stored.col_pos = 0

        get_pdb_file(pdbid)
        cmd.load(os.path.join(pdb_path, '{}.pdb'.format(pdbid)))
        print('Columns:\n{}'.format(', '.join(stored.df.columns)))

    # actual coloring
    cmd.alter("sele", "b=0.0")
    cmd.select("all")
    cmd.hide("everything")
    cmd.select("sele", "chain {}".format(chain))
    cmd.show("cartoon", "sele")
    cmd.color("grey", "sele")

    if show_hetatm:
        cmd.select("het", "hetatm")
        cmd.show("sticks", "het")
        cmd.color("yellow", "het")

    if show_chains:
        cmd.select("other", "not chain {}".format(chain))
        cmd.show("cartoon", "other")
        cmd.color("orange", "other")

    if not column:
        column = stored.df.columns[stored.col_pos % len(stored.df.columns)]
        print('Showing {}'.format(column))
    seq_df = ''.join(stored.df['AA'][1:])
    seq_pdb = cmd.get_fastastr('chain {}'.format(chain))
    seq_pdb = ''.join(seq_pdb.split()[1:])
    print('Seq sensitivity:\n{}\n\nSeq pdb:\n{}\n\n'.format(seq_df, seq_pdb))
    alignment_obj = pairwise2.align.globalms(seq_df, seq_pdb, 1, 0, -.5, -.1)[0] #one_alignment_only=True,
    print(alignment_obj)
    aligned_df, aligned_pdb = alignment_obj[:2]

    stored.aligned_values = []
    current_pos = 0
    for aa_df, aa_pdb in zip(aligned_df, aligned_pdb):
        if aa_df == '-':
            stored.aligned_values.append(float('nan'))

        elif aa_pdb == '-':
            current_pos += 1
            continue

        else:
            stored.aligned_values.append(stored.df.loc[current_pos+1, column])
            current_pos += 1
    if normalize:
        stored.aligned_values = list(np.asarray(stored.aligned_values / np.nanmax(np.abs(stored.aligned_values))))
    else:
        pass

    for idx in range(len(stored.aligned_values)):
        if np.isnan(stored.aligned_values[idx]):
            stored.aligned_values[idx] = -1000


    stored.aligned_values = iter(stored.aligned_values)

    stored.last_resi = -1
    def helper(resi):
        try:
            if resi != stored.last_resi:
                stored.b = next(stored.aligned_values)
        except StopIteration:
            stored.b = np.nan

        stored.last_resi = resi
        return stored.b

    stored.helper = helper

    cmd.alter("sele", "b = stored.helper(resi)") 

    cmd.spectrum("b", "red_white_blue", "sele", str(min_val), str(max_val))
    cmd.select("nan", "b=-1000")
    cmd.color("grey", "nan")
    stored.col_pos += 1


cmd.extend("color_sensitivity", color_sensitivity)
