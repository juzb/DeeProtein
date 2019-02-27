import sys
import pandas as pd
import numpy as np
import os
import json
from Bio import AlignIO
from Bio.Align import AlignInfo
from Bio.SubsMat import FreqTable

import add_columns


def count_freqs(seqs):
    """
    Count the AA frequencies in given seqs.
    """
    freqs = {}
    for seq in seqs:  # .values():
        for c in seq:
            try:
                freqs[c] += 1
            except:
                freqs[c] = 1
    return freqs


def get_ic(path, pdb):
    """
    Process given MSA (in file_path). Return consensus and information content.
    """

    alignment = AlignIO.read(path, "stockholm")

    summary_align = AlignInfo.SummaryInfo(alignment)

    consensus = summary_align.dumb_consensus()

    pdb2alignid = get_pdb2alignmentid(path)

    alignid = pdb2alignid[pdb]

    aligned_pdb_seq = None
    for seq in alignment:
        if seq.id == alignid:
            aligned_pdb_seq = seq
            break

    aligned_seqs = [str(a.seq) for a in alignment]
    freqs = count_freqs(aligned_seqs)

    info_content = []
    for pos in range(len(consensus)):
        info_content.append(summary_align.information_content(start=pos, end=pos + 1,
                                                              e_freq_table=FreqTable.FreqTable(freqs,
                                                                                               dict_type=FreqTable.FREQ)))

    return consensus, info_content, aligned_pdb_seq


def get_pdb2pfam(path):
    ret = {}
    with open(path, 'r') as ifile:
        ifile.readline()
        ifile.readline()
        for line in ifile:
            pdbid, chain, spid, pfamid, coverage = line.split(',')
            pdbid = pdbid.upper()

            ret['{}_{}'.format(pdbid, chain)] = pfamid
    return ret


def get_pdb2alignmentid(path):
    ret = {}
    with open(path, 'r') as ifile:
        for line in ifile:
            if 'DR PDB' in line:
                spl = line.split(';')
                pdb = spl[1].strip().replace(' ', '_')
                alignid = spl[0].replace('DR PDB', '').replace('#=GS', '').strip()
                ret[pdb] = alignid
    return ret


def main(in_path, out_path, pfam_path, pdb2pfam_path, ic_path):
    print('mdpath:\t\t{}\nout_path:\t{}\nalignment_path:\t{}\npdb2pfam_path:\t{}\n'.format(in_path, out_path,
                                                                                           alignment_path,
                                                                                           pdb2pfam_path))
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(ic_path, exist_ok=True)

    pdb2pfam = get_pdb2pfam(pdb2pfam_path)

    files = os.listdir(in_path)
    count = 0
    for file in files:
        count += 1
        _, pdbid, chain, _ = file.split('_')
        pdb = '{}_{}'.format(pdbid, chain)
        print(pdb)

        try:
            pfamid = pdb2pfam[pdb]
        except KeyError:
            print('No Pfam ID found for {}'.format(pdb))
            continue

        print(pfamid)

        path = os.path.join(pfam_path, '{}.pfam'.format(pfamid))

        try:
            consensus, ic, seq = get_ic(path, pdb)
        except FileNotFoundError as e:
            print(e)
            continue
        except KeyError as e:
            print('{} not found in alignment.\n{}'.format(pdb, e))
            continue

        ic_only_path = os.path.join(ic_path, file)

        ic_df = pd.DataFrame([[str(s) for s in seq.seq], ic]).transpose()
        ic_df.index.name = 'idx'
        ic_df.columns = ['AA', 'ic']
        ic_df.set_index('AA')

        ic_df.to_csv(ic_only_path, sep='\t')

        in_df_path = os.path.join(in_path, file)
        out_df_path = os.path.join(out_path, file)
        # fuse to the rest:
        add_columns.main(in1=in_df_path, in2=ic_only_path, out=out_df_path)
        print('Wrote the {}th file with ic'.format(count))
    print('Done.')


if __name__ == '__main__':
    masked_dump_path = sys.argv[1]
    out_path = masked_dump_path + '_ic' if not masked_dump_path.endswith('/') else masked_dump_path[:-1] + '_ic'
    ic_path = masked_dump_path + '_ic_only' if not masked_dump_path.endswith('/') else masked_dump_path[
                                                                                       :-1] + '_ic_only'
    alignment_path = sys.argv[2]
    pdb2pfam_path = sys.argv[3]
    main(masked_dump_path, out_path, alignment_path, pdb2pfam_path, ic_path)
