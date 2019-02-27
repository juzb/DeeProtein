import os
import sys
import pandas as pd
from Bio import pairwise2


def main(in1, in2, out):
    in1_df = pd.read_csv(in1, sep="\t", index_col=['replicate', 'idx'])
    in2_df = pd.read_csv(in2, sep="\t")

    s1 = ''.join(in1_df.loc[0, 'AA'][1:])
    s2 = ''.join(in2_df['AA'][1:])

    alignment_obj = pairwise2.align.globalxx(s1, s2, one_alignment_only=True)[0]
    a1, a2 = alignment_obj[:2]

    columns = list(in2_df.columns)

    aligned_contents = {c: [] for c in columns}

    current_pos = 0
    for aa1, aa2 in zip(a1, a2):
        if aa1 == '-':
            continue

        if aa2 == '-':
            for c in columns:
                aligned_contents[c].append('nan')

        else:
            for c in columns:
                aligned_contents[c].append(in2_df.loc[current_pos, c])

            current_pos += 1

    for c in columns:
        if c in ['AA', 'dis', 'sec', 'Pos']:
            continue
        if c in in1_df.columns:
            c = 'added_' + c

        for replicate in in1_df.index.levels[0]:
            in1_df.loc[(replicate, slice(None)), c] = ['nan'] + aligned_contents[c]

    in1_df.to_csv(out, sep="\t")
    print('Done')


if __name__ == '__main__':
    in1 = sys.argv[1]
    in2 = sys.argv[2]
    out = sys.argv[3]
    main(in1, in2, out)
