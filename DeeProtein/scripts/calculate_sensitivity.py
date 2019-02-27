import sys, os
import pandas as pd
import numpy as np


def one_protein(infile, outfile):
    """ calc the sensitivities"""
    # read data
    md_df = pd.read_csv(infile, sep='\t', engine='python')

    # make sure to get every GO only once
    for go in [c.strip('_-') for c in md_df.columns if all(['GO:' in c, '_-' in c])]:
        plus = '%s_+' % go
        minus = '%s_-' % go

        if not all([plus in md_df.columns, minus in md_df.columns]):
            continue

        ref_plus = md_df[plus].values[0]
        ref_minus = md_df[minus].values[0]
        mut_plus = md_df[plus].values[1:]
        mut_minus = md_df[minus].values[1:]

        # calculate the sensitivities:

        sensitivity_classic_p = mut_plus - ref_plus
        md_df['%s_classic_+' % go] = np.concatenate([[0], sensitivity_classic_p]).ravel()

        sensitivity_classic_m = mut_minus - ref_minus
        md_df['%s_classic_-' % go] = np.concatenate([[0], sensitivity_classic_m]).ravel()

    md_df.to_csv(outfile, sep='\t')

def rename(infile, outfile):
    to_drop = ['+', '-', 'Unnamed: 0', 'classic_-']
   
    df = pd.read_csv(infile, sep="\t", index_col='Pos')
    
    cols_to_drop = []
    for col in df.columns:
        for d in to_drop:
            if d in col:
                if 'classic_+' in col:
                    break
                cols_to_drop.append(col)
                break # breaks the inner loop, moves to the next column

    df.drop(cols_to_drop, axis=1, inplace=True)
    df.columns = [col.replace('_classic_+', '') for col in df.columns]
    df.to_csv(outfile, sep="\t")
    
    print('DONE')
    
def main(infile, outfile):
    """
    wraps the one_protein function to calculate the sensitivity of all proteins in one directory
    """
    one_protein(infile, infile + '.tmp')
    rename(infile + '.tmp', outfile)


if __name__ == '__main__':
    # masked dataset input
    infile = sys.argv[1]
    outfile = sys.argv[2]
    main(infile, outfile)
