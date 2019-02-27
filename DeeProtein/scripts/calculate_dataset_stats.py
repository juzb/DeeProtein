import pandas as pd
import argparse


def count_dataset(dataset_file, contains_non_uniprot=False):
    """
    Calculate important statistics solely derived from the dataset file
    :param dataset_file: Path to a dataset csv file
    :param contains_non_uniprot: Specify that you don't have uniprot identifiers necessary for organism count
    :return: A nested dict with first key GO identifier, second keys [n_samples, len, n_gos, n_org]
    """
    ret = {}
    go_counts = {}
    cnu = contains_non_uniprot
    # self.logger.info('Counting dataset {}'.format(self.FLAGS.traindata))
    with open(dataset_file, 'r') as ifile:
        for line in ifile:
            line_content = line.strip().split(';')
            id = line_content[0]
            if not cnu:
                try:
                    org = id.split('_')[1]
                except IndexError:
                    cnu = True
            seq = line_content[1]
            gos = line_content[2].strip().split(',')
            for go in gos:
                if go not in ret:
                    ret[go] = {}
                    ret[go]['len'] = 0
                    ret[go]['n_gos'] = 0
                    ret[go]['orgs'] = []
                    go_counts[go] = 0

                ret[go]['len'] += len(seq)
                ret[go]['n_gos'] += len(gos)
                if not cnu:
                    ret[go]['orgs'].append(org)
                go_counts[go] += 1

        for go in ret.keys():
            ret[go]['len'] /= go_counts[go]
            ret[go]['n_gos'] /= go_counts[go]
            if cnu:
                ret[go]["n_org"] = "NaN"
            else:
                ret[go]['n_org'] = len(set(ret[go]['orgs']))
            del ret[go]['orgs']
            ret[go]['n_samples'] = go_counts[go]

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Writes a gos.csv file for a training dataset with stats on the gos.")
    parser.add_argument("input", help="Input dataset CSV file")
    parser.add_argument("output", help="Output CSV file")
    args = parser.parse_args()
    ret = count_dataset(args.input)
    df = pd.DataFrame.from_dict(ret, orient="index")
    df.to_csv(args.output, columns=["n_samples", "len", "n_gos", "n_org"], index_label="GO",
              header=["n_samples", "avg_len", "avg_n_gos", "n_orgs"])
