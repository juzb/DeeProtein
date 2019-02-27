import os
import sys
import pandas as pd
import numpy as np


def main(test_path, deep_go_eval_pickle):
    preds_path = os.path.join(test_path, 'metrics/raw_predictions.npy')
    labels_path = os.path.join(test_path, 'metrics/raw_labels.npy')
    ids_path = os.path.join(test_path, 'metrics/ids.npy')
    out = os.path.join(test_path, 'metrics/dg_like_df.pkl')
    print('Reading data from {} ({}, {}) and {} to write to {}.'.format(test_path,
                                                                        labels_path,
                                                                        preds_path,
                                                                        deep_go_eval_pickle,
                                                                        out))

    df = pd.read_pickle(deep_go_eval_pickle)

    preds = list(np.load(preds_path)[:, :, 0])  # only interested in the predictions on the positive node
    labels = list(np.load(labels_path))
    ids = list(np.load(ids_path))
    ids = [i.decode() for i in ids]
    dp_metrics = pd.DataFrame.from_dict({'targets': ids,
                                         'labels': labels,
                                         'predictions': preds})
    df.drop(labels=['predictions', 'labels'], axis=1, inplace=True)
    df = pd.merge(left=df, right=dp_metrics, how='inner', on='targets')

    df.to_pickle(out)
    print('Wrote new df to {}'.format(out))

    print('DONE.')


if __name__ == '__main__':
    test_path = sys.argv[1]
    deep_go_eval_pickle = sys.argv[2]

    main(test_path, deep_go_eval_pickle)
