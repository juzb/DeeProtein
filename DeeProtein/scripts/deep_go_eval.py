import os
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_recall_curve
from goatools.obo_parser import GODag


"""
Note: This code is adopted based on the evaluation.py document from the deepgo github repository 
(https://github.com/bio-ontology-research-group/deepgo/blob/master/evaluation.py).

"""


def main(path_to_pkl, path_to_godag, path_to_go_list_pkl):
    df = pd.read_pickle(path_to_pkl)
    preds = reshape(df['predictions'].values)
    labels = reshape(df['labels'].values)
    global GODag
    GODag = GODag(path_to_godag, optional_attrs=['relationship'])

    func_df = pd.read_pickle(path_to_go_list_pkl)
    functions = func_df['functions'].values
    func_index = dict()
    for i, go_id in enumerate(functions):
        func_index[go_id] = i
    global func_set
    func_set = set(func_index)

    # preds = df['predictions'].values
    gos = df['gos'].values
    f, p, r, t, preds_max = compute_performance(preds, labels, gos)
    print('f: \t{}\np: \t{}\nr: \t{}'.format(f, p, r))
    # labels = list()
    # scores = list()
    # for i in range(len(preds)):
    #     all_gos = set()
    #     for go_id in gos[i]:
    #         if go_id in all_functions:
    #             all_gos |= get_anchestors(go, go_id)
    #     all_gos.discard(GO_ID)
    #     scores_dict = {}
    #     for val in preds[i]:
    #         go_id, score = val
    #         if go_id in all_functions:
    #             go_set = get_anchestors(go, go_id)
    #             for g_id in go_set:
    #                 if g_id not in scores_dict or scores_dict[g_id] < score:
    #                     scores_dict[g_id] = score
    #     all_preds = set(scores_dict) # | all_gos
    #     all_preds.discard(GO_ID)
    #     for go_id in all_preds:
    #         if go_id in scores_dict:
    #             scores.append(scores_dict[go_id])
    #         else:
    #             scores.append(0)
    #         if go_id in all_gos:
    #             labels.append(1)
    #         else:
    #             labels.append(0)

    # scores = np.array(scores)
    # labels = np.array(labels)
    roc_auc = compute_roc(preds, labels)
    print('AUROC: \t{}'.format(roc_auc))
    
    auprc = compute_prc(preds, labels)
    print('AUPRC: \t{}'.format(auprc))
    # preds_max = (scores > t).astype(np.int32)
    mcc = compute_mcc(preds_max, labels)
    print('MCC: \t{}'.format(mcc))


def reshape(values):
    values = np.hstack(values).reshape(
        len(values), len(values[0]))
    return values


def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_prc(preds, labels):
    # Compute PR curve and area under PRC for each class
    precision, recall, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    pr_auc = auc(recall, precision)
    return pr_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def get_anchestors(GO_id): # added as a wrapper for alternative deepgo function, uses the goatools implementation
    return GODag[GO_id].get_all_parents()


def compute_performance(preds, labels, gos):
    # added for convenience, since we only evaluate the molecular function ontology
    GO_ID = 'GO:0003674'
    all_functions = GODag[GO_ID].get_all_children()

    ### stopped adding

    # preds = np.round(preds, decimals=2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100): # changed from xrange
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        # predictions = list()
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(preds.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            all_preds = set()
            for go_id in gos[i]: # gos seems to be a list of GO terms given for the specific target (from the ground truth files)
                if go_id in all_functions: # all_functions seems to be a set of GO terms that are in the current ontology
                    all_gos |= get_anchestors(go_id) # get ancestors originally seems to be returning all nodes above in the GOdag, incuding the one for which this is performed
            all_gos.discard(GO_ID) # GO_ID seems to be the root go
            # for val in preds[i]:
            #     go_id, score = val
            #     if score > threshold and go_id in all_functions:
            #         all_preds |= get_anchestors(go, go_id)
            # all_preds.discard(GO_ID)
            # predictions.append(all_preds)
            # tp = len(all_gos.intersection(all_preds))
            # fp = len(all_preds) - tp
            # fn = len(all_gos) - tp
            all_gos -= func_set
            fn += len(all_gos)

            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if total > 0 and p_total > 0:
            r /= total
            p /= p_total
            if p + r > 0:
                f = 2 * p * r / (p + r)
                if f_max < f:
                    f_max = f
                    p_max = p
                    r_max = r
                    t_max = threshold
                    predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
