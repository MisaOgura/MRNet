import os
import sys
import csv

import pandas as pd
import numpy as np

from sklearn import metrics


def main(cases_path, preds_path, valid_labels_path):
    print('Reporting AUC scores...')

    preds_df = pd.read_csv(preds_path, header=None)
    valid_df = pd.read_csv(valid_labels_path)

    old_case = None

    cases = []
    with open(cases_path, 'r') as paths:
        for path in paths:
            case = os.path.splitext(os.path.basename(path.strip()))[0]
            if case == old_case:
                next
            else:
                cases.append(case)
                old_case = case

    ys = []
    Xs = []

    for i, case in enumerate(cases):
        case_row = valid_df[valid_df.case == int(case)]

        y = case_row.values[0,1:].astype(np.float32)
        ys.append(y)

        X = preds_df.iloc[i].values
        Xs.append(X)

    ys = np.asarray(ys).transpose()
    Xs = np.asarray(Xs).transpose()

    aucs = {}

    diagnoses = valid_df.columns.values[1:]

    for i,diagnosis in enumerate(diagnoses):
        auc = metrics.roc_auc_score(ys[i], Xs[i])

        aucs[diagnosis] = auc

    aucs['mean'] = np.array(list(aucs.values())).mean()

    for k, v in aucs.items():
        print(f'\t{k}: {v:.3f}')


if __name__ == '__main__':
    cases_path = sys.argv[1]
    preds_path = sys.argv[2]
    valid_labels_path = sys.argv[3]

    main(cases_path, preds_path, valid_labels_path)
