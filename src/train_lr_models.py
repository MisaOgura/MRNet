#!/usr/bin/env python3.6
"""Trains logistic regression models for abnormalities, ACL tears and meniscal
tears, by combine predictions from CNN models.

Usage:
  train_lr_models.py <data_dir> <models_dir>
  train_lr_models.py (-h | --help)

General options:
  -h --help         Show this screen.

Arguments:
  <data_dir>        Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <models_dir>      Directory where CNN models are saved e.g. 'models/2019-06-24_04-18'
"""

import sys
from glob import glob
from tqdm import tqdm
from docopt import docopt

import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import joblib

from model import MRNet
from data_loader import make_data_loader


def main(data_dir, models_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planes = ['axial', 'coronal', 'sagittal']
    conditions = ['abnormal', 'acl', 'meniscus']

    models = []

    print(f'Loading best CNN models from {models_dir}...')

    for condition in conditions:
        models_per_condition = []
        for plane in planes:
            checkpoint_pattern = glob(f'{models_dir}/*{plane}*{condition}*.pt')
            checkpoint_path = sorted(checkpoint_pattern)[-1]
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model = MRNet().to(device)
            model.load_state_dict(checkpoint['state_dict'])
            models_per_condition.append(model)

        models.append(models_per_condition)

    print(f'Creating data loaders...')

    axial_loader = make_data_loader(data_dir, 'train', 'axial')
    coronal_loader = make_data_loader(data_dir, 'train', 'coronal')
    sagittal_loader = make_data_loader(data_dir, 'train', 'sagittal')

    print(f'Collecting predictions on train dataset from the models...')

    ys = []
    Xs = [[],[],[]]  # Abnormal, ACL, Meniscus

    with tqdm(total=len(axial_loader)) as pbar:
        for (axial_inputs, labels), (coronal_inputs, _), (sagittal_inputs, _) in \
                zip(axial_loader, coronal_loader, sagittal_loader):

            axial_inputs, coronal_inputs, sagittal_inputs = \
                axial_inputs.to(device), coronal_inputs.to(device), sagittal_inputs.to(device)

            ys.append(labels[0].cpu().tolist())

            for i, model in enumerate(models):
                axial_pred = model[0](axial_inputs).detach().cpu().item()
                coronal_pred = model[1](coronal_inputs).detach().cpu().item()
                sagittal_pred = model[2](sagittal_inputs).detach().cpu().item()

                X = [axial_pred, coronal_pred, sagittal_pred]
                Xs[i].append(X)

            pbar.update(1)

    ys = np.asarray(ys).transpose()
    Xs = np.asarray(Xs)

    print(f'Training logistic regression models for each condition...')

    clfs = []

    for X, y in zip(Xs, ys):
        clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
        clfs.append(clf)

    for i, clf in enumerate(clfs):
        print(f'Cross validation score for {conditions[i]}: {clf.score(X, y):.3f}')
        clf_path = f'{models_dir}/lr_{conditions[i]}.pkl'
        joblib.dump(clf, clf_path)

    print(f'Logistic regression models saved to {models_dir}')


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_dir>'],
         arguments['<models_dir>'])
