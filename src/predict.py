#!/usr/bin/env python3.6
"""Calculates predictions on the validation dataset, using CNN models specified
in src/cnn_models_paths.txt and logistic regression models specified in
src/lr_models_paths.txt.

Usage:
  predict.py <valid_paths_csv> <output_dir>
  predict.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <valid_paths_csv>  csv file listing paths to validation set, which needs to
                     be in a specific order - an example is provided as
                     valid-paths.csv in the root of the project
                     e.g. 'valid-paths.csv'
  <output_dir>       Directory where predictions are saved as a 3-column csv
                     file (with no header), where each column contains a
                     prediction for abnormality, ACL tear, and meniscal tear,
                     in that order
                     e.g. 'out_dir'
"""

import os
import sys
import csv
from PIL import Image
from tqdm import tqdm
from docopt import docopt

import torch
import numpy as np
import pandas as pd
import joblib
from torchvision import transforms

from model import MRNet
from utils import preprocess_data


def main(valid_paths_csv, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_files_df = pd.read_csv(valid_paths_csv, header=None)
    cnn_models_paths = 'src/cnn_models_paths.txt'
    lr_models_paths = 'src/lr_models_paths.txt'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f'{output_dir}/predictions.csv'

    if os.path.exists(output_file):
        os.rename(output_file, f'{output_file}.back')
        print(f'!! {output_file} already exists, renamed to {output_file}.bak')

    # Load MRNet models
    print(f'Loading CNN models listed in {cnn_models_paths}...')

    cnn_models_paths = [line.rstrip('\n') for line in open(cnn_models_paths, 'r')]

    abnormal_mrnets =[]
    acl_mrnets = []
    meniscus_mrnets = []

    for i, mrnet_path in enumerate(cnn_models_paths):
        model = MRNet().to(device)
        checkpoint = torch.load(mrnet_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

        if i < 3:
            abnormal_mrnets.append(model)
        elif i >= 3 and i < 6:
            acl_mrnets.append(model)
        else:
            meniscus_mrnets.append(model)

    mrnets = [abnormal_mrnets, acl_mrnets, meniscus_mrnets]

    # Load logistic regression models
    print(f'Loading logistic regression models listed in {lr_models_paths}...')

    lr_models_paths = [line.rstrip('\n') for line in open(lr_models_paths, 'r')]
    lrs = [joblib.load(lr_path) for lr_path in lr_models_paths]

    # Parse input, 3 rows at a time (i.e. per case)

    npy_paths = [row.values[0] for _, row in input_files_df.iterrows()]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    print(f'Generating predictions per case...')
    print(f'Predictions will be saved as {output_file}')

    for i in tqdm(range(0, len(npy_paths), 3)):
        case_paths = [npy_paths[i], npy_paths[i+1], npy_paths[i+2]]

        data = []

        for case_path in case_paths:
            series = preprocess_data(case_path, transform)
            data.append(series.unsqueeze(0).to(device))

        # Make predictions per case

        case_preds = []

        for i, mrnet in enumerate(mrnets):  # For each condition (mrnet)
            # Based on each plane (data)
            sagittal_pred = mrnet[0](data[0]).detach().cpu().item()
            coronal_pred = mrnet[1](data[1]).detach().cpu().item()
            axial_pred = mrnet[2](data[2]).detach().cpu().item()

            # Combine predictions to make a final prediction

            X = [[axial_pred, coronal_pred, sagittal_pred]]
            case_preds.append(np.float64(lrs[i].predict_proba(X)[:,1]))

        # Write to output csv - append if it exists already

        with open(output_file, 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(case_preds)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<valid_paths_csv>'],
         arguments['<output_dir>'])
