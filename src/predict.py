#!/usr/bin/python3

import os
import sys
import csv
from PIL import Image
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import joblib
from torchvision import transforms

from model import MRNet

MAX_PIXEL_VAL = 255
MEAN = 58.09
STD = 49.73


def main(paths_csv, output_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_files_df = pd.read_csv(paths_csv, header=None)
    mrnet_paths = 'src/mrnet_paths.txt'
    lr_paths = 'src/lr_paths.txt'

    output_file = f'{output_dir}/predictions.csv'

    if os.path.exists(output_file):
        os.rename(output_file, f'{output_file}.back')
        print(f'***{output_file} already exists, renamed to {output_file}.bak')

    # Load MRNet models
    print(f'Loading CNN models listed in {mrnet_paths}...')

    mrnet_paths = [line.rstrip('\n') for line in open(mrnet_paths, 'r')]

    abnormal_mrnets =[]
    acl_mrnets = []
    meniscus_mrnets = []

    for i, mrnet_path in enumerate(mrnet_paths):
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
    print(f'Loading logistic regression models listed in {lr_paths}...')

    lr_paths = [line.rstrip('\n') for line in open(lr_paths, 'r')]
    lrs = [joblib.load(lr_path) for lr_path in lr_paths]

    # Parse input, 3 rows at a time (i.e. per case)

    npy_paths = [row.values[0] for _, row in input_files_df.iterrows()]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    print(f'Generating predictions per case...')
    print(f'Predictions will be saved to {output_file}')

    for i in tqdm(range(0, len(npy_paths), 3)):
        case_paths = [npy_paths[i], npy_paths[i+1], npy_paths[i+2]]

        # Convert npy series to rgb, then to torch tensor

        data = []

        for case_path in case_paths:
            series = np.load(case_path).astype(np.float32)
            series = torch.tensor(np.stack((series,)*3, axis=1))

            for i, slice in enumerate(series.split(1)):
                series[i] = transform(slice.squeeze())

            series = (series - series.min()) / (series.max() - series.min()) * MAX_PIXEL_VAL
            series = (series - MEAN) / STD

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
    paths_csv = sys.argv[1]
    output_dir = sys.argv[2]

    main(paths_csv, output_dir)
