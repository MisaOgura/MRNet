#!/usr/bin/python3

import sys
import csv
from PIL import Image

import torch
import numpy as np
import pandas as pd
import joblib
from torchvision import transforms

from model import MRNet


def main(paths_csv, output_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_files_df = pd.read_csv(paths_csv, header=None)
    mrnet_paths = 'src/mrnet_paths.txt'
    lr_paths = 'src/lr_paths.txt'

    # Load MRNet models

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

    lr_paths = [line.rstrip('\n') for line in open(lr_paths, 'r')]
    lrs = [joblib.load(lr_path) for lr_path in lr_paths]

    # Parse input, 3 rows at a time (i.e. per case)

    npy_paths = [row.values[0] for _, row in input_files_df.iterrows()]

    transform = transforms.ToTensor()

    preds = []

    for i in range(0, len(npy_paths), 3):
        case_paths = [npy_paths[i], npy_paths[i+1], npy_paths[i+2]]

        data = []

        # Convert npy to rgb, then to torch tensor

        for series_path in case_paths:
            series_npy = np.load(series_path)
            series = torch.tensor([]).to(device)

            for i, image in enumerate(series_npy):
                rgb = Image.fromarray(image, 'L').convert('RGB')
                tensor = transform(rgb).unsqueeze(0).to(device)
                series = torch.cat((series, tensor), 0)

            data.append(series.unsqueeze(0))

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

        with open(output_path, 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(case_preds)


if __name__ == '__main__':
    paths_csv = sys.argv[1]
    output_path = sys.argv[2]

    main(paths_csv, output_path)
