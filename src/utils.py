import os
import csv

import numpy as np
import torch
from sklearn import metrics

MAX_PIXEL_VAL = 255
MEAN = 58.09
STD = 49.73


def preprocess_data(case_path, transform=None):
    series =np.load(case_path).astype(np.float32)
    series = torch.tensor(np.stack((series,)*3, axis=1))

    if transform is not None:
        for i, slice in enumerate(series.split(1)):
            series[i] = transform(slice.squeeze())

    series = (series - series.min()) / (series.max() - series.min()) * MAX_PIXEL_VAL
    series = (series - MEAN) / STD

    return series


def create_output_dir(exp, plane):
    out_dir = f'./models/{exp}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    losses_path = create_losses_csv(out_dir, plane)

    return out_dir, losses_path


def create_losses_csv(out_dir, plane):
    losses_path = f'{out_dir}/losses_{plane}.csv'

    with open(f'{losses_path}', mode='w') as losses_csv:
        fields = ['t_abnormal', 't_acl', 't_meniscus',
                    'v_abnormal', 'v_acl', 'v_meniscus']
        writer = csv.DictWriter(losses_csv, fieldnames=fields)
        writer.writeheader()

    return losses_path


def calculate_aucs(all_labels, all_preds):
    all_labels = np.array(all_labels).transpose()
    all_preds =  np.array(all_preds).transpose()

    aucs = [metrics.roc_auc_score(labels, preds) for \
            labels, preds in zip(all_labels, all_preds)]

    return aucs


def print_stats(batch_train_losses, batch_valid_losses,
                valid_labels, valid_preds):
    aucs = calculate_aucs(valid_labels, valid_preds)

    print(f'Train losses - abnormal: {batch_train_losses[0]:.3f},',
          f'acl: {batch_train_losses[1]:.3f},',
          f'meniscus: {batch_train_losses[2]:.3f}',
          f'\nValid losses - abnormal: {batch_valid_losses[0]:.3f},',
          f'acl: {batch_valid_losses[1]:.3f},',
          f'meniscus: {batch_valid_losses[2]:.3f}',
          f'\nValid AUCs - abnormal: {aucs[0]:.3f},',
          f'acl: {aucs[1]:.3f},',
          f'meniscus: {aucs[2]:.3f}')


def save_losses(train_losses , valid_losses, losses_path):
    with open(f'{losses_path}', mode='a') as losses_csv:
        writer = csv.writer(losses_csv)
        writer.writerow(np.append(train_losses, valid_losses))


def save_checkpoint(epoch, plane, diagnosis, model, optimizer, out_dir):
    print(f'Min valid loss for {diagnosis}, saving the checkpoint...')

    checkpoint = {
        'epoch': epoch,
        'plane': plane,
        'diagnosis': diagnosis,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    chkpt = f'cnn_{plane}_{diagnosis}_{epoch:02d}.pt'
    torch.save(checkpoint, f'{out_dir}/{chkpt}')
