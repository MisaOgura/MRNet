#!/usr/bin/env python
"""Trains three CNN models to predict abnormalities, ACL tears and meniscal
tears for a given plane (axial, coronal or sagittal) of knee MRI images.

Usage:
  train_cnn_models.py <data_dir> <plane> <epochs> [options]
  train_cnn_models.py (-h | --help)

General options:
  -h --help             Show this screen.

Arguments:
  <data_dir>            Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <plane>               MRI plane of choice ('axial', 'coronal', 'sagittal')
  <epochs>              Number of epochs e.g. 50

Training options:
  --lr=<lr>             Learning rate for nn.optim.Adam optimizer [default: 0.00001]
  --weight-decay=<wd>   Weight decay for nn.optim.Adam optimizer [default: 0.01]
  --device=<device>     Device to run code ('cpu' or 'cuda') - if not provided,
                        it will be set to the value returned by torch.cuda.is_available()
"""

import sys
import numpy as np
import pandas as pd
from docopt import docopt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import make_data_loader
from model import MRNet
from utils import create_output_dir, \
                  print_stats,       \
                  save_losses,       \
                  save_checkpoint


def calculate_weights(data_dir, dataset_type, device):
    diagnoses = ['abnormal', 'acl', 'meniscus']

    labels_path = f'{data_dir}/{dataset_type}_labels.csv'
    labels_df = pd.read_csv(labels_path)

    weights = []

    for diagnosis in diagnoses:
        neg_count, pos_count = labels_df[diagnosis].value_counts().sort_index()
        weight = torch.tensor([neg_count / pos_count])
        weight = weight.to(device)
        weights.append(weight)

    return weights


def make_adam_optimizer(model, lr, weight_decay):
    return optim.Adam(model.parameters(), lr, weight_decay=weight_decay)


def make_lr_scheduler(optimizer,
                      mode='min',
                      factor=0.3,
                      patience=1,
                      verbose=False):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                mode=mode,
                                                factor=factor,
                                                patience=patience,
                                                verbose=verbose)


def batch_forward_backprop(models, inputs, labels, criterions, optimizers):
    losses = []

    for i, (model, label, criterion, optimizer) in \
            enumerate(zip(models, labels[0], criterions, optimizers)):
        model.train()
        optimizer.zero_grad()

        out = model(inputs)
        loss = criterion(out, label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.array(losses)


def batch_forward(models, inputs, labels, criterions):
    preds = []
    losses = []

    for i, (model, label, criterion) in \
            enumerate(zip(models, labels[0], criterions)):
        model.eval()

        out = model(inputs)
        preds.append(out.item())
        loss = criterion(out, label.unsqueeze(0))
        losses.append(loss.item())

    return np.array(preds), np.array(losses)


def update_lr_schedulers(lr_schedulers, batch_valid_losses):
    for scheduler, v_loss in zip(lr_schedulers, batch_valid_losses):
        scheduler.step(v_loss)


def main(data_dir, plane, epochs, lr, weight_decay, device=None):
    diagnoses = ['abnormal', 'acl', 'meniscus']

    exp = f'{datetime.now():%Y-%m-%d_%H-%M}'
    out_dir, losses_path = create_output_dir(exp, plane)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating data loaders...')

    train_loader = make_data_loader(data_dir, 'train', plane, device, shuffle=True)
    valid_loader = make_data_loader(data_dir, 'valid', plane, device)

    print(f'Creating models...')

    # Create a model for each diagnosis

    models = [MRNet().to(device), MRNet().to(device), MRNet().to(device)]

    # Calculate loss weights based on the prevalences in train set

    pos_weights = calculate_weights(data_dir, 'train', device)
    criterions = [nn.BCEWithLogitsLoss(pos_weight=weight) \
                  for weight in pos_weights]

    optimizers = [make_adam_optimizer(model, lr, weight_decay) \
                  for model in models]

    lr_schedulers = [make_lr_scheduler(optimizer) for optimizer in optimizers]

    min_valid_losses = [np.inf, np.inf, np.inf]

    print(f'Training a model using {plane} series...')
    print(f'Checkpoints and losses will be save to {out_dir}')

    for epoch, _ in enumerate(range(epochs), 1):
        print(f'=== Epoch {epoch}/{epochs} ===')

        batch_train_losses = np.array([0.0, 0.0, 0.0])
        batch_valid_losses = np.array([0.0, 0.0, 0.0])

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_loss = batch_forward_backprop(models, inputs, labels,
                                                criterions, optimizers)
            batch_train_losses += batch_loss

        valid_preds = []
        valid_labels = []

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_preds, batch_loss = \
                batch_forward(models, inputs, labels, criterions)
            batch_valid_losses += batch_loss

            valid_labels.append(labels.detach().cpu().numpy().squeeze())
            valid_preds.append(batch_preds)

        batch_train_losses /= len(train_loader)
        batch_valid_losses /= len(valid_loader)

        print_stats(batch_train_losses, batch_valid_losses,
                    valid_labels, valid_preds)
        save_losses(batch_train_losses, batch_valid_losses, losses_path)

        update_lr_schedulers(lr_schedulers, batch_valid_losses)

        for i, (batch_v_loss, min_v_loss) in \
                enumerate(zip(batch_valid_losses, min_valid_losses)):

            if batch_v_loss < min_v_loss:
                save_checkpoint(epoch, plane, diagnoses[i], models[i],
                                optimizers[i], out_dir)

                min_valid_losses[i] = batch_v_loss


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_dir>'],
         arguments['<plane>'],
         int(arguments['<epochs>']),
         float(arguments['--lr']),
         float(arguments['--weight-decay']),
         arguments['--device'])
