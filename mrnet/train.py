import os
import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import make_data_loader
from model import MRNet


def forward_backprop(model, inputs, label, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(inputs)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()

    return loss.item()


def batch_forward_backprop(models, inputs, labels, criterion, optimizers):
    losses = []

    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        loss = forward_backprop(model, inputs, labels[:,i],
                                criterion, optimizer)
        losses.append(loss)

    return np.array(losses)


def forward(model, inputs, label, criterion):
    model.eval()

    out = model(inputs)
    loss = criterion(out, label)

    return loss.item()


def batch_forward(models, inputs, labels, criterion):
    losses = []

    for i, (model) in enumerate(models):
        loss = forward(model, inputs, labels[:,i], criterion)
        losses.append(loss)

    return np.array(losses)


def calculate_total_loss(abnormal_loss, acl_loss, meniscus_loss):
    abnormal_loss = abnormal_loss * (1.0 - 0.806)
    acl_loss = acl_loss * (1.0 - 0.233)
    meniscus_loss = meniscus_loss * (1.0 - 0.371)

    loss = abnormal_loss + acl_loss + meniscus_loss

    return loss


def make_adam_optimizer(model, lr, weight_decay):
    optim_params = [
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': lr}
    ]

    return optim.Adam(optim_params, lr, weight_decay=weight_decay)


def print_losses(batch_train_losses, batch_valid_losses):
    print(f'Train losses - abnormal: {batch_train_losses[0]:.3f},',
          f'acl: {batch_train_losses[1]:.3f},',
          f'meniscus: {batch_train_losses[2]:.3f}',
          f'\nValid losses - abnormal: {batch_valid_losses[0]:.3f},',
          f'acl: {batch_valid_losses[1]:.3f},',
          f'meniscus: {batch_valid_losses[2]:.3f}')


def save_checkpoint(epoch, plane, diagnosis, model, optimizer, now, chkpt_dir):
    print(f'Min valid loss detected for {diagnosis},',
          f'saving the model in {chkpt_dir}/...')

    checkpoint = {
        'state_dicts': model.state_dict(),
        'optimizers': optimizer.state_dict()
    }

    chkpt = f'mrnet_p-{plane}_d-{diagnosis}_e-{epoch:02d}.pt'

    torch.save(checkpoint, f'{chkpt_dir}/{chkpt}')


def main(data_dir, plane, epochs, lr, weight_decay, device=None):
    diagnoses = ['abnormal', 'acl', 'meniscus']

    now = datetime.now()
    now = f'{now:%Y-%m-%d_%H-%M}'

    chkpt_dir = f'./models/{now}'
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating data loaders...')

    train_loader = make_data_loader(data_dir, 'train', plane, device, shuffle=True)
    valid_loader = make_data_loader(data_dir, 'valid', plane, device)

    print('Creating models...')

    model_abnormal = MRNet().to(device)
    model_acl = MRNet().to(device)
    model_meniscus = MRNet().to(device)
    models = [model_abnormal, model_acl, model_meniscus]

    criterion = nn.BCELoss()

    optimizers = [make_adam_optimizer(model_abnormal, lr, weight_decay),
                  make_adam_optimizer(model_acl, lr, weight_decay),
                  make_adam_optimizer(model_meniscus, lr, weight_decay)]

    train_losses = []
    valid_losses = []
    min_valid_losses = [np.inf, np.inf, np.inf]

    print(f'Training a model using {plane} series...')

    for epoch, _ in enumerate(range(epochs), 1):
        batch_train_losses = np.array([0.0, 0.0, 0.0])
        batch_valid_losses = np.array([0.0, 0.0, 0.0])

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_loss = batch_forward_backprop(models, inputs, labels,
                                                criterion, optimizers)
            batch_train_losses += batch_loss

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_loss = batch_forward(models, inputs, labels, criterion)
            batch_valid_losses += batch_loss

        batch_train_losses /= len(train_loader)
        batch_valid_losses /= len(valid_loader)

        train_losses.append(batch_train_losses)
        valid_losses.append(batch_valid_losses)

        print(f'=== Epoch {epoch}/{epochs} ===')
        print_losses(batch_train_losses, batch_valid_losses)

        for i, (batch_v_loss, min_v_loss) in \
                enumerate(zip(batch_valid_losses, min_valid_losses)):

            if batch_v_loss < min_v_loss:
                save_checkpoint(epoch, plane, diagnoses[i], models[i],
                                optimizers[i], now, chkpt_dir)

                min_valid_losses[i] = batch_v_loss


if __name__ == '__main__':
    print('Parsing arguments...')
    data_dir = sys.argv[1]
    plane = sys.argv[2]
    epochs = int(sys.argv[3])
    lr = float(sys.argv[4])
    weight_decay = float(sys.argv[5])

    try:
        device = sys.argv[6]
    except IndexError:
        device = None

    main(data_dir, plane, epochs, lr, weight_decay, device)
