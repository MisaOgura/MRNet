import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from mrnet.data_loader import make_data_loader
from mrnet.model import MRNet
from mrnet.utils import create_output_dir, print_losses, save_losses, save_checkpoint


def make_adam_optimizer(model, lr, weight_decay):
    optim_params = [
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': lr}
    ]

    return optim.Adam(optim_params, lr, weight_decay=weight_decay)


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


def batch_forward_backprop(models, inputs, labels, criterion, optimizers):
    prevalences = [0.806, 0.233, 0.371]
    losses = []

    for i, (model, label, prevalence, optimizer) in \
            enumerate(zip(models, labels[0], prevalences, optimizers)):
        model.train()
        optimizer.zero_grad()

        out = model(inputs)

        loss = criterion(out, label.unsqueeze(0))
        loss.mul_(1 - prevalence)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

    return np.array(losses)


def batch_forward(models, inputs, labels, criterion):
    prevalences = [0.806, 0.233, 0.371]
    losses = []

    for i, (model, label, prevalence) in \
            enumerate(zip(models, labels[0], prevalences)):
        model.eval()

        out = model(inputs)
        loss = criterion(out, label.unsqueeze(0))
        loss.mul_(1 - prevalence)
        losses.append(loss.item())

    return np.array(losses)


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

    model_abnormal = MRNet().to(device)
    model_acl = MRNet().to(device)
    model_meniscus = MRNet().to(device)
    models = [model_abnormal, model_acl, model_meniscus]

    criterion = nn.BCELoss()

    optimizers = [make_adam_optimizer(model_abnormal, lr, weight_decay),
                  make_adam_optimizer(model_acl, lr, weight_decay),
                  make_adam_optimizer(model_meniscus, lr, weight_decay)]

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
                                                criterion, optimizers)
            batch_train_losses += batch_loss

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_loss = batch_forward(models, inputs, labels, criterion)
            batch_valid_losses += batch_loss

        batch_train_losses /= len(train_loader)
        batch_valid_losses /= len(valid_loader)

        print_losses(batch_train_losses, batch_valid_losses)
        save_losses(batch_train_losses, batch_valid_losses, losses_path)

        update_lr_schedulers(lr_schedulers, batch_valid_losses)

        for i, (batch_v_loss, min_v_loss) in \
                enumerate(zip(batch_valid_losses, min_valid_losses)):

            if batch_v_loss < min_v_loss:
                save_checkpoint(epoch, plane, diagnoses[i], models[i],
                                optimizers[i], out_dir)

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
