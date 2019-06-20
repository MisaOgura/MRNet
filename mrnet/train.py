import os
import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import make_data_loader
from model import MRNet, yield_unfrozen_params


def forward_and_backprop(model, inputs, labels, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(inputs)
    loss = criterion(out, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def forward(model, inputs, labels, criterion):
    model.eval()

    out = model(inputs)
    loss = criterion(out, labels)

    return loss.item()


def calculate_total_loss(abnormal_loss, acl_loss, meniscus_loss):
    abnormal_loss = abnormal_loss * (1.0 - 0.806)
    acl_loss = acl_loss * (1.0 - 0.233)
    meniscus_loss = meniscus_loss * (1.0 - 0.371)

    loss = abnormal_loss + acl_loss + meniscus_loss

    return loss


def make_adam_optimizer(model, lr, weight_decay):
    optim_params = [
        {'params': yield_unfrozen_params(model.features), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': lr}
    ]

    return optim.Adam(optim_params, lr, weight_decay=weight_decay)


def save_checkpoint(epoch, models, optimizers, plane, now, chkpt_dir):
    checkpoint = {
        'state_dicts': [model.state_dict() for model in models],
        'optimizers': [optimizer.state_dict() for optimizer in optimizers]
    }

    torch.save(checkpoint, f'{chkpt_dir}/mrnet_p-{plane}_e-{epoch}.pt')


def main(data_dir, plane, epochs, batch_size, lr, weight_decay, device=None):
    now = datetime.now()
    now = f'{now:%Y-%m-%d_%H-%M}'

    chkpt_dir = f'./models/{now}'
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating data loaders...')

    train_loader = make_data_loader(data_dir, 'train', plane,
                                    batch_size, device, shuffle=True)

    valid_loader = make_data_loader(data_dir, 'valid', plane,
                                    batch_size, device)

    print('Creating models...')

    model_abnormal = MRNet().to(device)
    model_acl = MRNet().to(device)
    model_meniscus = MRNet().to(device)

    models = [model_abnormal, model_acl, model_meniscus]

    criterion = nn.BCELoss()

    optimizers = [
        make_adam_optimizer(model_abnormal, lr, weight_decay),
        make_adam_optimizer(model_acl, lr, weight_decay),
        make_adam_optimizer(model_meniscus, lr, weight_decay),
    ]

    train_losses = []
    valid_losses = []
    min_valid_loss = np.inf

    print('Starting the training...')

    for epoch, _ in enumerate(range(epochs), 1):
        train_loss = 0.0
        valid_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            train_abnormal_loss = forward_and_backprop(model_abnormal,
                                                       inputs,
                                                       labels[:,0],
                                                       criterion,
                                                       optimizers[0])

            train_acl_loss = forward_and_backprop(model_acl,
                                                  inputs,
                                                  labels[:,1],
                                                  criterion,
                                                  optimizers[1])

            train_meniscus_loss = forward_and_backprop(model_meniscus,
                                                       inputs,
                                                       labels[:,2],
                                                       criterion,
                                                       optimizers[2])

            # TODO - scale losses inversely proportionally to the
            # prevelence of the corresponding conditions

            loss = calculate_total_loss(train_abnormal_loss,
                                        train_acl_loss,
                                        train_meniscus_loss)
            train_loss += loss

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            valid_abnormal_loss = forward(model_abnormal, inputs, labels[:,0], criterion)
            valid_acl_loss = forward(model_acl, inputs, labels[:,1], criterion)
            valid_meniscus_loss = forward(model_meniscus, inputs, labels[:,2], criterion)

            loss = calculate_total_loss(valid_abnormal_loss,
                                        valid_acl_loss,
                                        valid_meniscus_loss)
            valid_loss += loss

        train_loss = train_loss/len(train_loader)
        train_losses.append(train_loss)

        valid_loss = valid_loss/len(valid_loader)
        valid_losses.append(valid_loss)

        print(f'Epoch {epoch}/{epochs} -',
              f'train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}')

        if valid_loss < min_valid_loss:
            print(f'***Validation loss decreased',
                  f'{min_valid_loss:.3f} --> {valid_loss:.3f}',
                  f'model saved to {chkpt_dir}')
            save_checkpoint(epoch, models, optimizers, plane, now, chkpt_dir)
            min_valid_loss = valid_loss


if __name__ == '__main__':
    print('Parsing arguments...')
    data_dir = sys.argv[1]
    plane = sys.argv[2]
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    lr = float(sys.argv[5])
    weight_decay = float(sys.argv[6])

    try:
        device = sys.argv[7]
    except IndexError:
        device = None

    main(data_dir, plane, epochs, batch_size, lr, weight_decay, device)
