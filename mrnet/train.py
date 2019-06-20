import os
import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import make_data_loader
from model import MRNet


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


def main(data_dir, plane, epochs, batch_size, lr, weight_decay, device=None):
    now = datetime.now()
    now = f'{now:%Y-%m-%d_%H:%M:%S}'

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

    criterion = nn.BCELoss()

    optimizers = [
        optim.Adam(model_abnormal.parameters(), lr, weight_decay=weight_decay),
        optim.Adam(model_acl.parameters(), lr, weight_decay=weight_decay),
        optim.Adam(model_meniscus.parameters(), lr, weight_decay=weight_decay)
    ]

    train_losses = []
    valid_losses = []
    min_valid_loss = np.inf

    print('Starting the training...')

    for epoch in range(epochs):
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

            loss = train_abnormal_loss + train_acl_loss + train_meniscus_loss
            train_loss += loss

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            valid_abnormal_loss = forward(model_abnormal, inputs, labels[:,0], criterion)
            valid_acl_loss = forward(model_acl, inputs, labels[:,1], criterion)
            valid_meniscus_loss = forward(model_meniscus, inputs, labels[:,2], criterion)

            loss = valid_abnormal_loss + valid_acl_loss + valid_meniscus_loss
            valid_loss += loss

        train_loss = train_loss/len(train_loader)
        train_losses.append(train_loss)

        valid_loss = valid_loss/len(valid_loader)
        valid_losses.append(valid_loss)

        print(f'Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}')

        if valid_loss < min_valid_loss:
            print(f'\tValidation loss decreased {min_valid_loss:.3f} --> {valid_loss:.3f}.')

            # checkpoint = {
            #     'epoch': epoch + 1,
            #     'state_dict': model_abnormal.state_dict(),
            #     'optimizer': optimizer.state_dict()
            # }

            # if not os.path.exists('./models'):
            #     os.mkdir('./models')

            # torch.save(checkpoint, f'./models/mrnet_{now}.pt')

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
