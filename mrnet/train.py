import os
import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import make_data_loaders
from model import MRNet


def main(data_dir, plane, diagnosis, epochs, batch_size, lr, weight_decay, device=None):
    now = datetime.now()
    now = f'{now:%Y-%m-%d_%H:%M:%S}'

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating data loaders...')

    train_loader, valid_loader = make_data_loaders(data_dir,
                                                   plane,
                                                   diagnosis,
                                                   batch_size,
                                                   device)

    print('Creating a model...')

    model = MRNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    train_losses = []
    valid_losses = []
    min_valid_loss = np.inf

    print('Starting the training...')

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        for inputs, labels in train_loader:
            model.train()

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            out = model(inputs)
            loss = criterion(out.unsqueeze(1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        for inputs, labels in valid_loader:
            model.eval()

            inputs, labels = inputs.to(device), labels.to(device)

            out = model(inputs)
            loss = criterion(out.unsqueeze(1), labels)
            valid_loss += loss.item()

        train_loss = train_loss/len(train_loader)
        train_losses.append(train_loss)

        valid_loss = valid_loss/len(train_loader)
        valid_losses.append(valid_loss)

        print(f'Epoch {epoch + 1}/{epochs}: train loss - {train_loss:.3f}, valid loss - {valid_loss:.3f}')

        if valid_loss < min_valid_loss:
            print(f'\tValidation loss decreased {min_valid_loss:.3f} --> {valid_loss:.3f}.')

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            if not os.path.exists('../models'):
                os.mkdir('../models')

            torch.save(checkpoint, f'../models/model_flower_{now}.pt')

            min_valid_loss = valid_loss


if __name__ == '__main__':
    print('Parsing arguments...')
    data_dir = sys.argv[1]
    plane = sys.argv[2]
    diagnosis = sys.argv[3]
    epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    lr = float(sys.argv[6])
    weight_decay = float(sys.argv[7])

    main(data_dir, plane, diagnosis, epochs, batch_size, lr, weight_decay)
