import torch
from torch.utils.data import DataLoader

from dataset import make_dataset


def make_data_loader(data_dir, dataset_type, plane, device=None, shuffle=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = make_dataset(data_dir, dataset_type, plane, device=device)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    return data_loader
