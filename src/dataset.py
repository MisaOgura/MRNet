import os
import torch
import numpy as np
import pandas as pd

from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

MEAN = 58.09
STDDEV = 49.73


class MRNetDataset(Dataset):
    def __init__(self, dataset_dir, labels_path, plane, transform=None, device=None):
        self.case_paths = sorted(glob(f'{dataset_dir}/{plane}/**.npy'))
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        self.window = 7
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        case_id = int(os.path.splitext(os.path.basename(case_path))[0])

        series = np.load(case_path).astype(np.uint8)
        series = list(np.stack((series,) * 3, axis=1))

        data = torch.tensor([])

        if self.transform is not None:
            for slice in series:
                slice = self.transform(slice.transpose(1, 2, 0))
                slice = slice.unsqueeze(0)
                data = torch.cat((data, slice), 0)

        case_row = self.labels_df[self.labels_df.case == case_id]
        diagnoses = case_row.values[0,1:].astype(np.float32)
        label = torch.tensor(diagnoses)

        return (data, label)


def make_dataset(data_dir, dataset_type, plane, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_dir = f'{data_dir}/{dataset_type}'
    labels_path = f'{data_dir}/{dataset_type}_labels.csv'
    transform = None

    if dataset_type == 'train':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(25, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    elif dataset_type == 'valid':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    else:
        raise ValueError('Dataset needs to be train or valid.')

    dataset = MRNetDataset(dataset_dir, labels_path, plane, transform=transform, device=device)

    return dataset
