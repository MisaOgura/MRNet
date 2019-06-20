import torch
import numpy as np
import pandas as pd

from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class MRNetDataset(Dataset):
    planes = ['axial', 'coronal', 'sagittal']

    def __init__(self, dataset_dir, labels_path, plane, transform=None, device=None):
        self.case_paths = sorted(glob(f'{dataset_dir}/**'))
        self.labels_df = pd.read_csv(labels_path)
        self.plane = plane
        self.transform = transform
        self.window = 7

        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        series = sorted(glob(f'{case_path}/{self.plane}/*.png'))

        # Get indecies of middle 15 slices

        mid_idx = len(series) // 2
        from_idx = mid_idx - self.window
        to_idx = mid_idx + self.window + 1
        image_paths = series[from_idx:to_idx]

        data = torch.tensor([]).to(self.device)

        for path in image_paths:
            image = Image.open(path)
            if self.transform is not None:
                image = self.transform(image).unsqueeze(0).to(self.device)

            data = torch.cat((data, image), 0)

        diagnoses = self.labels_df.iloc[idx].values[1:].astype(np.float32)
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
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(25, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])
    elif dataset_type == 'valid':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        raise ValueError('Dataset needs to be train or valid.')

    dataset = MRNetDataset(dataset_dir, labels_path, plane, transform=transform, device=device)

    return dataset

