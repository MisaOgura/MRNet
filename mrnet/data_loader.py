from torch.utils.data import DataLoader

from dataset import make_datasets


def make_data_loaders(data_dir, plane, diagnosis, batch_size, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, valid_dataset = make_datasets(data_dir, plane, diagnosis)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, device=device)
    valid_loader = DataLoader(valid_dataset, batch_size, device=device)

    return train_loader, valid_loader
