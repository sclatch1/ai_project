import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        xi = self.transform(sample)
        xj = self.transform(sample)
        return (xi, xj), label

    def __len__(self):
        return len(self.dataset)


def get_contrastive_dataloaders(data_dir, batch_size, num_workers, transform):
    base_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    contrastive_dataset = ContrastiveDataset(base_dataset, transform)

    loader = DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader
