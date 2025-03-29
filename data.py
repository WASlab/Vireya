import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from typing import Tuple

def get_dataset(name: str, data_dir: str = "./data", split_ratio: float = 0.9) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads and returns the training and validation datasets.

    Args:
        name (str): Dataset name ("cifar10" supported).
        data_dir (str): Where to download/store the data.
        split_ratio (float): Proportion of data to use for training.

    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """
    name = name.lower()

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        full_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )

        train_len = int(len(full_dataset) * split_ratio)
        val_len = len(full_dataset) - train_len
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
        return train_dataset, val_dataset

    else:
        raise NotImplementedError(f"Dataset '{name}' is not supported yet.")
