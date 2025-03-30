import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


def get_dataset(
    name: str,
    data_dir: str = "./data",
    split_ratio: float = 0.9
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads and returns the training and validation datasets with on-the-fly augmentations.

    Args:
        name (str): Dataset name ("cifar10", "mnist", or "kmnist").
        data_dir (str): Where to download/store the data.
        split_ratio (float): Proportion of data to use for training.

    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """
    name = name.lower()

    if name == "cifar10":
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        augmentation_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

        class CIFAR10(datasets.CIFAR10):
            def __getitem__(self, index):
                img, target = self.data[index], self.targets[index]
                img = transforms.functional.to_pil_image(img)
                if self.train:
                    img = augmentation_transform(img)
                img = base_transform(img)
                return img, target

        dataset = CIFAR10(root=data_dir, train=True, download=True)

    elif name in ["mnist", "kmnist"]:
        base_dataset = datasets.MNIST if name == "mnist" else datasets.KMNIST
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = base_dataset(root=data_dir, train=True, download=True, transform=transform)

    else:
        raise NotImplementedError(f"Dataset '{name}' is not supported.")

    train_len = int(len(dataset) * split_ratio)
    val_len = len(dataset) - train_len
    return random_split(dataset, [train_len, val_len])


def get_dataloader(
    train_dataset,
    val_dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int = 8,
    prefetch_factor: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns distributed data loaders for training and validation.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Batch size per GPU.
        world_size: Total number of distributed processes.
        rank: Rank of the current process.
        num_workers: Data loading worker threads.
        prefetch_factor: Number of batches to prefetch per worker.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and validation.
    """
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader
