import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_cifar10_splits(
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(val_fraction * len(dataset))
    train_size = len(dataset) - val_size

    g = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=g
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def make_celeba_splits(
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """CelebA train/val/test splits."""
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    full_train = datasets.CelebA(
        root=data_root,
        split="train",
        download=False,
        transform=transform,
    )
    test_dataset = datasets.CelebA(
        root=data_root,
        split="test",
        download=False,
        transform=transform,
    )

    g = torch.Generator().manual_seed(seed)
    val_size = int(val_fraction * len(full_train))
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=g
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
