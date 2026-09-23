"""
Dataset loading. Uses torchvision's MNIST / FashionMNIST.
Images are normalized to [-1, 1] (standard for diffusion models).
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASS_NAMES = {
    'mnist': [str(i) for i in range(10)],
    'fashion_mnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
}


def get_dataloaders(dataset='mnist', batch_size=128, data_dir='./data', num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0,1]
        transforms.Normalize((0.5,), (0.5,)),         # -> [-1,1]
    ])
    if dataset == 'mnist':
        cls = datasets.MNIST
    elif dataset == 'fashion_mnist':
        cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'mnist' or 'fashion_mnist'.")

    train_set = cls(root=data_dir, train=True, download=True, transform=transform)
    test_set = cls(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, test_loader, CLASS_NAMES[dataset]
