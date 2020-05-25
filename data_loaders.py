import torch
from torchvision import transforms, datasets

def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    channels = 3
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset

def get_MNIST(root="./"):
    input_size = 28
    channels = 1
    num_classes = 10
    fmnist = datasets.MNIST(root + "data/MNIST", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    data_loader = torch.utils.data.DataLoader(fmnist,
                                          batch_size=10,
                                          shuffle=True)
    ftest = datasets.MNIST(root + "data/MNIST", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    return input_size, num_classes, fmnist, ftest

def get_FMNIST(root="./"):
    input_size = 28
    channels = 1
    num_classes = 10
    fmnist = datasets.FashionMNIST(root + "data/FMNIST", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    data_loader = torch.utils.data.DataLoader(fmnist,
                                          batch_size=10,
                                          shuffle=True)
    ftest = datasets.FashionMNIST(root + "data/FMNIST", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    return input_size, num_classes, fmnist, ftest

def get_synthetic_data(latent, d, n):
  x = np.random.rand(n, latent)
  
  # random projection
  px = np.matmul(x, np.random.rand(latent, d))

  # sin/cosine transformation
  fx = px#np.concatenate([np.cos(px), np.sin(px)])
  # y = w^T phi(x)
  y = np.matmul(fx, np.random.rand(d))
  return x, y