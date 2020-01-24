import torch
import torchvision.datasets


def load_data():
    """
    Download CIFAR10 dataset in the right shape
        
    Returns
    -------
    X_train : torch.Tensor of shape (n_train_samples, 3, 32, 32)
    y_train : torch.Tensor of shape (n_train_samples)
    X_test : torch.Tensor of shape (n_test_samples, 3, 32, 32)
    y_test : torch.Tensor of shape (n_test_samples)
    """
    CIFAR_train = torchvision.datasets.CIFAR10('./', download=True, train=True)
    CIFAR_test = torchvision.datasets.CIFAR10('./', download=True, train=False)

    print('CIFAR classes:')
    print(CIFAR_train.classes)

    X_train = torch.FloatTensor(CIFAR_train.data)
    y_train = torch.LongTensor(CIFAR_train.targets)
    X_test = torch.FloatTensor(CIFAR_test.data)
    y_test = torch.LongTensor(CIFAR_test.targets)

    # Normalize data to be in [0, 1]
    X_train /= 255.
    X_test /= 255.

    # Make shapes of tensors (n, 3, 32, 32)
    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)

    return X_train, y_train, X_test, y_test