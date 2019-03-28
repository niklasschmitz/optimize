import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms



def main(args):
    root = 'data'
    if not os.path.exists(root):
        os.mkdir(root)

    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False)