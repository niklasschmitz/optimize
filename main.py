import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import argparse


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

    # a simple multilayer perceptron for demonstration purposes
    model = nn.Sequential(
        nn.Linear(28 * 28, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)

    # training
    losses = []
    for epoch in range(args.training_epochs):
        avg_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(train_loader)
        losses.append(avg_loss)
        print(avg_loss)


if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, default="MNIST",
        help='Which dataset to use: MNIST or KMNIST')
    parser.add_argument('--optimizer', type=str, default="adam",
        help='Name of the optimizer: sgd, momentumsgd, adam, adagrad, rmsprop')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
        help='Learning rate')
    parser.add_argument('--training_epochs', type=int, default=15,
        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100,
        help='Number of batch sizes')
    parser.add_argument('--epsilon', type=float, default=1e-8,
        help='Smoothing term to avoid zero-division')
    parser.add_argument('--tau', default=0.9,
        help='Decaying parameter')
    parser.add_argument('--rho', type=float, default=0.9,
        help='momentum')
    parser.add_argument('--beta1', type=float, default=0.9,
        help='first order decaying parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
        help='second order decaying parameter')
    parser.add_argument('--output', type=str, default=None,
        help='Output file to save training loss and accuracy.')
    args = parser.parse_args()
    main(args)
