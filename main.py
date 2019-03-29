import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
from optimizers import create_optimizer

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 28 * 28)


def main(args):
    root = "data"
    if not os.path.exists(root):
        os.mkdir(root)

    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    # a simple multilayer perceptron for demonstration purposes
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.SGD(model.parameters(), lr=1)
    optimizer = create_optimizer(args)

    print("Training model...")
    losses = []
    for epoch in range(args.training_epochs):
        avg_loss = 0
        correct = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            base_optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.update(model)
            base_optimizer.step()
            _, prediction = torch.max(out.data, 1)
            correct += (prediction == target.data).sum().item()
            avg_loss += loss.item()
        avg_loss /= len(train_loader.dataset)
        accuracy = correct / len(train_loader.dataset)

        losses.append(avg_loss)
        print(
            "Epoch=%-3d" % (epoch + 1)
            + " loss={:.8f}   accuracy={:.4f}".format(avg_loss, accuracy)
        )

    print("Testing on test set")
    avg_loss = 0
    correct = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        out = model(x)
        loss = criterion(out, target)
        _, prediction = torch.max(out.data, 1)
        correct += (prediction == target.data).sum().item()
        avg_loss += loss.item()
    avg_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print("Test:  loss={:.8f}  accuracy={:.4f}".format(avg_loss, accuracy))


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--optimizer', type=str, default="adagrad",
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
        help='Momentum')
    parser.add_argument('--beta1', type=float, default=0.9,
        help='First order decaying parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
        help='Fecond order decaying parameter')
    parser.add_argument('--output', type=str, default=None,
        help='Output file to save training loss and accuracy.')
    args = parser.parse_args()
    main(args)
