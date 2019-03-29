import torch
from abc import abstractmethod


class Optimizer(object):
    @abstractmethod
    def update(self, gradients):
        raise NotImplementedError("compute_update function is not implemented.")


class SGDOptimizer(Optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate

    def update(self, model):
        for p in model.parameters():
            p.grad *= self.learning_rate


class MomentumSGDOptimizer(Optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.m = None

    def update(self, model):
        if self.m is None:
            self.m = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            self.m[i] = self.rho * self.m[i] + p.grad
            p.grad = self.learning_rate * self.m[i]


class AdagradOptimizer(Optimizer):
    def __init__(self, args):
        self.delta = args.delta
        self.learning_rate = args.learning_rate
        self.r = None

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            grad = p.grad
            self.r[i] += grad * grad
            p.grad = (self.learning_rate / (self.delta + torch.sqrt(self.r[i]))) * grad


def create_optimizer(args):
    if args.optimizer == "sgd":
        return SGDOptimizer(args)
    elif args.optimizer == "momentumsgd":
        return MomentumSGDOptimizer(args)
