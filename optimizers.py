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
        self.epsilon = args.epsilon
        self.learning_rate = args.learning_rate
        self.r = None

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            grad = p.grad
            self.r[i] += grad * grad
            p.grad = (
                self.learning_rate / (self.epsilon + torch.sqrt(self.r[i]))
            ) * grad


class RMSPropOptimizer(Optimizer):
    def __init__(self, args):
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.r = None
        self.epsilon = args.epsilon

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            grad = p.grad
            self.r[i] = self.tau * self.r[i] + (1 - self.tau) * (grad * grad)
            p.grad = (
                self.learning_rate / (self.epsilon + torch.sqrt(self.r[i]))
            ) * grad


class AdamOptimizer(Optimizer):
    def __init__(self, args):
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.iteration = None
        self.m1 = None
        self.m2 = None

    def update(self, model):
        if self.m1 is None:
            self.m1 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.m2 is None:
            self.m2 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(model.parameters()):
            grad = p.grad
            self.m1[i] = self.beta1 * self.m1[i] + (1 - self.beta1) * grad
            self.m2[i] = self.beta2 * self.m2[i] + (1 - self.beta2) * (grad * grad)

            # correct bias
            m1_corrected = self.m1[i] / (1 - self.beta1 ** self.iteration)
            m2_corrected = self.m2[i] / (1 - self.beta2 ** self.iteration)

            p.grad = self.learning_rate * (
                m1_corrected / (self.epsilon + torch.sqrt(m2_corrected))
            )

        self.iteration += 1


def create_optimizer(args):
    if args.optimizer == "sgd":
        return SGDOptimizer(args)
    elif args.optimizer == "momentumsgd":
        return MomentumSGDOptimizer(args)
    elif args.optimizer == "adagrad":
        return AdagradOptimizer(args)
    elif args.optimizer == "rmsprop":
        return RMSPropOptimizer(args)
    elif args.optimizer == "adam":
        return AdamOptimizer(args)
    else:
        raise ValueError("invalid optimizer")
