import os
import torch
from torchvision import datasets, transforms

import numpy as np

from args import args


class MNIST:
    def __init__(self):
        super(MNIST, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

    def update_task(self, i):
        return


class FashionMNIST:
    def __init__(self):
        super(FashionMNIST, self).__init__()

        data_root = os.path.join(args.data, "fashionmnist")

        use_cuda = torch.cuda.is_available()

        train_dataset = datasets.FashionMNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

    def update_task(self, i):
        return



class Permute(object):
    def __call__(self, tensor):
        out = tensor.flatten()
        out = out[self.perm]
        return out.view(1, 28, 28)

    def __repr__(self):
        return self.__class__.__name__

class MNISTPerm:
    def __init__(self):
        super(MNISTPerm, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        self.permuter = Permute()

        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    self.permuter,
                ]
            ),
        )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        self.permuter,
                    ]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
        )

    def update_task(self, i):
        np.random.seed(i + args.seed)
        self.permuter.__setattr__("perm", np.random.permutation(784))

