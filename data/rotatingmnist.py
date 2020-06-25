import os
import torch
from torchvision import datasets, transforms

import numpy as np

from args import args

count = 0


class Rotate(object):
    def __call__(self, img):
        out = transforms.functional.rotate(img, self.angle)
        return out

    def __repr__(self):
        return self.__class__.__name__


class RotatingMNIST:
    def __init__(self):
        global count
        super(RotatingMNIST, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        self.rotater = Rotate()

        train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(3),
                    self.rotater,
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
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
                        transforms.Grayscale(3),
                        self.rotater,
                        transforms.Grayscale(1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
        )

    def update_task(self, i):
        self.rotater.__setattr__('angle', i*(360 // args.num_tasks))
