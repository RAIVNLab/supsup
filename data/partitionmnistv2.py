import os
import torch
from torchvision import datasets, transforms

import copy

from args import args

def partition_dataset(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]

    newdataset.targets = [
        label #- torch.tensor(i)
        for label in newdataset.targets
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]
    return newdataset


class PartitionMNISTV2:
    def __init__(self):
        super(PartitionMNISTV2, self).__init__()
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

        val_dataset = datasets.MNIST(
            data_root,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        splits = [
            (partition_dataset(train_dataset, 2*i), partition_dataset(val_dataset, 2*i))
            for i in range(5)
        ]

        for i in range(5):
            print(len(splits[i][0].data))
            print(len(splits[i][1].data))
            print('==')


        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}


        self.loaders = [
            (torch.utils.data.DataLoader(
                x[0], batch_size=args.batch_size, shuffle=True, **kwargs
            ),
             torch.utils.data.DataLoader(
                 x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
             ))
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
