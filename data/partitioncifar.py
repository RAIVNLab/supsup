import numpy as np
import os
import torch

from torchvision import datasets, transforms

import copy

from args import args
import utils


def partition_dataset(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]

    newdataset.targets = [
        label - torch.tensor(i)
        for label in newdataset.targets
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]
    return newdataset


class PartitionCIFAR10:
    def __init__(self):
        super(PartitionCIFAR10, self).__init__()
        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_dataset(train_dataset, 2 * i),
                partition_dataset(val_dataset, 2 * i),
            )
            for i in range(5)
        ]

        for i in range(5):
            print()
            print(f"=> Size of train split {i}: {len(splits[i][0].data)}")
            print(f"=> Size of val split {i}: {len(splits[i][1].data)}")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]


def partition_datasetv2(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]

    newdataset.targets = [
        label
        for label in newdataset.targets
        if label == torch.tensor(i) or label == torch.tensor(i + 1)
    ]
    return newdataset


class PartitionCIFAR10V2:
    def __init__(self):
        super(PartitionCIFAR10V2, self).__init__()
        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_datasetv2(train_dataset, 2 * i),
                partition_datasetv2(val_dataset, 2 * i),
            )
            for i in range(5)
        ]

        for i in range(5):
            print(len(splits[i][0].data))
            print(len(splits[i][1].data))
            print("==")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]


def partition_datasetv3(dataset, i):
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label == torch.tensor(i)
        or label == torch.tensor(i + 1)
        or label == torch.tensor(i + 2)
        or label == torch.tensor(i + 3)
        or label == torch.tensor(i + 4)
    ]

    newdataset.targets = [
        label - torch.tensor(i)
        for label in newdataset.targets
        if label == torch.tensor(i)
        or label == torch.tensor(i + 1)
        or label == torch.tensor(i + 2)
        or label == torch.tensor(i + 3)
        or label == torch.tensor(i + 4)
    ]
    return newdataset


class PartitionCIFAR100V2:
    def __init__(self):
        super(PartitionCIFAR100V2, self).__init__()
        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        splits = [
            (
                partition_datasetv3(train_dataset, 5 * i),
                partition_datasetv3(val_dataset, 5 * i),
            )
            for i in range(args.num_tasks)
        ]

        # for i in range(20):
        #     print(len(splits[i][0].data))
        #     print(len(splits[i][1].data))
        #     print("==")

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]




def partition_datasetv4(dataset, perm):
    lperm = perm.tolist()
    newdataset = copy.copy(dataset)
    newdataset.data = [
        im
        for im, label in zip(newdataset.data, newdataset.targets)
        if label in lperm
    ]

    newdataset.targets = [
        lperm.index(label)
        for label in newdataset.targets
        if label in lperm
    ]
    return newdataset

class RandSplitCIFAR100:
    def __init__(self):
        super(RandSplitCIFAR100, self).__init__()
        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        np.random.seed(args.seed)
        perm = np.random.permutation(100)
        print(perm)

        splits = [
            (
                partition_datasetv4(train_dataset, perm[5 * i:5 * (i+1)]),
                partition_datasetv4(val_dataset, perm[5 * i:5 * (i+1)]),
            )
            for i in range(args.num_tasks)
        ]

        # for i in range(20):
        #     print(len(splits[i][0].data))
        #     print(len(splits[i][1].data))
        #     print("==")
        [print(perm[5 * i:5 * (i+1)]) for i in range(args.num_tasks)]

        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        self.loaders = [
            (
                torch.utils.data.DataLoader(
                    x[0], batch_size=args.batch_size, shuffle=True, **kwargs
                ),
                torch.utils.data.DataLoader(
                    x[1], batch_size=args.test_batch_size, shuffle=True, **kwargs
                ),
            )
            for x in splits
        ]

    def update_task(self, i):
        self.train_loader = self.loaders[i][0]
        self.val_loader = self.loaders[i][1]
