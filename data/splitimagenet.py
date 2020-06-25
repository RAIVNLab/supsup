import os

import torch
from torchvision import datasets, transforms
from args import args

import torch.multiprocessing

import numpy as np

from copy import copy, deepcopy
from itertools import chain

torch.multiprocessing.set_sharing_strategy("file_system")


class SplitImageNet:
    def __init__(self):
        super(SplitImageNet, self).__init__()

        data_root = os.path.join(args.data, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        train_datasets, val_datasets = self._construct_dataset_splits(
            train_dataset, val_dataset
        )

        self.train_loaders = []
        self.val_loaders = []

        for td, vd in zip(train_datasets, val_datasets):
            self.train_loaders.append(
                torch.utils.data.DataLoader(
                    td, batch_size=args.batch_size, shuffle=True, **kwargs
                )
            )

            self.val_loaders.append(
                torch.utils.data.DataLoader(
                    vd, batch_size=args.batch_size, shuffle=False, **kwargs
                )
            )

    def _construct_dataset_splits(self, train_dataset, val_dataset):
        print(f"==> Using seed {1} for ImageNet split")
        np.random.seed(1)

        print("=> Generating split order with seed")
        class_split_order = np.random.permutation(1000)

        print("=> Splitting train dataset")
        train_datasets = self._split_dataset(train_dataset, class_split_order)

        print("=> Splitting val dataset")
        val_datasets = self._split_dataset(val_dataset, class_split_order)

        return train_datasets, val_datasets

    def _split_dataset(self, dataset, class_split_order):
        task_length = len(class_split_order) // args.num_tasks

        # Used to map from random task_length classes in {0...1000} -> {0,1...task_length}
        tiled_class_map = np.tile(np.arange(task_length), args.num_tasks)
        inv_class_split_order = np.argsort(class_split_order)
        class_map = tiled_class_map[inv_class_split_order]

        # Constructing class splits
        paths, targets = zip(*dataset.samples)

        paths = np.array(paths)
        targets = np.array(targets)

        print("==> Extracting per class paths")
        class_samples = [
            list(zip(paths[targets == c], class_map[targets[targets == c]]))
            for c in range(1000)
        ]

        datasets = []

        print(f"==> Splitting dataset into {args.num_tasks} tasks")
        for i in range(0, 1000, task_length):
            task_classes = class_split_order[i : i + task_length]

            samples = []

            for c in task_classes:
                samples.append(class_samples[c])

            redataset = copy(dataset)
            redataset.samples = list(chain.from_iterable(samples))

            datasets.append(redataset)

        return datasets

    def update_task(self, i):
        self.train_loader = self.train_loaders[i]
        self.val_loader = self.val_loaders[i]
