# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Optional

import torch
from PIL import ImageFilter
from torchvision import datasets, transforms


def collate_fn(batch):
    return (
        torch.stack([x[0][0] for x in batch], dim=0),  # sender_input
        torch.cat([torch.Tensor([x[1]]).long() for x in batch], dim=0),  # labels
        torch.stack([x[0][1] for x in batch], dim=0),  # receiver_input
    )


def get_dataloader(dataset_dir, batch_size, image_size, num_workers, 
                   seed=111):
    transformations = ImageTransformation(image_size)
    
    train_dataset = datasets.CIFAR100(
        root=dataset_dir, 
        download=True,
        transform=transformations
        )
    
    test_dataset = datasets.CIFAR100(
        root = dataset_dir,
        train = False,
        transform = transformations
    )
    
    eval_dataset, train_dataset = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) // 10, len(train_dataset) // 10 * 9],
        torch.Generator().manual_seed(seed),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    return train_loader, eval_loader, test_loader


class ImageTransformation:
    
    def __init__(self, image_size):
    
        transformations = [transforms.Resize(size=(image_size, image_size))]
        transformations.append(transforms.ToTensor())
        transformations.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
        self.transform = transforms.Compose(transformations)

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        return x_i, x_j
