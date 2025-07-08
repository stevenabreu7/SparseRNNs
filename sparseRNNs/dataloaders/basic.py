"""Implementation of basic benchmark datasets used in S4 experiments: MNIST, CIFAR10 and Speech Commands."""

import numpy as np
import torch
import torchvision
from einops.layers.torch import Rearrange

from sparseRNNs.dataloaders.base import (ImageResolutionSequenceDataset,
                                         ResolutionSequenceDataset,
                                         SequenceDataset, default_data_path)
from sparseRNNs.dataloaders.utils import permutations


class MNIST(SequenceDataset):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    l_output = 0
    L = 784

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(self.d_input, self.L).t()),
        ]  # (L, d_input)
        if self.permute:
            # below is another permutation that other works have used
            # permute = np.random.RandomState(92916)
            # permutation = torch.LongTensor(permute.permutation(784))
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        transform = torchvision.transforms.Compose(transform_list)
        self.dataset_train = torchvision.datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.dataset_test = torchvision.datasets.MNIST(
            self.data_dir,
            train=False,
            transform=transform,
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class SpeechCommands(ResolutionSequenceDataset):
    _name_ = "sc"

    @property
    def init_defaults(self):
        return {
            "mfcc": False,
            "dropped_rate": 0.0,
            "length": 16000,
            "all_classes": False,
        }

    @property
    def d_input(self):
        _d_input = 20 if self.mfcc else 1
        _d_input += 1 if self.dropped_rate > 0.0 else 0
        return _d_input

    @property
    def d_output(self):
        return 10 if not self.all_classes else 35

    @property
    def l_output(self):
        return 0

    @property
    def L(self):
        return 161 if self.mfcc else self.length

    def setup(self):
        self.data_dir = (
            self.data_dir or default_data_path
        )  # TODO make same logic as other classes

        from sparseRNNs.dataloaders.SC35.sc35 import _SpeechCommands

        # TODO refactor with data_dir argument
        self.dataset_train = _SpeechCommands(
            partition="train",
            length=self.L,
            mfcc=self.mfcc,
            sr=self.sr,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_val = _SpeechCommands(
            partition="val",
            length=self.L,
            mfcc=self.mfcc,
            sr=self.sr,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )

        self.dataset_test = _SpeechCommands(
            partition="test",
            length=self.L,
            mfcc=self.mfcc,
            sr=self.sr,
            dropped_rate=self.dropped_rate,
            path=self.data_dir,
            all_classes=self.all_classes,
        )
