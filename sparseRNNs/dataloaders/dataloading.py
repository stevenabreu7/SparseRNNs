import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, TypeVar, Union

import torch

from sparseRNNs.dataloaders.base import default_cache_path, default_data_path

DataLoader = TypeVar("DataLoader")
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]

NDNS_TRAIN_SET = os.environ.get("NDNS_TRAIN_SET", None)
NDNS_TEST_SET = os.environ.get("NDNS_TEST_SET", None)
NDNS_VALIDATION_SET = os.environ.get("NDNS_VALIDATION_SET", None)


def custom_loader(cache_dir: str, bsz: int = 50, seed: int = 42) -> ReturnType: ...


def make_data_loader(
    dset,
    dobj,
    seed: int,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    collate_fn: callable = None,
):
    """

    :param dset: 			(PT dset):		PyTorch dataset object.
    :param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
    :param seed: 			(int):			Int for seeding shuffle.
    :param batch_size: 		(int):			Batch size for batches.
    :param shuffle:         (bool):			Shuffle the data loader?
    :param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
    :return:
    """

    # Create a generator for seeding random number draws.
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    if dobj is not None:
        assert collate_fn is None
        collate_fn = dobj._collate_fn

    # Generate the dataloaders.
    return torch.utils.data.DataLoader(
        dataset=dset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=rng,
    )


def create_ndns_dataset(
    cache_dir: Union[str, Path] = default_cache_path,
    seed: int = 42,
    bsz: int = 128,
    return_metadata: bool = False,
) -> ReturnType:
    name = "ndns"

    from .NDNS.ndns import DNSAudio

    assert (
        NDNS_TRAIN_SET is not None
    ), "NDNS_TRAIN_SET environment variable must be set."
    assert NDNS_TEST_SET is not None, "NDNS_TEST_SET environment variable must be set."
    assert (
        NDNS_VALIDATION_SET is not None
    ), "NDNS_VALIDATION_SET environment variable must be set."

    train_set = DNSAudio(root=NDNS_TRAIN_SET)
    validation_set = DNSAudio(root=NDNS_VALIDATION_SET)
    test_set = DNSAudio(root=NDNS_TEST_SET)

    # copied from N-DNS baseline solution train_sdnn.py
    # https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/blob/main/baseline_solution/sdnn_delays/train_sdnn.py
    def collate_fn(batch):
        noisy, clean, noise = [], [], []
        metadata = []

        for sample in batch:
            noisy += [torch.FloatTensor(sample[0])]
            clean += [torch.FloatTensor(sample[1])]
            noise += [torch.FloatTensor(sample[2])]
            if return_metadata:
                metadata += [sample[3]]

        if return_metadata:
            return (
                torch.stack(noisy),
                torch.stack(clean),
                torch.stack(noise),
                metadata,
            )
        else:
            return torch.stack(noisy), torch.stack(clean), torch.stack(noise)
        # return a jax tensor instead of torch
        # or tensorflow dataloader from others

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_set,
        batch_size=bsz,
        shuffle=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=bsz,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=bsz,
        shuffle=True,
        collate_fn=collate_fn,
    )

    N_CLASSES = 257
    SEQ_LENGTH = 3751
    IN_DIM = 257
    TRAIN_SIZE = len(train_set)
    aux_loaders = {}
    return (
        train_loader,
        validation_loader,
        test_loader,
        aux_loaders,
        N_CLASSES,
        SEQ_LENGTH,
        IN_DIM,
        TRAIN_SIZE,
    )


Datasets = {
    "ndns": create_ndns_dataset,
}
