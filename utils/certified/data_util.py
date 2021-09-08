"""Data handler classes and methods"""

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch.nn.functional as F
import random

# NEIGHBOR_FILE = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/counterfitted_neighbors.json'


def dict_batch_to_device(batch, device):
    """
    Moves a batch of data to device
    Args:
      - batch: Can be a Torch tensor or a dict where the values are torch tensors
      - device: A Torch device to move all the tensors to
    Returns:
      - a batch of the same type as input batch but on the device
        If a dict, also a dict with same keys
    """
    try:
        return batch.to(device)
    except AttributeError:
        # don't have a to function, must be a dict, recursively move to device
        return {k: dict_batch_to_device(v, device) for k, v in batch.items()}


def multi_dim_padded_cat(tensors, dim, padding_value=0):
    """
    Concatenates tensors along dim, padding elements to the largest length at
    each dimension. Assumes all tensors have the same dimensionality but makes no
    other assumptions about their size
    """
    if dim == 0:
        original_ordering = dim_first_ordering = list(range(len(tensors[0].shape)))
    else:
        # If dim is not 0, we make it so for ease later and re-permute at the end
        dims = list(range(len(tensors[0].shape)))
        dims.pop(dim)
        dim_first_ordering = [dim] + dims
        original_ordering = []
        for dim_idx in range(len(dim_first_ordering)):
            if dim_idx < dim:
                original_ordering.append(dim_idx + 1)
            elif dim_idx == dim:
                original_ordering.append(0)
            else:
                original_ordering.append(dim_idx)
    out_shape = []
    for in_dim in dim_first_ordering:
        out_shape.append(max(tensor.shape[in_dim] for tensor in tensors))
    out_shape[0] = sum(tensor.shape[dim] for tensor in tensors)
    out_tensor = tensors[0].new_empty(out_shape)
    cur_idx = 0
    for tensor in tensors:
        out_shape[0] = tensor.shape[dim]
        pad = []
        # see torch.nn.functional.pad documentation for why we need this format
        for tensor_dim, out_dim in list(zip(tensor.shape, out_shape))[:0:-1]:
            pad = pad + [0, out_dim - tensor_dim]
        out_tensor[cur_idx:cur_idx + out_shape[0], ...] = F.pad(tensor.permute(*dim_first_ordering), pad)
        cur_idx += out_shape[0]
    if dim != 0:
        out_tensor = out_tensor.permute(*original_ordering)
    return out_tensor


class DataAugmenter(object):
    def __init__(self, augment_by):
        self.augment_by = augment_by

    def augment(self, dataset):
        raise NotImplementedError
