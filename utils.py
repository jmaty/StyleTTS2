# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from monotonic_align.core import maximum_path_c
from munch import Munch
from torchaudio.transforms import Resample


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_list = f.readlines()
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_list = f.readlines()

    return train_list, val_list


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    if isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


def log_print(message, logger):
    logger.info(message)
    print(message)


# NCCL warm-up to avoid the following warning:
# [rank0]:[W917 12:33:57.081879346 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 0]
# using GPU 0 to perform barrier as devices used by this process are currently unknown.
# This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify
# device_ids in barrier() to force use of a particular device, or call init_process_group()
# with a device_id.
def nccl_warmup(device=None, local_rank=None):
    """
    Perform NCCL warmup to initialize CUDA collective communication.
    This function performs a simple all-reduce operation to warm up the NCCL backend
    for distributed training. It helps ensure that subsequent collective operations
    run smoothly by initializing the communication infrastructure.
    Args:
        device (torch.device, int, or None, optional): The CUDA device to use for warmup.
            If None, attempts to determine device from local_rank or current device.
            If int, creates a CUDA device with that index.
            If torch.device, must be a CUDA device. Defaults to None.
        local_rank (int, optional): The local rank of the process in distributed training.
            Used to determine the CUDA device if device is None. Defaults to None.
    Returns:
        None
    Note:
        - Only performs warmup if distributed training is available and initialized
        - Only works with NCCL backend and when CUDA is available
        - Falls back to LOCAL_RANK environment variable if local_rank is not provided
        - Silently returns if any requirements are not met or if errors occur
        - Performs synchronization after the all-reduce operation
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() != "nccl" or not torch.cuda.is_available():
        return

    dev = device
    if dev is None:
        if isinstance(local_rank, int):
            dev = torch.device("cuda", local_rank)
        else:
            try:
                dev = torch.device("cuda", torch.cuda.current_device())
            except Exception:
                try:
                    lr = int(os.environ.get("LOCAL_RANK", 0))
                    dev = torch.device("cuda", lr)
                except Exception:
                    return
    elif isinstance(dev, int):
        dev = torch.device("cuda", dev)
    elif isinstance(dev, torch.device):
        if dev.type != "cuda":
            return
    else:
        return

    try:
        t = torch.zeros(1, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    except Exception:
        pass


def warmup_scheduler(step, beg, end):
    # Compute warmup progress
    if step >= end:
        progress = 1.0
    else:
        progress = (step - beg) / float(end - beg)
        progress = 1.0 if progress >= 1.0 else (0.0 if progress <= 0.0 else progress)
    return progress


class Resampler:
    """
    A wrapper class for audio resampling functionality.

    This class provides a convenient interface for resampling audio waves from one
    frequency to another using the underlying Resample functionality.

    Args:
        orig_freq (int): The original sampling frequency of the input audio.
        new_freq (int): The target sampling frequency for the output audio.

    Methods:
        __call__(waves): Resamples the input audio waves to the target frequency.

    Example:
        >>> resampler = Resampler(orig_freq=44100, new_freq=22050)
        >>> resampled_audio = resampler(audio_waves)
    """

    def __init__(self, orig_freq, new_freq, device=None):
        self.resampler = Resample(orig_freq=orig_freq, new_freq=new_freq)

        # Auto-detect device if not provided
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.resampler = self.resampler.to(device)  # Move resampler to the specified device
        self.device = device

    def __call__(self, waves):
        return self.resampler(waves)
