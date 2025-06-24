import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
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


# Resampler for speaker encoder waveforms
def get_speaker_encoder_resampler(sr, spkenc_sr):
    """
    Returns a resampler for the speaker encoder based on the configuration.
    """
    if spkenc_sr != sr:
        spkenc_resampler = Resample(
            orig_freq=sr,
            new_freq=spkenc_sr,
        )
    else:
        spkenc_resampler = None
    return spkenc_resampler


# def get_segments_for_spkenc(waves, segment_begs, segment_len):
#     """
#     Get segments for speaker encoder from the waveform.
#     waves: [B, T]
#     segment_begs: [B]
#     segment_len: int
#     """
#     batch_size = waves.size(0)
#     device = waves.device

#     # Create a tensor of indices for the segments
#     batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
#     # Create a tensor of segment beginnings
#     segment_offsets = torch.arange(segment_len, device=device).unsqueeze(0)
#     # Calculate the indices for the segments
#     indices = segment_begs.unsqueeze(1) + segment_offsets
#     # Extract the segments using advanced indexing
#     segments = waves[batch_indices, indices]

#     return segments


# def get_segments_for_spkenc(waves, segment_begs, segment_len, spkenc_resampler=None, pad_value=0.0):#
#     """
#     Get segments for speaker encoder from the waveform with padding.

#     Args:
#         waves: [B, T] - Batch of waveforms
#         segment_begs: [B] - Starting indices for each segment in batch
#         segment_len: int - Length of each segment
#         pad_value: float - Value to use for padding (default 0.0)

#     Returns:
#         segments: [B, segment_len] - Extracted segments with padding
#     """

#     print("Orig waves shape:", waves.shape)
#     print("Segment length:", segment_len)

#     # Resample the waveforms for speaker encoder
#     if spkenc_resampler:
#         waves = spkenc_resampler(waves)

#     print("Processed waves shape:", waves.shape)

#     batch_size, total_len = waves.shape
#     device = waves.device
#     dtype = waves.dtype

#     # Pre-allocate output tensor filled with pad_value
#     segments = torch.full((batch_size, segment_len), pad_value, device=device, dtype=dtype)

#     # Create index tensors for matrix operations
#     batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
#     segment_offsets = torch.arange(segment_len, device=device).unsqueeze(0)  # [1, segment_len]
#     indices = segment_begs.unsqueeze(1) + segment_offsets  # [B, segment_len]

#     # Create mask for valid indices (within bounds)
#     valid_mask = (indices >= 0) & (indices < total_len)

#     # Clamp indices to valid range for safe indexing
#     safe_indices = torch.clamp(indices, 0, total_len - 1)

#     # Extract values using advanced indexing
#     extracted = waves[batch_indices, safe_indices]

#     # Apply mask - only use valid values, keep pad_value for invalid positions
#     segments = torch.where(valid_mask, extracted, pad_value)

#     print("Final segments shape:", segments.shape)

#     return segments


def resample(waves, resampler):
    """
    Resample the waveform using the provided resampler.

    Args:
        waves (torch.Tensor): Input waveform tensor of shape [B, T].
        resampler (Resample): Resample waveform to the desired sample rate.

    Returns:
        torch.Tensor: Resampled waveform tensor.
    """
    if resampler:
        waves = resampler(waves)
    return waves
