import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monotonic_align.core import maximum_path_c
from munch import Munch


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


class ColoredFormatter(logging.Formatter):
    """Formatter for colored logging output."""

    COLORS = {
        "DEBUG": "\033[94m",  # modrá
        "INFO": "\033[92m",  # zelená
        "WARNING": "\033[93m",  # žlutá
        "ERROR": "\033[91m",  # červená
        "CRITICAL": "\033[41m\033[97m",  # bílá na červeném pozadí
        "RESET": "\033[0m",  # reset formátování
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure root logger with consistent formatting and handling.

    Args:
        log_level: Overall logging level (e.g. logging.DEBUG, logging.INFO)
        log_file: Optional path to log file. If provided, logs will be written to this file.

    Returns:
        The configured root logger
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates when function is called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter for stdout
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%y%m%d-%H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Create file handler if log file is specified
    if log_file:
        # Create formatter for file handler
        file_formatter = logging.Formatter(
            fmt="%(levelname)s:%(asctime)s: %(message)s",
            datefmt="%y%m%d-%H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name, level=None):
    """
    Get a logger for a specific module with optional level override.

    Args:
        name: Name of the logger, typically __name__ of the module
        level: Optional specific level for this logger

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Set specific level if provided, otherwise inherit from parent
    if level is not None:
        logger.setLevel(level)

    return logger
