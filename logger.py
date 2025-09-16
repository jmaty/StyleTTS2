import logging
import os
import sys
from typing import Optional, Callable

try:
    import torch.distributed as dist
except Exception:
    dist = None

try:
    from accelerate import Accelerator  # type: ignore
except Exception:
    Accelerator = None  # type: ignore

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _get_rank_env() -> int:
    for k in ("RANK", "LOCAL_RANK"):
        v = os.getenv(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def _is_dist_initialized() -> bool:
    return bool(
        dist
        and getattr(dist, "is_available", lambda: False)()
        and getattr(dist, "is_initialized", lambda: False)()
    )


def current_rank(accelerator: Optional["Accelerator"] = None) -> int:
    if accelerator is not None:
        # accelerate >=0.20
        try:
            return int(getattr(accelerator, "process_index"))
        except Exception:
            # fallback
            return 0 if is_main_process(accelerator) else 1
    if _is_dist_initialized():
        try:
            return int(dist.get_rank())
        except Exception:
            pass
    return _get_rank_env()


def is_main_process(accelerator: Optional["Accelerator"] = None) -> bool:
    if accelerator is not None:
        try:
            return bool(getattr(accelerator, "is_main_process"))
        except Exception:
            return True
    if _is_dist_initialized():
        try:
            return int(dist.get_rank()) == 0
        except Exception:
            pass
    return current_rank(accelerator) == 0


class _MainOnlyFilter(logging.Filter):
    def __init__(self, is_main_fn: Callable[[], bool]) -> None:
        super().__init__()
        self._is_main_fn = is_main_fn

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(self._is_main_fn())


class _RankAugmentFilter(logging.Filter):
    def __init__(self, rank_fn: Callable[[], int]) -> None:
        super().__init__()
        self._rank_fn = rank_fn

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.rank = self._rank_fn()
        except Exception:
            record.rank = 0
        return True


def add_logging_args(parser) -> None:
    parser.add_argument(
        "--log-level",
        "--log_level",
        default="INFO",
        choices=list(_LEVELS.keys()),
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        "--log_file",
        default=None,
        help="Path to the log file (only writes to main process).",
    )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    accelerator: Optional["Accelerator"] = None,
    *,
    fmt: Optional[str] = None,
    fmt_file: Optional[str] = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    level_num = _LEVELS.get(str(level).upper(), logging.INFO)

    root = logging.getLogger()
    # Remove all existing handlers to prevent duplicates in spawn/torchrun
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(level_num)

    if fmt is None:
        # fmt = "%(asctime)s | %(levelname).1s | r%(rank)d | %(name)s | %(message)s"
        fmt = "%(message)s"
    if fmt_file is None:
        fmt_file = "%(asctime)s | %(levelname).1s | r%(rank)d | %(name)s | %(message)s"

    is_main_fn = lambda: is_main_process(accelerator)
    rank_fn = lambda: current_rank(accelerator)

    # Konzolový handler (jen hlavní proces)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level_num)
    ch.addFilter(_RankAugmentFilter(rank_fn))
    ch.addFilter(_MainOnlyFilter(is_main_fn))
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(ch)

    # Souborový handler (volitelný, jen hlavní proces)
    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level_num)
        fh.addFilter(_RankAugmentFilter(rank_fn))
        fh.addFilter(_MainOnlyFilter(is_main_fn))
        fh.setFormatter(logging.Formatter(fmt=fmt_file, datefmt=datefmt))
        root.addHandler(fh)

    # Utišit upovídané knihovny
    for noisy in ("urllib3", "matplotlib", "numba", "asyncio", "accelerate.state"):
        try:
            logging.getLogger(noisy).setLevel(max(level_num, logging.WARNING))
        except Exception:
            pass

    return logging.getLogger(__name__)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)


# import sys
# import logging


# class ColoredFormatter(logging.Formatter):
#     """Formatter for colored logging output."""

#     COLORS = {
#         "DEBUG": "\033[94m",  # blue
#         "INFO": "\033[92m",  # green
#         "WARNING": "\033[93m",  # yellow
#         "ERROR": "\033[91m",  # red
#         "CRITICAL": "\033[41m\033[97m",  # white on red background
#         "RESET": "\033[0m",  # reset formatting
#     }

#     def format(self, record):
#         levelname = record.levelname
#         if levelname in self.COLORS:
#             # Store original length before adding ANSI codes
#             colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
#             # Set levelname with colors
#             record.levelname = colored_levelname
#             # Compensation for padding when using colors (ANSI codes are not visible characters)
#             record._padding_len = len(self.COLORS[levelname]) + len(self.COLORS["RESET"])

#         formatted = super().format(record)
#         return formatted


# class AlignedColoredFormatter(logging.Formatter):
#     """Formatter with perfect alignment even with color codes."""

#     COLORS = {
#         "DEBUG": "\033[94m",  # blue
#         "INFO": "\033[92m",  # green
#         "WARNING": "\033[93m",  # yellow
#         "ERROR": "\033[91m",  # red
#         "CRITICAL": "\033[41m\033[97m",  # white on red background
#         "RESET": "\033[0m",  # reset formatting
#     }

#     def __init__(self, fmt=None, datefmt=None, style="%", level_width=8, name_width=25):
#         self.level_width = level_width
#         self.name_width = name_width
#         # Store the formatting string for later modification
#         self.original_fmt = fmt
#         super().__init__(fmt, datefmt, style)

#     def format(self, record):
#         # Store original values
#         orig_levelname = record.levelname
#         orig_name = record.name

#         # Prepare aligned levelname with padding for proper width
#         padding = max(0, self.level_width - len(orig_levelname))
#         padding_spaces = " " * padding

#         # Apply colors to levelname and add padding
#         if orig_levelname in self.COLORS:
#             record.levelname = f"{self.COLORS[orig_levelname]}{orig_levelname}{self.COLORS['RESET']}{padding_spaces}"
#         else:
#             record.levelname = f"{orig_levelname}{padding_spaces}"

#         # Truncate overly long module names
#         if len(orig_name) > self.name_width:
#             # Shorten name and add '.' at the start
#             record.name = f".{orig_name[-(self.name_width-1):]}"
#         else:
#             # Pad with spaces to the required width
#             record.name = f"{orig_name}{' ' * (self.name_width - len(orig_name))}"

#         # Truncate overly long module names - zachovat jen část po poslední tečce
#         if len(orig_name) > self.name_width:
#             # Get the last part of the name (after the last dot)
#             last_part = orig_name.split(".")[-1]

#             # If the last part is also too long, truncate it
#             if len(last_part) > self.name_width:
#                 record.name = f".{last_part[-(self.name_width-1):]}"
#             else:
#                 # Otherwise, display the whole last part with a possible truncation indicator
#                 prefix = "." if "." in orig_name else ""
#                 record.name = f"{prefix}{last_part}"
#                 # Pad with spaces to the required width
#                 record.name = f"{record.name}{' ' * (self.name_width - len(record.name))}"

#         # Adjust formatting string - remove width specifications as we have handled them manually
#         adjusted_fmt = self.original_fmt.replace("%(levelname)-8s", "%(levelname)s")
#         adjusted_fmt = adjusted_fmt.replace("%(name)-20s", "%(name)s")
#         self._style._fmt = adjusted_fmt

#         # Format the record
#         result = super().format(record)

#         # Restore original values and formatting string
#         record.levelname = orig_levelname
#         record.name = orig_name
#         self._style._fmt = self.original_fmt

#         return result


# def setup_logging(
#     level=logging.INFO,
#     formatter=None,
#     file=None,
#     level_file=None,
#     formatter_file=None,
# ):
#     """
#     Configure root logger with consistent formatting and handling.

#     Args:
#         level: Overall logging level (e.g. logging.DEBUG, logging.INFO)
#         level_file: Optional log file level. If provided, logs will be written to this file.
#         file: Optional path to log file. If provided, logs will be written to this file.

#     Returns:
#         The configured root logger
#     """
#     # Configure root logger
#     root_logger = logging.getLogger()

#     # Set default values for file logging if not provided
#     if file:
#         if level_file is None:
#             level_file = level
#         if formatter_file is None:
#             formatter_file = formatter
#         # Set root logger level to the minimum of console and file levels
#         # to ensure all logs are captured
#         root_logger.setLevel(min(level, level_file))
#     else:
#         # Set root logger level to the specified level
#         root_logger.setLevel(level)

#     # Remove existing handlers to avoid duplicates when function is called multiple times
#     for handler in root_logger.handlers[:]:
#         root_logger.removeHandler(handler)

#     # Create console handler
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(formatter)
#     console_handler.setLevel(level)
#     root_logger.addHandler(console_handler)

#     # Create file handler if log file is specified
#     if file:
#         # Create basic formatter for file handler
#         file_handler = logging.FileHandler(file)
#         file_handler.setFormatter(formatter_file)
#         file_handler.setLevel(level_file)
#         root_logger.addHandler(file_handler)

#     return root_logger


# def get_logger(name, level=None):
#     """
#     Get a logger for a specific module with optional level override.

#     Args:
#         name: Name of the logger, typically __name__ of the module
#         level: Optional specific level for this logger

#     Returns:
#         Logger instance
#     """
#     logger = logging.getLogger(name)

#     # Set specific level if provided, otherwise inherit from parent
#     if level is not None:
#         logger.setLevel(level)

#     return logger
