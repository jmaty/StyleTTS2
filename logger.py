import sys
import logging


class ColoredFormatter(logging.Formatter):
    """Formatter for colored logging output."""

    COLORS = {
        "DEBUG": "\033[94m",  # blue
        "INFO": "\033[92m",  # green
        "WARNING": "\033[93m",  # yellow
        "ERROR": "\033[91m",  # red
        "CRITICAL": "\033[41m\033[97m",  # white on red background
        "RESET": "\033[0m",  # reset formatting
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            # Store original length before adding ANSI codes
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            # Set levelname with colors
            record.levelname = colored_levelname
            # Compensation for padding when using colors (ANSI codes are not visible characters)
            record._padding_len = len(self.COLORS[levelname]) + len(self.COLORS["RESET"])

        formatted = super().format(record)
        return formatted


class AlignedColoredFormatter(logging.Formatter):
    """Formatter with perfect alignment even with color codes."""

    COLORS = {
        "DEBUG": "\033[94m",  # blue
        "INFO": "\033[92m",  # green
        "WARNING": "\033[93m",  # yellow
        "ERROR": "\033[91m",  # red
        "CRITICAL": "\033[41m\033[97m",  # white on red background
        "RESET": "\033[0m",  # reset formatting
    }

    def __init__(self, fmt=None, datefmt=None, style="%", level_width=8, name_width=25):
        self.level_width = level_width
        self.name_width = name_width
        # Store the formatting string for later modification
        self.original_fmt = fmt
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        # Store original values
        orig_levelname = record.levelname
        orig_name = record.name

        # Prepare aligned levelname with padding for proper width
        padding = max(0, self.level_width - len(orig_levelname))
        padding_spaces = " " * padding

        # Apply colors to levelname and add padding
        if orig_levelname in self.COLORS:
            record.levelname = f"{self.COLORS[orig_levelname]}{orig_levelname}{self.COLORS['RESET']}{padding_spaces}"
        else:
            record.levelname = f"{orig_levelname}{padding_spaces}"

        # Truncate overly long module names
        if len(orig_name) > self.name_width:
            # Shorten name and add '.' at the start
            record.name = f".{orig_name[-(self.name_width-1):]}"
        else:
            # Pad with spaces to the required width
            record.name = f"{orig_name}{' ' * (self.name_width - len(orig_name))}"

        # Truncate overly long module names - zachovat jen část po poslední tečce
        if len(orig_name) > self.name_width:
            # Get the last part of the name (after the last dot)
            last_part = orig_name.split(".")[-1]

            # If the last part is also too long, truncate it
            if len(last_part) > self.name_width:
                record.name = f".{last_part[-(self.name_width-1):]}"
            else:
                # Otherwise, display the whole last part with a possible truncation indicator
                prefix = "." if "." in orig_name else ""
                record.name = f"{prefix}{last_part}"
                # Pad with spaces to the required width
                record.name = f"{record.name}{' ' * (self.name_width - len(record.name))}"

        # Adjust formatting string - remove width specifications as we have handled them manually
        adjusted_fmt = self.original_fmt.replace("%(levelname)-8s", "%(levelname)s")
        adjusted_fmt = adjusted_fmt.replace("%(name)-20s", "%(name)s")
        self._style._fmt = adjusted_fmt

        # Format the record
        result = super().format(record)

        # Restore original values and formatting string
        record.levelname = orig_levelname
        record.name = orig_name
        self._style._fmt = self.original_fmt

        return result


def setup_logging(
    level=logging.INFO,
    formatter=None,
    file=None,
    level_file=None,
    formatter_file=None,
):
    """
    Configure root logger with consistent formatting and handling.

    Args:
        level: Overall logging level (e.g. logging.DEBUG, logging.INFO)
        level_file: Optional log file level. If provided, logs will be written to this file.
        file: Optional path to log file. If provided, logs will be written to this file.

    Returns:
        The configured root logger
    """
    # Configure root logger
    root_logger = logging.getLogger()

    # Set default values for file logging if not provided
    if file:
        if level_file is None:
            level_file = level
        if formatter_file is None:
            formatter_file = formatter
        # Set root logger level to the minimum of console and file levels
        # to ensure all logs are captured
        root_logger.setLevel(min(level, level_file))
    else:
        # Set root logger level to the specified level
        root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates when function is called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # Create file handler if log file is specified
    if file:
        # Create basic formatter for file handler
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter_file)
        file_handler.setLevel(level_file)
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
