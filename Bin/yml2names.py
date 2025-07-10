#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import sys

import yaml

from logger import AlignedColoredFormatter, get_logger, setup_logging


def reduce_speakers(stats, num_speakers):
    if len(stats) > num_speakers:
        # Randomly select a subset of speakers if more than num_speakers
        selected_speakers = random.sample(list(stats.keys()), num_speakers)
        stats = {spk_id: stats[spk_id] for spk_id in selected_speakers}
    return stats


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Limit number of speakers.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="input file list in YAML format (if not specified, input from stdin)",
    )
    parser.add_argument(
        "output", nargs="?", help="output file (if not specified, output to stdout)"
    )
    parser.add_argument(
        "-L",
        "--loglevel",
        type=str.upper,
        help="Set logging level. Default=INFO",
        default="INFO",
    )
    args = parser.parse_args()

    # Set up logging
    formatter = AlignedColoredFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%y%m%d-%H:%M:%S",
        name_width=15,
    )
    setup_logging(
        level=getattr(logging, args.loglevel, logging.INFO),
        formatter=formatter,
        file=None,
    )
    logger = get_logger(__name__)  # Get a logger

    # Load input YAML file from file or stdin
    input_source = open(args.input, "r", encoding="utf-8") if args.input else sys.stdin
    with input_source as f:
        stats = yaml.safe_load(f)
    logger.info("Loaded %d speakers", len(stats))

    # Output determination (stdout or file)
    output_target = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    # Save selected speaker names to a file
    for spk_id in stats.keys():
        fpath = next(iter(stats[spk_id].keys()))
        name = fpath.split(os.sep)[0]
        output_target.write(f"{name}\n")

    # Close the file if it was opened
    if args.output:
        output_target.close()


if __name__ == "__main__":
    main()
