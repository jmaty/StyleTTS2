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
        "-n",
        "--num_speakers",
        type=int,
        default=float("inf"),
        help="number of speakers of one sex to keep (inf)",
    )
    parser.add_argument(
        "-N",
        "--num_files",
        type=int,
        default=1,
        help="number of required files (inf)",
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

    stats_f, stats_m = {}, {}
    for spk_id, files in stats.items():
        if len(files) < args.num_files:
            logger.debug("Skipping speaker %s with %d files", spk_id, len(files))
            continue
        fpath = next(iter(stats[spk_id].keys()))
        name = fpath.split(os.sep)[0]
        is_female = name.split("-")[0][-1] == "á"
        if is_female:
            stats_f[spk_id] = files
            logger.debug("Speaker %s (%s) is a female", name, spk_id)
        else:
            stats_m[spk_id] = files
            logger.debug("Speaker %s (%s) is a male", name, spk_id)
    logger.info("Remaining female speakers: %d", len(stats_f))
    logger.info("Remaining male speakers: %d", len(stats_m))

    # Reduce speakers to the specified number
    logger.info("Selecting %d random females from %d available", args.num_speakers, len(stats_f))
    stats_f = reduce_speakers(stats_f, args.num_speakers)
    logger.info("Selecting %d random males from %d available", args.num_speakers, len(stats_m))
    stats_m = reduce_speakers(stats_m, args.num_speakers)

    # Combine female and male speakers
    stats = {**stats_f, **stats_m}
    logger.info(
        "Final selection: %d speakers (%d female, %d male)", len(stats), len(stats_f), len(stats_m)
    )

    # Output determination (stdout or file)
    output_target = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    # Write filtered statistics to YAML file
    yaml.dump(stats, output_target, default_flow_style=False, allow_unicode=True, width=999)
    # Close the file if it was opened
    if args.output:
        output_target.close()


if __name__ == "__main__":
    main()
