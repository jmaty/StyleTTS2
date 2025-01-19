#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import sys
import random

import yaml


# Function for limiting the number of audio files per speaker
def limit_spk_audio_files(stats, max_n_files=float("inf")):
    keys = list(stats.keys())
    random.shuffle(keys)
    return {wav: stats[wav] for wav in itertools.islice(keys, max_n_files)}


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Limit speaker files.\n\n
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
        "--max_n_files_per_spk",
        type=int,
        default=float("inf"),
        help="maximum number of audio files per speaker (inf)",
    )
    args = parser.parse_args()

    # Load input YAML file from file or stdin
    input_source = open(args.input, "r", encoding="utf-8") if args.input else sys.stdin
    with input_source as f:
        stats = yaml.safe_load(f)

    filtered_stats = {}  # Initialize empty dictionary
    # Iterate over speakers
    for spk_id in stats.keys():
        filtered_stats[spk_id] = limit_spk_audio_files(stats[spk_id], args.max_n_files_per_spk)

    # Output determination (stdout or file)
    output_target = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    # Write filtered statistics to YAML file
    yaml.dump(
        filtered_stats, output_target, default_flow_style=False, allow_unicode=True, width=999
    )
    # Close the file if it was opened
    if args.output:
        output_target.close()


if __name__ == "__main__":
    main()
