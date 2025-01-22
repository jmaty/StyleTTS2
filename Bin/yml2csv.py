#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys

import yaml


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Convert YAML to CSV.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "yaml",
        nargs="?",
        default=None,
        help="input data stats in YAML format (if not specified, input from stdin)",
    )
    parser.add_argument(
        "csv", nargs="?", help="output file in CSV format (if not specified, output to stdout)"
    )
    args = parser.parse_args()

    # Load input YAML file from file or stdin
    input_source = open(args.yaml, "r", encoding="utf-8") if args.yaml else sys.stdin
    with input_source as f:
        dataset = yaml.safe_load(f)

    # Output determination (stdout or file)
    output_target = open(args.csv, "w", encoding="utf-8") if args.csv else sys.stdout
    # Write set to CSV file
    with output_target as csvfile:
        writer = csv.writer(csvfile, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        # Iterate over speakers
        for spk_id in dataset.keys():
            # Iterate over wavs of given speaker
            for wav, data in dataset[spk_id].items():
                writer.writerow([wav, data["text"], spk_id])


if __name__ == "__main__":
    main()
