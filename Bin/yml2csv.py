#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import random

import yaml


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Convert YAML to CSV.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("yaml", default=None, help="input data stats in YAML format")
    parser.add_argument("csv", default=None, help="output dta in CSV format")
    args = parser.parse_args()

    # Load data statistics
    with open(args.yaml, "r", encoding="utf-8") as infile:
        dataset = yaml.safe_load(infile)

    # Write set to CSV file
    with open(args.csv, "w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        # Iterate over speakers
        for spk_id in dataset.keys():
            # Iterate over wavs of given speaker
            for wav, data in dataset[spk_id].items():
                writer.writerow([wav, data["text"], spk_id])

if __name__ == "__main__":
    main()
