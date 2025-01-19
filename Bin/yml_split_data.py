#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random

import yaml


# Split data to train and validation sets
def split_data(data_stats, train, valid, wavs_per_spk=1):
    # Init sets
    train_set = {}
    valid_set = {}

    # Iterate over speakers
    for spk_id, wavs in data_stats.items():
        all_wavs = list(wavs)
        valid_wavs = []
        if valid:
            # Randomly select wavs_per_spk wavs for validation
            valid_wavs = random.sample(all_wavs, min(wavs_per_spk, len(all_wavs)))
            # Add selected wavs to the validation set
            valid_set[spk_id] = wavs_to_set(spk_id, valid_wavs, data_stats)
        # Add remaining wavs to the training set
        if train:
            train_wavs = list(set(all_wavs) - set(valid_wavs))
            train_set[spk_id] = wavs_to_set(spk_id, train_wavs, data_stats)
    # Return the sets
    return train_set, valid_set


# Add wavs to the set
def wavs_to_set(spk_id, wavs, data_stats):
    return {wav: data_stats[spk_id][wav] for wav in wavs}


# Write set to YAML file
def write_set_yaml(dataset, filename):
    with open(filename, "w", encoding="utf-8") as yamlfile:
        yaml.dump(dataset, yamlfile, default_flow_style=False, allow_unicode=True, width=999)


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Split data to sets.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("data_stats", default=None, help="input data stats in YAML format")
    parser.add_argument("-t", "--train", default=None, help="output train set in YAML format")
    parser.add_argument(
        "-v", "--valid", default=None, help="output validation/test set in YAML format"
    )
    parser.add_argument(
        "-s",
        "--wavs_per_spk",
        type=int,
        default=1,
        help="number of wavs per speaker. Default is 1",
    )
    args = parser.parse_args()

    # Load data statistics
    with open(args.data_stats, "r", encoding="utf-8") as infile:
        data_stats = yaml.safe_load(infile)

    # Compute wabeform statistics
    train_set, valid_set = split_data(data_stats, args.train, args.valid, args.wavs_per_spk)

    # Write sets to CSV files
    if args.train:
        write_set_yaml(train_set, args.train)
    if args.valid:
        write_set_yaml(valid_set, args.valid)


if __name__ == "__main__":
    main()
