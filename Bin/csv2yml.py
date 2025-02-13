#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys
from collections import defaultdict

import librosa
import yaml
from tqdm import tqdm


# Convert defaultdict to dict recursively
def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


# Function for getting audio file duration in seconds using librosa
def get_audio_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Načtení zvuku s původní vzorkovací frekvencí
    duration = librosa.get_duration(y=y, sr=sr)
    return duration


# Function for filtering rows by audio file duration and computing total durations
def compute_stats(input_csv, audio_directory, spk_id_separator=None, default_spk_id=0):
    stats = defaultdict(lambda: defaultdict(dict))
    spk_names = {}
    spk_id = 0

    # Open CSV file
    with open(input_csv, encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="|")

        for row in tqdm(reader, desc="Computing stats", unit=" rows"):
            # If the row has less than 3 columns,
            # use default speaker ID or extract it from the file name
            if len(row) < 3:
                if spk_id_separator is not None:
                    # Extract speaker name from the file name
                    spk_name = row[0].split(spk_id_separator)[0]
                    if spk_name in spk_names:
                        id_ = spk_names[spk_name]  # Get speaker ID from the dictionary
                    else:
                        # Add new speaker to the dictionary
                        spk_names[spk_name] = spk_id
                        id_ = spk_id
                        spk_id += 1
                else:
                    id_ = default_spk_id  # Use default speaker ID
            else:
                id_ = int(row[2])  # Extract speaker ID from the row's 3rd column

            # # Extract speaker ID from the row's 3rd column or use default_spk_id
            # spk_id = default_spk_id if len(row) < 3 else int(row[2])
            file_name = row[0]  # the first column is the file name
            text = row[1]  # the second column is the text

            # Add .wav extension if not present
            if not file_name.lower().endswith(".wav"):
                file_name += ".wav"

            audio_file_path = os.path.join(audio_directory, file_name)

            # If the audio file exists, get its duration
            if os.path.isfile(audio_file_path):
                dur = get_audio_duration(audio_file_path)
            else:
                print(f"File {audio_file_path} not found.", file=sys.stderr)
                dur = None

            # Store the duration in the stats dictionary
            stats[id_][file_name]["text"] = text
            stats[id_][file_name]["duration"] = dur

    return stats


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Compute statistics.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("inp_csv", default=None, help="input file list in CSV format")
    parser.add_argument("out_yaml", default=None, help="output file stats in YAML format")
    parser.add_argument(
        "-a",
        "--audio_dir",
        type=str,
        default="./wavs",
        required=True,
        help="directory with audio files",
    )
    parser.add_argument(
        "-s",
        "--spk_id",
        type=int,
        default=0,
        help="defaut speaker ID. Default is 0",
    )
    parser.add_argument(
        "-S",
        "--spk_id_separator",
        type=str,
        default=None,
        help="separator for speaker ID in the file name. Default is None",
    )
    args = parser.parse_args()

    # Compute waveform statistics
    stats = compute_stats(args.inp_csv, args.audio_dir, args.spk_id_separator, args.spk_id)
    # Convert defaultdict to dict
    stats = defaultdict_to_dict(stats)

    # Write statistics to stdout as YAML
    with open(args.out_yaml, "w", encoding="utf-8") as yamlfile:
        yaml.dump(stats, yamlfile, default_flow_style=False, allow_unicode=True, width=999)


if __name__ == "__main__":
    main()
