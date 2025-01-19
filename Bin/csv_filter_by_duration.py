#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys

import yaml


# Funkce pro filtrování řádků podle délky zvukového souboru a výpočet celkových délek
def filter_audio_files(input_csv, stats, min_dur=0.0, max_dur=999.0):
    with open(input_csv, encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="|")
        writer = csv.writer(sys.stdout, delimiter="|", lineterminator="\n")

        # Zapsání hlavičky, pokud existuje
        headers = next(reader, None)
        if headers:
            writer.writerow(headers)

        for row in reader:
            file_name = row[0]  # Název souboru je první položka v řádce

            # Přidání přípony .wav, pokud tam není
            if not file_name.lower().endswith(".wav"):
                file_name += ".wav"

            # Get file duration
            dur = stats.get(file_name)

            # Write row if duration is within the specified range
            if dur and min_dur < dur < max_dur:
                writer.writerow(row)


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Filter list of utterances by audio duration.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("inp_csv", default=None, help="input file list in CSV format")
    parser.add_argument(
        "-M", "--max_dur", type=float, default=999.0, help="maximum audio duration (999.0 s)"
    )
    parser.add_argument(
        "-m", "--min_dur", type=float, default=0.0, help="minimum audio duration (0.0 s)"
    )
    parser.add_argument(
        "-s",
        "--stats",
        type=str,
        required=True,
        help="wav stats file in YAML format",
    )
    args = parser.parse_args()

    # Load wav statistics
    with open(args.stats, "r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)

    filter_audio_files(args.inp_csv, stats, args.min_dur, args.max_dur)


if __name__ == "__main__":
    main()
