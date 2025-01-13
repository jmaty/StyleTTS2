#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys

import librosa
from tqdm import tqdm
import yaml


# Funkce na získání délky zvukového souboru v sekundách pomocí librosa
def get_audio_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Načtení zvuku s původní vzorkovací frekvencí
    duration = librosa.get_duration(y=y, sr=sr)
    return duration


# Funkce pro filtrování řádků podle délky zvukového souboru a výpočet celkových délek
def compute_stats(input_csv, audio_directory):
    stats = {}
    with open(input_csv, encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="|")

        for row in tqdm(reader, desc="Computing stats", unit=" rows"):
            file_name = row[0]  # Název souboru je první položka v řádce

            # Přidání přípony .wav, pokud tam není
            if not file_name.lower().endswith(".wav"):
                file_name += ".wav"

            audio_file_path = os.path.join(audio_directory, file_name)

            # Pokud zvukový soubor existuje, zjisti jeho délku
            if os.path.isfile(audio_file_path):
                dur = get_audio_duration(audio_file_path)
                stats[file_name] = dur
    return stats


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Compute wav statistics.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("inp_csv", default=None, help="input file list in CSV format")
    parser.add_argument(
        "-a",
        "--audio_dir",
        type=str,
        default="./wavs",
        required=True,
        help="directory with audio files",
    )
    args = parser.parse_args()

    # Compute wabeform statistics
    stats = compute_stats(args.inp_csv, args.audio_dir)
    # Write statistics to stdout as YAML
    yaml.dump(stats, sys.stdout, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
