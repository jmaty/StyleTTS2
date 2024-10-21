#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import os
import librosa
import numpy as np
import argparse


# Funkce na získání délky zvukového souboru v sekundách pomocí librosa
def get_audio_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Načtení zvuku s původní vzorkovací frekvencí
    duration = librosa.get_duration(y=y, sr=sr)
    return duration


# Funkce pro filtrování řádků podle délky zvukového souboru a výpočet celkových délek
def filter_audio_files(input_csv, audio_directory, min_dur=0.0, max_dur=999.0):
    with open(input_csv, encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='|')
        writer = csv.writer(sys.stdout, delimiter='|', lineterminator='\n')
        
        # Zapsání hlavičky, pokud existuje
        headers = next(reader, None)
        if headers:
            writer.writerow(headers)

        for row in reader:
            file_name = row[0]  # Název souboru je první položka v řádce

            # Přidání přípony .wav, pokud tam není
            if not file_name.lower().endswith('.wav'):
                file_name += '.wav'
            
            audio_file_path = os.path.join(audio_directory, file_name)
            
            # Pokud zvukový soubor existuje, zjisti jeho délku
            if os.path.isfile(audio_file_path):
                dur = get_audio_duration(audio_file_path)

                # Pokud délka je menší než zadaná hodnota, zapiš řádek do výstupního souboru
                if dur < max_dur and dur > min_dur:
                    writer.writerow(row)


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Filter list of utterances by audio duration.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "inp_csv",
        default=None,
        help="input file list in CSV format")
    parser.add_argument(
        "-M", "--max_dur",
        type=float,
        default=999.0,
        help="maximum audio duration (999.0 s)")
    parser.add_argument(
        "-m", "--min_dur",
        type=float,
        default=0.0,
        help="minimum audio duration (0.0 s)")
    parser.add_argument(
        "-a", "--audio_dir",
        type=str,
        default='.wavs',
        required=True,
        help="minimum audio duration (0.0 s)")
    args = parser.parse_args()

    filter_audio_files(args.inp_csv, args.audio_dir, args.min_dur, args.max_dur)

if __name__ == "__main__":
    main()
