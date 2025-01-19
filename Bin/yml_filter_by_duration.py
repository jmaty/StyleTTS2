#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import yaml


# Funkce pro filtrování řádků podle délky zvukového souboru a výpočet celkových délek
def filter_spk_audio_files(stats, min_dur=0.0, max_dur=999.0):
    return {wav: data for wav, data in stats.items() if min_dur < data["duration"] < max_dur}


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Filter utterances by audio duration.\n\n
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
        "-M", "--max_dur", type=float, default=999.0, help="maximum audio duration (999.0 s)"
    )
    parser.add_argument(
        "-m", "--min_dur", type=float, default=0.0, help="minimum audio duration (0.0 s)"
    )
    args = parser.parse_args()

    # Load input YAML file from file or stdin
    input_source = open(args.input, "r", encoding="utf-8") if args.input else sys.stdin
    with input_source as f:
        stats = yaml.safe_load(f)

    filtered_stats = {}  # Initialize empty dictionary
    # Iterate over speakers
    for spk_id in stats.keys():
        filtered_stats[spk_id] = filter_spk_audio_files(stats[spk_id], args.min_dur, args.max_dur)

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
