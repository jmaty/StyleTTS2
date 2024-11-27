#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import argparse

def load_symbol_dict(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        symbol_dict = {row[0]: int(row[1]) for row in reader}
    return symbol_dict


# Funkce pro filtrování řádků podle délky zvukového souboru a výpočet celkových délek
def remove_unsupported_phonemes(input_csv, symbol_dict):
    with open(input_csv, encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='|')
        writer = csv.writer(sys.stdout, delimiter='|', lineterminator='\n')

        # Zapsání hlavičky, pokud existuje
        headers = next(reader, None)
        if headers:
            writer.writerow(headers)

        supported_phonemes = symbol_dict.keys()

        for row in reader:
            if all(p in supported_phonemes for p in row[1]):
                writer.writerow(row)

def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Remove lines with unsupported phonemes.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "inp_csv",
        default=None,
        help="input file list in CSV format")
    parser.add_argument(
        "-d", "--symbol_dict",
        type=str,
        help="path to symbol definition file")
    args = parser.parse_args()

    remove_unsupported_phonemes(args.inp_csv, load_symbol_dict(args.symbol_dict))

if __name__ == "__main__":
    main()
