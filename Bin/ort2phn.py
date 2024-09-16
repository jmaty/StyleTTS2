#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import phonemizer

def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Convert ortographic input to StyleTTS2-compatible phonetic output.\n\n
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Input checkpoint
    parser.add_argument(
        "in_file",
        default=None,
        help="Input file.")
    # Input config
    parser.add_argument(
        "-l", "--lang",
        type=str,
        default=None,
        help="language")
    parser.add_argument(
        "-u", "--out_unames",
        type=str,
        default=None,
        help="output utterance names")
    parser.add_argument(
        "-s", "--separator",
        type=str,
        default='|',
        help="CSV separator")
    parser.add_argument(
        "-S", "--spk_id",
        type=str,
        default='0',
        help="Speaker ID")
    parser.add_argument(
        "-r", "--remove_diff_lang",
        action='store_true',
        default=False,
        help="Remove utterances with another language")
    args = parser.parse_args()

    # Define phonemizer
    phn_obj = phonemizer.backend.EspeakBackend(
        language=args.lang,
        preserve_punctuation=True,
        with_stress=True,
        language_switch='remove-utterance' if args.remove_diff_lang else 'remove-flags',
    )

    with open(args.in_file, 'r') as f_inp:
        for line in f_inp:
            # Read line
            line = line.strip()
            if not line:
                continue
            items = line.strip().split(args.separator)
            if len(items) == 1:
                # only text
                uname = None
                utt = items[0]
            elif len(items) == 2:
                # CSV: filename|text
                uname, utt = items[0], items[1]
                if not uname.endswith('.wav'):
                    uname += '.wav'
            else:
                raise RuntimeError(f'Up to 1 separator {args.separator} is expected in line:\n{line}')
            
            try:
                # Write output
                phn = phn_obj.phonemize([utt])[0]

                # Replacemnets
                #---
                if args.lang == 'de':
                    # Error in espeak-ng: UR results in ?? (eg. wurde)
                    # https://github.com/bootphon/phonemizer/issues/141
                    phn = phn.replace('??', 'ʊɐ')
                    phn = phn.replace('1', '')

            except IndexError:
                raise IndexError(f'{utt}')

            if uname:
                print(f'{uname}{args.separator}{phn.strip()}{args.separator}{args.spk_id}')
            else:
                print(f'{phn.strip()}{args.separator}{args.spk_id}')
         
    # unames = []
    # speakers = []
    # utts = []

    # with open(args.in_file, 'r') as f_inp:
    #     for line in f_inp:
    #         items = line.strip().split(args.separator)
    #         if len(items) == 1:
    #             utts.append(line.strip())
    #         elif len(items) == 2:
    #             uname, utt = items[0], items[1]
    #             if not uname.endswith('.wav'):
    #                 uname += '.wav'
    #             unames.append(uname)
    #             utts.append(utt)
    #             speakers.append(args.spk_id)
    #         else:
    #             raise RuntimeError(f'Up to 1 separator {args.separator} is expected in line:\n{line}')
    
    # # Phonemize
    # phns = phn_obj.phonemize(utts)

    # for idx, phn in enumerate(phns):
    #      if unames:
    #         print(f'{unames[idx]}{args.separator}{phn.strip()}{args.separator}{speakers[idx]}')
    #      else:
    #         print(f'{phn.strip()}{args.separator}{args.spk_id}')

    # if args.out_unames:
    #     with open(args.out_unames, 'w') as f_ids:
    #         for uname in unames:
    #             print(f'{uname}', file=f_ids)

if __name__ == "__main__":
    main()
