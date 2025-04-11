#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
from argparse import RawTextHelpFormatter

from Modules.pts import PTS, set_random_seed


def main():
    """
    # Example Runs:

    - Basic synthesis from text file:
        $ ./pts.py input.txt config.yml /model.pth --out_path=output.wav

    - Synthesis from stdin:
        $ echo "tohle je skouSka." | ./pts.py config.yml model.pth

    - Voice cloning from reference speaker (for a multi-speaker model):
        $ ./pts.py input.txt config.yml model.pth --ref_spk=reference_speaker.wav

    - Adjusting synthesis parameters:
        $ ./pts.py input.txt config.yml model.pth --diffusion_steps=5 --embedding_scale=1.2 --alpha=0.1 --beta=0.6

    - Using fixed noise for more consistent output:
        $ ./pts.py input.txt config.yml model.pth --use_glob_noise --random-seed=42
    """
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Synthesize speech from phonetic text file.\n\n""",
        formatter_class=RawTextHelpFormatter,
    )
    # Input text file
    parser.add_argument(
        "ifile",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Text file to generate speech from.",
    )
    parser.add_argument("config", type=str, help="Path to model config file.")
    parser.add_argument("model", type=str, default=None, help="Path to model file.")
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./out.wav",
        help="Output wav file path.",
    )
    parser.add_argument(
        "-g",
        "--use_glob_noise",
        action="store_true",
        help="Use the same noise for the entire documents. Default=False.",
        default=False,
    )
    parser.add_argument(
        "-n",
        "--fix_noise_in_ph_string",
        action="store_true",
        help="Fix noise across sentences in phonetic string. Cancel with a newline. Default=False.",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--diffusion_steps",
        type=float,
        help="Diffusion steps. Default=10",
        default=10,
    )
    parser.add_argument(
        "-e",
        "--embedding_scale",
        type=float,
        help="Embedding scale. Default=1.",
        default=1.0,
    )
    parser.add_argument(
        "-t",
        "--style_combination",
        type=float,
        help="Weight for convex combination of current and previous styles. Default=0.7",
        default=0.7,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Weight for speech timbre. Default=0.3",
        default=0.3,
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        help="Weight for speech prosody. Default=0.7",
        default=0.7,
    )
    parser.add_argument(
        "-r",
        "--speech_rate",
        type=float,
        help="Speech rate. Default=1.0",
        default=1.0,
    )
    parser.add_argument(
        "-s",
        "--random-seed",
        type=int,
        help="Random seed. None means no seed. Default=None.",
        default=None,
    )
    parser.add_argument(
        "-R",
        "--ref_spk",
        type=str,
        default=None,
        help="Path to reference speaker wav for voice cloning. Default=None.",
    )
    parser.add_argument(
        "-L", "--loglevel", type=str.upper, help="Set logging level. Default=INFO", default="INFO"
    )
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.loglevel, logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        level=log_level,
    )
    # formatter = logging.Formatter(
    #             "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    #         )
    logger = logging.getLogger(__name__)

    # Set random seed if specified
    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    pts = PTS(
        args.config,
        args.model,
        t=args.style_combination,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale,
        speech_rate=args.speech_rate,
        use_glob_noise=args.use_glob_noise,
        fix_noise_in_ph_string=args.fix_noise_in_ph_string,
        log_level=log_level,
    )

    # Synthesize speech from phonetic text file
    with args.ifile as f:
        lines = [line.strip() for line in f]
        wavs = pts(lines, args.ref_spk)

    # Save wavs as a single file
    pts.save_wav(wavs, args.out_path)
    logger.debug("Wav file saved to %s", args.out_path)


if __name__ == "__main__":
    main()
