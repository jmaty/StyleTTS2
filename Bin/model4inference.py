#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import argparse
from argparse import RawTextHelpFormatter
from munch import munchify

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path, PurePath
import torch

from models import load_ASR_models, load_F0_models, build_model
from Utils.PLBERT.util import load_plbert


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Reduce model for inference.\n\n
        """,
        formatter_class=RawTextHelpFormatter,
    )
    # Input checkpoint
    parser.add_argument("inp_model", default=None, help="Input model.")
    # Output checkpoint
    parser.add_argument("out_model", default=None, help="Output model.")
    # Input config
    parser.add_argument("-c", "--inp_config", type=str, default=None, help="Path to input config")
    # Parser for config
    args = parser.parse_args()

    # Load config
    with open(args.inp_config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # # Load pretrained models
    # text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])  # Text aligner
    # pitch_extractor = load_F0_models(config["F0_path"])  # F0 extractor
    # plbert = load_plbert(config["PLBERT_dir"])  # PLBERT

    # # Build model
    # model = build_model(munchify(config["model_params"]), text_aligner, pitch_extractor, plbert)
    # _ = [model[key].eval() for key in model]  # Set model to eval mode

    # Define multispeaker
    multispeaker = config["model_params"].get("multispeaker", False)

    # # Reduce the final model size by removing redundant components
    # del model["net"]["mpd"]
    # del model["net"]["msd"]
    # del model["net"]["wd"]
    # del model["net"]["text_aligner"]
    # del model["net"]["pitch_extractor"]
    # if not multispeaker:
    #     del model["net"]["style_encoder"]
    #     del model["net"]["predictor_encoder"]

    # Load params
    params = torch.load(args.inp_model, map_location="cpu")["net"]
    del params["mpd"]
    del params["msd"]
    del params["wd"]
    del params["text_aligner"]
    del params["pitch_extractor"]
    if not multispeaker:
        del params["style_encoder"]
        del params["predictor_encoder"]

    # # Load the model
    # model = {key: model[key].load_state_dict(params[key]) for key in model if key in params}

    # # Save the reduced model
    # state_dict = {key: model[key].state_dict() for key in model}
    # torch.save(state_dict, args.out_model)

    torch.save(params, args.out_model)


if __name__ == "__main__":
    main()
