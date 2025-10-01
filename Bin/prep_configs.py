#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import os.path as osp
from argparse import RawTextHelpFormatter

import yaml


# Floats are not written in scientific notation when storing to YAML
def float_representer(representer, data):
    value = f"{data:.15f}".rstrip("0")
    return representer.represent_scalar("tag:yaml.org,2002:float", value)


# Load config
def load_config(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Save config
def save_config(config, filename):
    config = config.toDict() if hasattr(config, "toDict") else config
    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Create config files for various training stages.\n\n
        """,
        formatter_class=RawTextHelpFormatter,
    )
    # Input checkpoint
    parser.add_argument("settings", help="Input settings file")
    # Output checkpoint
    parser.add_argument("config", help="Input config file")
    parser.add_argument("-s", "--simple", action="store_true", help="Use simple config generation")
    # Parser for config
    args = parser.parse_args()

    yaml.add_representer(float, float_representer)

    settings = load_config(args.settings)  # Load settings
    config = load_config(args.config)  # Load config
    cfg_name, cfg_ext = osp.splitext(osp.basename(args.config))  # Get config name, extension
    out_dir = osp.dirname(args.settings)  # Output directory

    if args.simple:
        # --- Simple 2-stage config generation ---

        # Stage 1
        config_out = copy.deepcopy(config)
        config_out["epochs"]["stage1"] = settings["stage1"]["epochs"]
        config_out["batch_size"] = settings["stage1"]["batch_size"]
        config_out["grad_accum_steps"] = settings["stage1"]["grad_accum_steps"]
        config_out["max_len"] = settings["stage1"]["max_len"]
        config_out["data_params"]["train_data"] = settings["stage1"]["train_data"]
        config_out["data_params"]["val_data"] = settings["stage1"]["val_data"]
        config_out["second_stage_load_pretrained"] = False
        config_out["label"] = settings["stage1"]["label"]
        out_file = osp.join(out_dir, f"{cfg_name}{settings['stage1']['label']}{cfg_ext}")
        save_config(config_out, out_file)

        # Stage 2
        config_out = copy.deepcopy(config)
        config_out["epochs"]["stage2"] = settings["stage2"]["epochs"]
        config_out["batch_size"] = settings["stage2"]["batch_size"]
        config_out["max_len"] = settings["stage2"]["max_len"]
        config_out["data_params"]["train_data"] = settings["stage2"]["train_data"]
        config_out["data_params"]["val_data"] = settings["stage2"]["val_data"]
        config_out["second_stage_load_pretrained"] = True
        config_out["label"] = settings["stage2"]["label"]
        out_file = osp.join(out_dir, f"{cfg_name}{settings['stage2']['label']}{cfg_ext}")
        save_config(config_out, out_file)

    else:
        # --- Full 5-phase config generation ---

        # Stage 1: epochs = 0 - TMA
        config_out = copy.deepcopy(config)
        # Set TMA epoch as the last epoch
        config_out["epochs"]["stage1"] = settings["start1"]["epochs"]
        config_out["batch_size"] = settings["start1"]["batch_size"]
        config_out["grad_accum_steps"] = settings["start1"]["grad_accum_steps"]
        config_out["max_len"] = settings["start1"]["max_len"]
        config_out["data_params"]["train_data"] = settings["start1"]["train_data"]
        config_out["second_stage_load_pretrained"] = False
        config_out["label"] = settings["start1"]["label"]
        out_file = osp.join(out_dir, f"{cfg_name}{settings['start1']['label']}{cfg_ext}")
        save_config(config_out, out_file)

        # Stage 1: epochs = TMA - end
        config_out = copy.deepcopy(config)
        # Set TMA epoch as the last epoch
        config_out["epochs"]["stage1"] = settings["tma"]["epochs"]
        config_out["batch_size"] = settings["tma"]["batch_size"]
        config_out["grad_accum_steps"] = settings["tma"]["grad_accum_steps"]
        config_out["max_len"] = settings["tma"]["max_len"]
        config_out["data_params"]["train_data"] = settings["tma"]["train_data"]
        pretrained_model = osp.join(out_dir, f"epoch_1st_{config['epochs']['tma']-1:05}.pth")
        config_out["pretrained_model"] = pretrained_model
        config_out["second_stage_load_pretrained"] = False
        config_out["label"] = settings["tma"]["label"]
        out_file = osp.join(out_dir, f"{cfg_name}{settings['tma']['label']}{cfg_ext}")
        save_config(config_out, out_file)

        # Stage 2: epochs = 0 - diffusion
        config_out = copy.deepcopy(config)
        # Set diffusion epoch as the last epoch
        config_out["epochs"]["stage2"] = settings["start2"]["epochs"]
        config_out["batch_size"] = settings["start2"]["batch_size"]
        config_out["max_len"] = settings["start2"]["max_len"]
        config_out["data_params"]["train_data"] = settings["start2"]["train_data"]
        config_out["second_stage_load_pretrained"] = True
        config_out["label"] = settings["start2"]["label"]
        out_file = osp.join(out_dir, f"{cfg_name}{settings['start2']['label']}{cfg_ext}")
        save_config(config_out, out_file)

        # Stage 2: epochs = diffusion - joint training
        config_out = copy.deepcopy(config)
        # Set joint epoch as the last epoch
        config_out["epochs"]["stage2"] = settings["diff"]["epochs"]
        config_out["batch_size"] = settings["diff"]["batch_size"]
        config_out["max_len"] = settings["diff"]["max_len"]
        config_out["data_params"]["train_data"] = settings["diff"]["train_data"]
        config_out["second_stage_load_pretrained"] = "true"
        pretrained_model = osp.join(out_dir, f"epoch_2nd_{config['epochs']['diff']-1:05}.pth")
        config_out["pretrained_model"] = pretrained_model
        config_out["second_stage_load_pretrained"] = True
        config_out["label"] = settings["diff"]["label"]
        out_file = osp.join(out_dir, f"{cfg_name}{settings['diff']['label']}{cfg_ext}")
        save_config(config_out, out_file)

        # Stage 2: epochs =  joint training - end
        config_out = copy.deepcopy(config)
        config_out["epochs"]["stage2"] = settings["joint"]["epochs"]  # Set joint epoch = last epoch
        config_out["batch_size"] = settings["joint"]["batch_size"]
        config_out["max_len"] = settings["joint"]["max_len"]
        config_out["data_params"]["train_data"] = settings["joint"]["train_data"]
        pretrained_model = osp.join(out_dir, f"epoch_2nd_{config['epochs']['joint']-1:05}.pth")
        config_out["pretrained_model"] = pretrained_model
        config_out["second_stage_load_pretrained"] = True
        config_out["label"] = settings["joint"]["label"]
        out_file = osp.join(out_dir, f"{cfg_name}{settings['joint']['label']}{cfg_ext}")
        save_config(config_out, out_file)


if __name__ == "__main__":
    main()
