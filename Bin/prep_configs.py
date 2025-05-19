#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path as osp
from argparse import RawTextHelpFormatter

import yaml
from munch import munchify


# Floats are not written in scientific notation when storing to YAML
def float_representer(representer, data):
    value = "{0:.15f}".format(data).rstrip("0")
    return representer.represent_scalar("tag:yaml.org,2002:float", value)


# Load config
def load_config(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return munchify(yaml.safe_load(f))


# Save config
def save_config(config, filename):
    config = config.toDict() if hasattr(config, "toDict") else config
    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


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
    # Parser for config
    args = parser.parse_args()

    yaml.add_representer(float, float_representer)

    settings = load_config(args.settings)  # Load settings
    config = load_config(args.config)  # Load config
    cfg_name, cfg_ext = osp.splitext(osp.basename(args.config))  # Get config name, extension
    out_dir = osp.dirname(args.settings)  # Output directory

    # Stage 1: epochs = 0 - TMA
    config_out = config.copy()
    config_out.epochs.stage1 = config.epochs.tma  # Set TMA epoch as the last epoch
    config_out.batch_size = settings.start1.batch_size
    config_out.grad_accum_steps = settings.start1.grad_accum_steps
    config_out.max_len = settings.start1.max_len
    config_out.data_params.train_data = settings.start1.train_data
    out_file = osp.join(out_dir, f"{cfg_name}{settings.start1.label}{cfg_ext}")
    save_config(config_out, out_file)

    # Stage 1: epochs = TMA - end
    config_out = config.copy()
    config_out.epochs.stage1 = config.epochs.stage1  # Set TMA epoch as the last epoch
    config_out.batch_size = settings.tma.batch_size
    config_out.grad_accum_steps = settings.tma.grad_accum_steps
    config_out.max_len = settings.tma.max_len
    config_out.data_params.train_data = settings.tma.train_data
    pretrained_model = osp.join(out_dir, f"epoch_1st_{config.epochs.tma-1:05}.pth")
    config_out.pretrained_model = pretrained_model
    out_file = osp.join(out_dir, f"{cfg_name}{settings.tma.label}{cfg_ext}")
    save_config(config_out, out_file)

    # Stage 2: epochs = 0 - diffusion
    config_out = config.copy()
    config_out.epochs.stage2 = config.epochs.diff  # Set diffusion epoch as the last epoch
    config_out.batch_size = settings.start2.batch_size
    config_out.max_len = settings.start2.max_len
    config_out.data_params.train_data = settings.start2.train_data
    out_file = osp.join(out_dir, f"{cfg_name}{settings.start2.label}{cfg_ext}")
    save_config(config_out, out_file)

    # Stage 2: epochs = diffusion - joint training
    config_out = config.copy()
    config_out.epochs.stage2 = config.epochs.joint  # Set joint epoch as the last epoch
    config_out.batch_size = settings.diff.batch_size
    config_out.max_len = settings.diff.max_len
    config_out.data_params.train_data = settings.diff.train_data
    pretrained_model = osp.join(out_dir, f"epoch_2nd_{config.epochs.diff-1:05}.pth")
    config_out.pretrained_model = pretrained_model
    out_file = osp.join(out_dir, f"{cfg_name}{settings.diff.label}{cfg_ext}")
    save_config(config_out, out_file)

    # Stage 2: epochs =  joint training - end
    config_out = config.copy()
    config_out.epochs.stage2 = config.epochs.stage2  # Set joint epoch as the last epoch
    config_out.batch_size = settings.joint.batch_size
    config_out.max_len = settings.joint.max_len
    config_out.data_params.train_data = settings.joint.train_data
    pretrained_model = osp.join(out_dir, f"epoch_2nd_{config.epochs.joint-1:05}.pth")
    config_out.pretrained_model = pretrained_model
    out_file = osp.join(out_dir, f"{cfg_name}{settings.joint.label}{cfg_ext}")
    save_config(config_out, out_file)


if __name__ == "__main__":
    main()
