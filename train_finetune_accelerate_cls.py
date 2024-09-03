#!/usr/bin/env python3
import os.path as osp
import logging
import torch
from accelerate import Accelerator
from styletts2 import StyleTTS2Finetune, logger
import argparse


def main():
    parser = argparse.ArgumentParser(description="StyleTTS2 finetuning")
    parser.add_argument('config', type=str, help='path to config')
    args = parser.parse_args()

    # Init finetuning
    ft = StyleTTS2Finetune(args.config, Accelerator())

    # write logs
    file_handler = logging.FileHandler(osp.join(ft.log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    # Check if pretrained model for the 2nd stage exists
    if not ft.load_pretrained_for_stage2:
        # Load 1st-stage model
        ft.load_stage1_model()

    ft.init_losses()
    ft.prep_optimizer()

    # load models if there is a model for the 2nd stage
    if ft.load_pretrained_for_stage2:
        ft.load_stage2_model()

    torch.cuda.empty_cache()

    print('BERT', ft.optimizer.optimizers['bert'])
    print('decoder', ft.optimizer.optimizers['decoder'])

    ft.init_sampler()
    ft.init_slmadv_loss()
    ft.prep_accelerator()

    # Start finetuning
    ft.finetune()

    # Clean
    ft.shutdown()


if __name__=="__main__":
    main()
