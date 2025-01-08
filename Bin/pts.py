#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import random as python_random
import re
import sys
from argparse import RawTextHelpFormatter
from collections import OrderedDict

import numpy as np
import torch

# import torchaudio
import yaml

# from utils import recursive_munch
from munch import munchify
from scipy.io.wavfile import write

import models
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from text_utils2 import TextCleaner
from Utils.PLBERT.util import load_plbert


class Synthesizer:
    def __init__(
        self,
        model_path,
        config_path,
        device="cuda",
        log_level=logging.INFO,
    ):
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Load config
        with open(config_path, encoding="utf-8") as f:
            self.config = munchify(yaml.safe_load(f))

        # Load models
        self.text_aligner = models.load_ASR_models(
            self.config.ASR_path, self.config.ASR_config
        )  # text aligner
        self.f0_extractor = models.load_F0_models(self.config.F0_path)  # F0 extractor
        self.plbert = load_plbert(self.config.PLBERT_dir)  # PL-BERT

        self.model_path = model_path
        self.device = device
        self.text_cleaner = None
        self.sampler = None

        self._load_symbols()
        self.logger.info("Number of symbols: %s", {len(self.text_cleaner)})
        assert (
            len(self.text_cleaner) == 81
        ), f"Number of symbols must be 81 but it is {len(self.text_cleaner)}"

        self._build_model()
        self._setup_sampler()

    def _to_eval(self):
        # Set model to eval mode
        _ = [self.model[key].eval() for key in self.model]

    def _to_device(self):
        # Move model to device
        _ = [self.model[key].to(self.device) for key in self.model]

    def _load_symbols(self):
        """Load symbols from symbol dictionary."""
        self.text_cleaner = TextCleaner(
            self.config.data_params.symbol_dict_path, pad=self.config.data_params.pad
        )

    def _setup_sampler(self):
        """Setup diffusion sampler."""
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

    def generate_noise(self):
        """Generate noise

        Returns:
            tensor: Noise for diffusion.
        """
        return torch.randn(1, 1, 256, device=self.device)

    def _build_model(self):
        """Build model."""
        # Build model
        self.model = models.build_model(
            munchify(self.config["model_params"]), self.text_aligner, self.f0_extractor, self.plbert
        )
        self._to_eval()
        self._to_device()
        params = torch.load(self.model_path, map_location="cpu")["net"]  # Load model parameters

        # Hack to cope with model prefix
        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                    self.logger.debug("%s loaded", key)
                except Exception:
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # Reload fixed params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
                    self.logger.debug("%s loaded and fixed", key)

            else:
                self.logger.warning("Key %s not found in the model parameters.", key)

        self._to_eval()  # Set model to eval mode

    def synthesize(
        self,
        ph_strings,
        diffusion_steps=5,
        embedding_scale=1,
        alpha=0.7,
        fix_noise=False,
    ):
        """Synthesize speech from phonetic strings.

        Args:
            ph_strings (list): Phonetic strings (made up from phonetic sentences).
            diffusion_steps (int, optional): Number of diffusion steps. Defaults to 5.
            embedding_scale (int, optional): Embedding scale. Defaults to 1.
            alpha (float, optional): Weight for convex combination of current and previous styles. Defaults to 0.7.
            fix_noise (bool, optional): Whether to fix noise across sentences. Defaults to False.

        Returns:
            list: Generated waveforms.
        """
        # Initialize previous style and wavs
        wavs = []
        s_prev = None

        # Iterate over phonetic strings (lines in the input phonetic file)
        for ph_string in ph_strings:
            # Use the same noise within a phonetic string (one phonetic line) or
            # generate new noise for each sentence (None)
            noise = self.generate_noise() if fix_noise else None

            # Iterate over sentences
            for ph_sent in re.split(r"[.!?]", ph_string):
                if not ph_sent.strip():  # skip empty phonetic string
                    continue

                # add padding and tokenize phonetic sentence
                ph_ids = [0] + self.text_cleaner(ph_sent)

                # Generate wav
                wav, s_prev = self._inference(
                    ph_ids,
                    noise=noise,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                    s_prev=s_prev,
                    alpha=alpha,
                )
            # Collect wavs
            wavs.append(wav)

        return wavs

    def _inference(
        self, ph_ids, noise=None, diffusion_steps=5, embedding_scale=1, s_prev=None, alpha=0.7
    ):
        """_summary_

        Args:
            ph_ids (list): Phoneme IDs
            noise (tensor, optional): Noise for diffusion. Defaults to None.
            diffusion_steps (int, optional): Number of diffusion steps. Defaults to 5.
            embedding_scale (int, optional): Embedding scale. Defaults to 1.
            s_prev (tensor, optional): Previous sentence style embedding. Defaults to None.
            alpha (float, optional): Weight for convex combination of current and previous styles. Defaults to 0.7.

        Returns:
            tuple(numpy array, tensor): Current sentence waveform and style embedding.
        """
        with torch.no_grad():
            input_lengths = torch.tensor([ph_ids.shape[-1]], dtype=torch.long, device=self.device)
            text_mask = self.length_to_mask(input_lengths)

            t_en = self.model.text_encoder(ph_ids, input_lengths, text_mask)
            bert_dur = self.model.bert(ph_ids, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_curr = self.sampler(
                self.generate_noise() if noise is None else noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

            # Combine styles
            if s_prev is not None:
                s_curr = (
                    alpha * s_curr + (1 - alpha) * s_prev
                )  # convex combination of previous and current styles

            s = s_curr[:, 128:]
            ref = s_curr[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # Encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            f0_pred, n_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(
                t_en @ pred_aln_trg.unsqueeze(0).to(self.device),
                f0_pred,
                n_pred,
                ref.squeeze().unsqueeze(0),
            )

        return out.squeeze().cpu().numpy(), s_curr

    @staticmethod
    def length_to_mask(lengths):
        mask = (
            torch.arange(lengths.max(), device=lengths.device)
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def save_wav(self, wavs, path):
        """Save wavs to a single wav file.

        Args:
            wavs (list): Waveform numpy arrays
            path (string): Output wav file path
        """
        wav = np.concatenate(wavs)
        write(path, self.config.preprocess_params.sr, wav)
        # torchaudio.save(path, torch.tensor(wav).float(), 24000)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# to_mel = torchaudio.transforms.MelSpectrogram(
#     n_mels=80, n_fft=2048, win_length=1200, hop_length=300
# )
# mean, std = -4, 4


# def preprocess(wave):
#     wave_tensor = torch.from_numpy(wave).float()
#     mel_tensor = to_mel(wave_tensor)
#     mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
#     return mel_tensor

# def compute_style(ref_dicts, model):
#     reference_embeddings = {}
#     for key, path in ref_dicts.items():
#         wave, sr = librosa.load(path, sr=24000)
#         audio, _ = librosa.effects.trim(wave, top_db=30)
#         if sr != 24000:
#             audio = librosa.resample(audio, sr, 24000)
#         mel_tensor = preprocess(audio).to(DEVICE)

#         with torch.no_grad():
#             ref = model.style_encoder(mel_tensor.unsqueeze(1))
#         reference_embeddings[key] = (ref.squeeze(1), audio)

#     return reference_embeddings


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Synthesize speech from text file.\n\n"""
        """
    # Example Runs:

    - Simple usage

    ```
    $ ./pts.py --model=path/to/model.pth --config=path/to/config.yaml --out_path output/path/speech.wav
    ```
    """,
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
    parser.add_argument("--model", type=str, default=None, help="Path to model file.")
    parser.add_argument("--config", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--out_path",
        type=str,
        default="./out.wav",
        help="Output wav file path.",
    )
    parser.add_argument(
        "-n",
        "--fixed_noise",
        action="store_true",
        help="Fix noise across sentences. Cancel with a newline. Default=False.",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--diffusion_steps",
        type=float,
        help="Diffusion steps. Default=5.",
        default=5.0,
    )
    parser.add_argument(
        "-e",
        "--embedding_scale",
        type=float,
        help="Embedding scale. Default=1.",
        default=1.0,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Weight for convex combination of current and previous styles. Default=0.7",
        default=0.7,
    )
    parser.add_argument("--use_cuda", action="store_true", help="Run model on CUDA.", default=False)
    parser.add_argument(
        "-r",
        "--random-seed",
        type=int,
        help="Random seed. None means no seed. Default=None.",
        default=None,
    )
    parser.add_argument(
        "-D", "--debug", type=str.upper, help="Set debug level. Default=INFO", default="INFO"
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)-10s %(message)s",
        stream=sys.stdout,
        level=args.debug,
    )
    logger = logging.getLogger("__name__")

    # Set random seed if specified
    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    # Set device
    device = "cuda" if args.use_cuda else "cpu"
    logger.debug("Device: %s", device)

    # Define synthesizer
    synth = Synthesizer(args.model, args.config, device=device, log_level=args.debug)

    # Synthesize speech
    with args.ifile as f:
        wavs = synth.synthesize(
            f.readlines(),
            args.diffusion_steps,
            args.embedding_scale,
            args.alpha,
            args.fix_noise,
        )

    # Save wavs as a single file
    synth.save_wav(wavs, args.out_path)


if __name__ == "__main__":
    main()
