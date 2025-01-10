# coding: utf-8
import logging
import os.path as osp
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {"n_fft": 2048, "win_length": 1200, "hop_length": 300}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list,
        root_path,
        text_cleaner,
        preprocess_text_fn=None,
        data_augmentation=False,
        validation=False,
        OOD_data="Data/OOD_texts.txt",
        **kwargs,
    ):

        # spect_params = SPECT_PARAMS     # TODO: not reading from config!?
        # mel_params = MEL_PARAMS         # TODO: not reading from config!?

        self.mean, self.std = mean, std
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.root_path = root_path  # Set up path to waveform directory
        self._preprocess_text_fn = preprocess_text_fn

        # Silence duration at the beginning and end of the waveform (in samples)
        self.sr = kwargs.get("sr", 24000)
        self.min_length = kwargs.get("min_length", 50)
        self.max_length = kwargs.get("max_length", 512)
        self.silence_beg = kwargs.get("silence_beg", 4800)
        self.silence_end = kwargs.get("silence_end", 4800)

        # Set up text cleaner for phone ID encoding and padding ID
        self.text_cleaner = text_cleaner

        # Load texts from the input data list
        self.data_list = self._load_texts(data_list)

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        # Load Out-of-distribution texts
        self.ptexts = self._load_ood_texts(OOD_data)

    def _load_texts(self, data_list):
        """
        Load texts from a list of data lines.
        Also load speaker IDs if available and path to the corresponding waveforms.

        Args:
            data_list (list): List of data lines.

        Returns:
            list: (waveform_path, list of OOD phone IDs, speaker id).
        """
        ph_texts = []  # Initialize the list of data

        # Read input list of text data lines delimited by "|" and ignore too long lines
        for l in data_list:
            data = l.strip().split("|")  # Remove leading/trailing whitespaces and split the string
            # Ensure data has at least two elements
            assert len(data) in (2, 3), f"Invalid data format, 2-3 elements expected: {l}"
            # data[:, 1] is phonetic string
            # Check if the length of data[1] exceeds `max_length` characters (typically 512)
            # -2: padding at the start/end of the phonetic string
            if len(self.text_cleaner.add_spaces_around_punctuation(data[1])) > self.max_length - 2:
                logger.warning(
                    "Skipping %s: phone length %d > %d",
                    data[0],
                    len(self.text_cleaner.add_spaces_around_punctuation(data[1])),
                    self.max_length - 2,
                )
                continue  # Skip this item
            ph_texts.append(data if len(data) == 3 else data + ["0"])
        return ph_texts

    def _load_ood_texts(self, ood_file):
        """
        Load out-of-distribution (OOD) texts from a specified file.

        Args:
            ood_file (str): Path to the file containing OOD texts.

        Returns:
            list: OOD phone IDs per text line.
        """
        # Load OOD texts from the specified file
        with open(ood_file, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
        # Extract the index of text part (either 0 or 1) based on
        # if the first element contains '.wav'
        idx = 1 if ".wav" in text_lines[0].split("|")[0] else 0
        # Read the text parts from the lines and filter out lines
        # with text length not in `<min_length, max_length>`)
        # (to avoid incompatibility with ALBERT's input size and ensure minimum length)
        ph_texts = []
        for t in text_lines:
            parts = t.split("|")
            # Length of the phonetic string after adding spaces around punctuation
            ph_string_len = len(self.text_cleaner.add_spaces_around_punctuation(parts[idx]))
            # Check if the length of the phonetic string is within the specified range
            if self.min_length <= ph_string_len <= self.max_length - 2:
                ph_texts.append(parts[idx])
        return ph_texts

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]  # [wavfile, phone IDs, speaker_id]
        # Load the waveform, phonetic string, and speaker ID
        wave, text_tensor, speaker_id = self._load_tensor(data)

        mel_tensor = preprocess(wave).squeeze()

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        # Ensure feature tensor with even length
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]

        # get reference sample of max length `self.max_mel_length` (192)
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])  # ref_label is speaker ID

        # Randomly select a phonetic sentence from the OOD texts
        ref_ph_string = self.ptexts[np.random.randint(0, len(self.ptexts) - 1)]
        # Encode phonetic string as a list of phoneme IDs with padding
        ref_ph_ids = self.text_cleaner(ref_ph_string, pad=True)

        return (
            speaker_id,  # speaker ID
            acoustic_feature,  # mel spectrogram of input waveform
            text_tensor,  # phoneme IDs of input text
            torch.LongTensor(ref_ph_ids),  # phone IDs of OOD text with padding
            ref_mel_tensor,  # reference mel vector of the given speaker
            ref_label,  # reference speaker ID
            data[0],  # wavfile
            wave,  # raw waveform
        )

    def _load_tensor(self, data):
        wave_path, ph_string, speaker_id = data
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            logger.warning("%s: sampling rate is %d, resampling to %d", wave_path, sr, self.sr)
        # Add padding to the waveform
        wave = np.concatenate(
            [np.zeros([self.silence_beg]), wave, np.zeros([self.silence_end])], axis=0
        )

        return (
            wave,  # raw waveform
            torch.LongTensor(self.text_cleaner(ph_string, pad=True)),  # phone IDs with padding
            int(speaker_id),  # speaker ID
        )

    def _load_data(self, data):
        wave, _, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            # Randomly crop a segment of the mel spectrogram with length self.max_mel_length
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start : random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # Sort batch by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        # ref_labels = torch.zeros((batch_size)).long()
        paths = ["" for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]

        # Rearrange batch data according to mel length
        for bid, (label, mel, text, ref_text, ref_mel, _, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            waves[bid] = wave
            # ref_labels[bid] = ref_label

        return (
            waves,  # raw waveforms
            texts,  # input phoneme IDs
            input_lengths,  # input phoneme lengths
            ref_texts,  # OOD text phoneme IDs
            ref_lengths,  # OOD texts phoneme lengths
            mels,  # mel spectrograms
            output_lengths,  # mel spectrogram lengths
            ref_mels,  # given speaker reference melspectrograms
        )


def build_dataloader(
    path_list,
    root_path,
    text_cleaner,
    preprocess_text_fn=None,
    validation=False,
    OOD_data="Data/OOD_texts.txt",
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config=None,
    dataset_config=None,
):
    collate_config = collate_config or {}
    dataset_config = dataset_config or {}

    dataset = FilePathDataset(
        path_list,
        root_path,
        text_cleaner,
        preprocess_text_fn=preprocess_text_fn,
        OOD_data=OOD_data,
        validation=validation,
        **dataset_config,
    )
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader
