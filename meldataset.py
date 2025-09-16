# coding: utf-8
import math
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Sampler

from logger import get_logger

logger = get_logger(__name__)

np.random.seed(3407)
random.seed(3407)


def seconds2mel_length(seconds, sample_rate=24000, hop_length=300):
    """Converts seconds to the number of mel frames.
    Args:
        seconds (float): Duration in seconds.
        sample_rate (int): Sample rate of the audio. Defaults to 24000.
        hop_length (int): Hop length for STFT. Defaults to 300.
    Returns:
        int: Number of mel frames corresponding to the given duration.
    """
    # Calculate the number of mel frames: samples / hop_length
    return int(seconds * sample_rate / hop_length)


def samples2mel_length(samples, hop_length):
    """Converts number of samples to the number of mel frames.
    Args:
        samples (int): Number of audio samples.
        hop_length (int): Hop length for STFT. Defaults to 300.
    Returns:
        int: Number of mel frames corresponding to the given number of samples.
    """
    # Calculate the number of mel frames: samples / hop_length
    return int(samples / hop_length)


def mel2samples_length(mel_frames, hop_length):
    """Converts number of mel frames to the number of samples.
    Args:
        mel_frames (int): Number of mel frames.
        hop_length (int): Hop length for STFT. Defaults to 300.
    Returns:
        int: Number of audio samples corresponding to the given number of mel frames.
    """
    # Calculate the number of samples: mel_frames * hop_length
    return int(mel_frames * hop_length - 1)


class AudioProcessor:
    """
    Processes audio waveforms into normalized mel spectrograms using torchaudio.
    This class encapsulates the parameters and logic for converting a raw audio waveform
    tensor into a mel spectrogram representation suitable for machine learning models.
    It applies a MelSpectrogram transformation followed by log scaling and normalization.

    Args:
        n_mels (int): Number of mel bins. Defaults to 80.
        n_fft (int): Size of the FFT. Defaults to 2048.
        win_length (int): Window size for STFT. Defaults to 1200.
        hop_length (int): Hop length for STFT. Defaults to 300.
        mean (float): Mean for normalization. Defaults to -4.
        std (float): Standard deviation for normalization. Defaults to 4.
    Raises:
        Exception: If `torchaudio.transforms.MelSpectrogram` fails to initialize.
    """

    def __init__(
        self,
        n_mels=80,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        mean=-4,
        std=4,
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.mean = mean
        self.std = std

        # Initialize the transform once
        try:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
            )
            logger.debug("AudioProcessor initialized MelSpectrogram transform.")
        except Exception as e:
            logger.error("Failed to initialize MelSpectrogram in AudioProcessor: %s", e)
            raise  # Re-raise the exception as it's critical

    def __call__(self, wave_tensor):
        """Converts a waveform tensor to a normalized mel spectrogram.
        Args:
            wave_tensor (torch.Tensor): Input waveform 1D tensor of shape [samples].
        Returns:
            torch.Tensor: Normalized mel spectrogram of shape [n_mels, n_frames].
        """
        # 1. Move transform to the same device as the input tensor
        # Input wave must be 1D tensor [samples] -> Output: [n_mels, n_frames]
        mel_transform_device = self.mel_transform.to(wave_tensor.device)

        # 2. Compute mel spectrogram
        # Input: [samples] -> Output: [n_mels, n_frames]
        mel_spec = mel_transform_device(wave_tensor)

        # 3. Normalize
        # Input: [n_mels, n_frames] -> Output: [n_mels, n_frames]
        mel_normalized = (torch.log(1e-5 + mel_spec) - self.mean) / self.std
        # Removed unsqueeze/squeeze, returns [n_mels, n_frames] directly

        return mel_normalized


class FilePathDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for loading audio file paths, corresponding texts, and speaker IDs.
    This dataset handles loading audio data, preprocessing it into mel spectrograms,
    converting text into phoneme IDs using a provided cleaner function, and
    preparing batches suitable for model training or validation. It also supports
    loading out-of-distribution (OOD) texts and selecting reference audio samples
    for tasks like style transfer or speaker conditioning.

    Args:
        data_list (list): A list of strings, where each string contains data
            separated by '|'. Expected formats are "wav_path|phonetic_text" or
            "wav_path|phonetic_text|speaker_id". Lines with text lengths
            outside [`min_length`, `max_length`] are skipped.
        root_path (str): The root directory where the audio files specified in
            `data_list` are located.
        text_cleaner: An object or function responsible for converting phonetic
            strings into sequences of numerical IDs and handling padding. It should
            have a method `add_spaces_around_punctuation` and be callable like
            `text_cleaner(text, pad=True)`.
        data_augmentation (bool, optional): Flag to enable data augmentation.
            Currently not implemented/used. Defaults to False.
        validation (bool, optional): Flag indicating if the dataset is for
            validation. If True, data augmentation is disabled. Defaults to False.
        ood_data (str, optional): Path to a file containing out-of-distribution
            (OOD) texts, one per line. Used for providing diverse text prompts
            during training/evaluation. Defaults to None.
        **kwargs: Additional keyword arguments for configuration:
            sr (int): Target sampling rate for audio. Defaults to 24000.
            min_length (int): Minimum allowed length for phonetic strings. Defaults to 50.
            max_length (int): Maximum allowed length for phonetic strings. Defaults to 512.
            silence_beg (int): Samples of silence to pad at the beginning of waveforms. Defaults to 4800.
            silence_end (int): Samples of silence to pad at the end of waveforms. Defaults to 4800.
            spect_params (dict): Parameters for spectrogram calculation (n_fft,
                win_length, hop_length). Defaults to {"n_fft": 2048,
                "win_length": 1200, "hop_length": 300}.
            n_mels (int): Number of mel bins for spectrograms. Defaults to 80.
            mean (float): Mean value for mel spectrogram normalization. Defaults to -4.
            std (float): Standard deviation for mel spectrogram normalization. Defaults to 4.
    Raises:
        AssertionError: If the input data format is invalid or if the phonetic
            string length exceeds the specified maximum length.
    """

    def __init__(
        self,
        data_list,
        root_path,
        text_cleaner,
        data_augmentation=False,
        validation=False,
        ood_data=None,
        **kwargs,
    ):
        # Get parameters from kwargs (config)
        self.sr = kwargs.get("sr", 24000)
        self.min_length = kwargs.get("min_length", 50)
        self.max_length = kwargs.get("max_length", 512)
        self.silence_beg = kwargs.get("silence_beg", 4800)
        self.silence_end = kwargs.get("silence_end", 4800)
        self.spect_params = kwargs.get(
            "spect_params",
            {"n_fft": 2048, "win_length": 1200, "hop_length": 300},
        )
        self.n_mels = kwargs.get("n_mels", 80)
        self.mean = kwargs.get("mean", -4)
        self.std = kwargs.get("std", 4)
        self.use_ref_sample = kwargs.get("use_ref_sample", True)
        # 2.4s at 24000 Hz (192 mel frames * 300 hop length - 1)
        self.max_ref_wave_length = mel2samples_length(
            kwargs.get("max_ref_mel_length", 192), self.spect_params["hop_length"]
        )

        self.data_augmentation = data_augmentation and (not validation)  # not used
        self.root_path = root_path  # Set up path to waveform directory
        self.ptexts = []  # Initialize list of OOD texts

        logger.info("%s dataset config: %s", "validation" if validation else "training", kwargs)

        # Set up text cleaner for phone ID encoding and padding ID
        self.text_cleaner = text_cleaner

        # Create AudioProcessor instance
        self.audio_processor = AudioProcessor(
            n_mels=self.n_mels,
            n_fft=self.spect_params["n_fft"],
            win_length=self.spect_params["win_length"],
            hop_length=self.spect_params["hop_length"],
            mean=self.mean,
            std=self.std,
        )

        # Load data lists
        self.data_list = self._load_texts(data_list)
        logger.info(
            "Loaded %d %s files.", len(self.data_list), "validation" if validation else "training"
        )
        self.df = pd.DataFrame(self.data_list)

        # Load Out-of-distribution texts if provided
        if ood_data is not None:
            self.ptexts = self._load_ood_texts(ood_data)
        logger.info("Loaded %d OOD texts.", len(self.ptexts))

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
        # Return the list of phonetic strings
        return ph_texts

    def __len__(self):
        return len(self.data_list)

    def number_ood_texts(self):
        """Returns the number of out-of-distribution (OOD) texts."""
        return len(self.ptexts)

    def _load_tensor(self, data):
        """Loads and preprocesses a single audio sample and its metadata.
        This method takes a tuple containing the relative path to a waveform file,
        a string of phonemes, and a speaker ID. It loads the waveform, ensures
        it's mono, resamples it to the target sample rate if necessary, adds
        silence padding at the beginning and end, cleans the phoneme string,
        converts it to a tensor of phone IDs with padding, and returns the
        processed data.
        Args:
            data (tuple): A tuple containing:
                - wave_path (str): Relative path to the waveform file.
                - ph_string (str): String representation of phonemes.
                - speaker_id (str or int): The speaker identifier.
        Returns:
            tuple: A tuple containing:
                - wave (torch.Tensor): The processed 1D waveform tensor.
                - phone_ids (torch.LongTensor): Padded tensor of phone IDs.
                - speaker_id (int): The integer speaker identifier.
        Raises:
            Warning: Logs a warning if resampling is performed.
        """
        wave_path, ph_string, speaker_id = data
        speaker_id = int(speaker_id)  # Ensure speaker_id is an integer

        # Load waveform directly into a tensor
        wave, sr = torchaudio.load(osp.join(self.root_path, wave_path))
        # Handle stereo audio by taking the first channel
        if wave.shape[0] > 1:
            wave = wave[0, :].unsqueeze(0)  # Keep it as a 2D tensor [1, n_samples]

        logger.debug("| > id: %d: | wav=%s | ph=%s", speaker_id, wave.shape, ph_string)

        # Resample if necessary
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            wave = resampler(wave)
            logger.warning("%s: sampling rate is %d, resampling to %d", wave_path, sr, self.sr)

        # Add padding to the waveform tensor
        silence_beg_tensor = torch.zeros((1, self.silence_beg), dtype=wave.dtype)
        silence_end_tensor = torch.zeros((1, self.silence_end), dtype=wave.dtype)
        wave = torch.cat([silence_beg_tensor, wave, silence_end_tensor], dim=1)

        logger.debug("| > silence padding: wav=%s", wave.shape)

        return (
            wave.squeeze(0),  # raw waveform as 1D tensor
            torch.LongTensor(self.text_cleaner(ph_string, pad=True)),  # phone IDs with padding
            speaker_id,  # speaker ID
        )

    def _load_speaker_embedding(self, spk_emb_path):
        """Loads the speaker embedding from a specified path.
        Args:
            spk_emb_path (str): Path to the speaker embedding file.
        Returns:
            torch.Tensor: The loaded speaker embedding tensor.
        """
        return torch.load(osp.join(self.root_path, spk_emb_path))

    def _load_ref_data(self, data):
        """
        Load and process reference audio data for style transfer.

        This method loads reference audio waveform data, processes it into mel-spectrogram
        features, and resamples the audio to the target sample rate. If the audio is longer
        than the maximum allowed reference length, it randomly crops a segment.

        Args:
            data: Input data containing audio file information or path

        Returns:
            tuple: A tuple containing:
                - wave_tensor_resampled (torch.Tensor): Resampled audio waveform tensor
                - mel_tensor (torch.Tensor): Mel-spectrogram features with shape [n_mels, n_frames]
                - speaker_id: Speaker identification information

        Note:
            - Audio longer than max_ref_wave_length is randomly cropped
            - Mel-spectrogram is computed using the configured audio processor
            - Audio is resampled to match the target sample rate
        """
        # Loads the reference audio waveform
        wave_tensor, _, speaker_id = self._load_tensor(data)
        logger.debug("| > Ref: %s |shape=%s", data[0], wave_tensor.shape)
        wave_len = wave_tensor.size(0)
        if wave_len > self.max_ref_wave_length:
            # Randomly crop a wave segment with length self.max_ref_wave_length
            random_start = np.random.randint(0, wave_len - self.max_ref_wave_length)
            wave_tensor = wave_tensor[random_start : random_start + self.max_ref_wave_length]

        # Process audio
        # Output: [n_mels, n_frames]
        mel_tensor = self.audio_processor(wave_tensor)
        logger.debug("| > Cropping: wav=%s, mel=%s", wave_tensor.shape, mel_tensor.shape)

        # # Resample to target sample rate
        # wave_tensor_resampled = self.resampler(wave_tensor.unsqueeze(0)).squeeze(0)

        return wave_tensor, mel_tensor, speaker_id

    def __getitem__(self, idx):
        """
        Retrieves a data sample for the given index.
        This method loads an audio file and its corresponding text transcription,
        processes the audio into a mel spectrogram, and selects a reference
        audio sample from the same speaker. If out-of-distribution (OOD)
        phonetic texts are provided, it also selects a random OOD text and
        encodes it.
        Args:
            idx (int): The index of the data sample to retrieve.
        Returns:
            tuple: A tuple containing the following elements:
            - speaker_id (torch.Tensor): The speaker ID of the primary audio sample.
            - acoustic_feature (torch.Tensor): The normalized mel spectrogram
              of the primary audio sample, with an even number of frames.
              Shape: [n_mels, n_frames].
            - text_tensor (torch.Tensor): The tensor of phoneme IDs for the
              primary text transcription.
            - ref_ph_ids (torch.Tensor): The tensor of phoneme IDs for a
              randomly selected OOD text (if available, padded), or an
              empty tensor otherwise.
            - ref_mel_tensor (torch.Tensor): The mel spectrogram of the
              reference audio sample from the same speaker.
            - ref_label (torch.Tensor): The speaker ID of the reference
              audio sample.
            - data[0] (str): The file path of the primary audio waveform.
            - wave (torch.Tensor): The raw waveform tensor of the primary
              audio sample.
        """
        data = self.data_list[idx]  # [wavfile, phone IDs, speaker_id]

        logger.debug("> %s (%d):", data[0], idx)

        # Load the waveform, phonetic string, and speaker ID
        wave, text_tensor, speaker_id = self._load_tensor(data)

        # Process audio: output [n_mels, n_frames]
        mel_normalized = self.audio_processor(wave)

        logger.debug("| > mel=%s", mel_normalized.shape)

        # acoustic_feature = mel_tensor.squeeze()
        acoustic_feature = mel_normalized
        length_feature = acoustic_feature.size(1)
        # Ensure feature tensor with even length
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]

        # Get reference sample of max length `self.max_ref_mel_length` (192)
        if self.use_ref_sample:
            ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            # ref_label is speaker ID
            ref_wave_tensor, ref_mel_tensor, ref_label = self._load_ref_data(ref_data[:3])
        else:
            ref_wave_tensor = torch.tensor([], dtype=torch.float)  # Empty tensor
            ref_mel_tensor = torch.tensor([])  # Empty tensor
            ref_label = 0

        # Randomly select a phonetic sentence from the OOD texts if available
        if self.ptexts:
            ref_ph_string = self.ptexts[np.random.randint(0, len(self.ptexts) - 1)]
            # Encode phonetic string as a list of phoneme IDs with padding
            ref_ph_ids = torch.tensor(self.text_cleaner(ref_ph_string, pad=True), dtype=torch.long)
        else:
            # Use empty tensor if no OOD texts are provided
            ref_ph_ids = torch.tensor([], dtype=torch.long)  # Empty tensor

        return (
            speaker_id,  # speaker ID
            acoustic_feature,  # mel spectrogram of input waveform
            text_tensor,  # phoneme IDs of input text
            ref_ph_ids,  # phone IDs of OOD text with padding or empty tensor
            ref_wave_tensor,  # reference waveform tensor of the given speaker
            ref_mel_tensor,  # reference mel vector of the given speaker
            ref_label,  # reference speaker ID
            data[0],  # wavfile path
            wave,  # raw waveform tensor
        )


class Collater(object):
    """
    Collate function for creating batches of data.
    This class is used to collate individual data samples into a batch
    suitable for model training or inference. It handles padding and
    batching of mel spectrograms, phoneme IDs, and other relevant data.
    Args:
        return_wave (bool): Flag to indicate whether to return the raw waveform
            tensor in the batch. Defaults to False.
    Raises:
        AssertionError: If the input data format is invalid or if the phonetic
            string length exceeds the specified maximum length.
    Note:
        The `return_wave` parameter is not used in the current implementation,
        but it is included for future extensibility.
        The `__call__` method is the main entry point for collating a batch of
        data samples.
    """

    def __init__(self, return_wave=False, **kwargs):
        self.return_wave = return_wave

        # Processing parameters from kwargs (config)
        self.sr = kwargs.get("sr", 24000)
        self.hop_length = kwargs.get("hop_length", 300)
        self.max_ref_mel_length = kwargs.get("max_ref_mel_length", 192)
        self.max_ref_wave_length = mel2samples_length(self.max_ref_mel_length, self.hop_length)

        logger.info("collate config: %s", kwargs)

    def __call__(self, batch):
        """
        Collate function for batching dataset samples with padding and sorting.

        This method processes a batch of samples by sorting them by acoustic feature length
        (descending order) and padding all sequences to match the maximum length within
        the batch. It handles multiple types of data including mel spectrograms, text
        phoneme IDs, reference audio, and raw waveforms.

        Args:
            batch (list): List of tuples, where each tuple contains:
                - [0] label (int): Speaker ID
                - [1] mel (torch.Tensor): Acoustic features/mel spectrogram [n_mels, T_mel]
                - [2] text (torch.Tensor): Input phoneme IDs [T_text]
                - [3] ref_text (torch.Tensor): Reference/OOD text phoneme IDs [T_ref_text]
                - [4] ref_wave (torch.Tensor): Reference waveform [1, T_ref_wave]
                - [5] ref_mel (torch.Tensor): Reference mel spectrogram [n_mels, T_ref_mel]
                - [6] ref_label (int): Reference speaker ID (unused)
                - [7] wavfile_path (str): Path to wave file (unused)
                - [8] wave (torch.Tensor): Raw waveform tensor [T_samples]

        Returns:
            tuple: A tuple containing:
                - waves (list): List of raw waveform tensors, length B
                - texts (torch.Tensor): Padded input phoneme IDs [B, max_text_length]
                - input_lengths (torch.Tensor): Actual lengths of input texts [B]
                - ref_texts (torch.Tensor): Padded reference phoneme IDs [B, max_rtext_length]
                - ref_lengths (torch.Tensor): Actual lengths of reference texts [B]
                - mels (torch.Tensor): Padded mel spectrograms [B, n_mels, max_mel_length]
                - output_lengths (torch.Tensor): Actual lengths of mel spectrograms [B]
                - ref_waves (torch.Tensor): Padded reference waveforms [B, 1, max_ref_wave_length]
                                           or empty tensor if no valid reference waves
                - ref_mels (torch.Tensor): Padded reference mel spectrograms [B, n_mels, max_mel_length]
                                          or empty tensor if no valid reference mels

        Note:
            - Batch is sorted by mel spectrogram length in descending order for efficient training
            - Reference waves and mels are conditionally initialized based on availability
            - Raw waves are kept as a list to manage memory constraints
            - Padding is applied to ensure all sequences in batch have uniform dimensions
        """

        bsize = len(batch)  # Number of samples in the batch

        # Sort batch by acoustic feature (mel) length (descending)
        # b[1] is acoustic_feature from __getitem__
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]  # Get indices for sorting
        batch = [batch[bid] for bid in batch_indexes]  # Sort the batch

        # Determine max lengths for padding within this batch
        nmels = batch[0][1].size(0)
        # Max length of acoustic_feature in this batch
        max_mel_length = max(b[1].shape[1] for b in batch)
        # b[2] is text_tensor
        max_text_length = max(b[2].shape[0] for b in batch)
        # b[3] is ref_ph_ids (OOD text)
        max_rtext_length = max(b[3].shape[0] for b in batch)

        # # Check if batch has valid reference mel tensors
        # # b[5] is ref_mel_tensor from __getitem__
        # has_valid_ref_mel = batch[0][5].numel() != 0

        # Initialize padded tensors
        # b[0] is speaker_id (integer)
        labels = torch.zeros((bsize)).long()
        # b[1] is acoustic_feature (mel spectrogram)
        mels = torch.zeros((bsize, nmels, max_mel_length)).float()
        # b[2] is text_tensor (input phoneme IDs)
        texts = torch.zeros((bsize, max_text_length)).long()
        # b[3] is ref_ph_ids (OOD text phoneme IDs)
        ref_texts = torch.zeros((bsize, max_rtext_length)).long()

        input_lengths = torch.zeros(bsize).long()
        ref_lengths = torch.zeros(bsize).long()
        output_lengths = torch.zeros(bsize).long()

        # Initialize ref wave tensors conditionally
        # b[4] is ref_wave_tensor, use `self.max_ref_wave_length` (fixed length)
        # Check if batch has valid reference mel tensors
        # b[4] is ref_wave_tensor from __getitem__
        has_valid_ref_wave = batch[0][4].numel() != 0
        ref_waves = (
            torch.zeros((bsize, 1, self.max_ref_wave_length)).float()
            if has_valid_ref_wave
            else torch.tensor([], dtype=torch.float)  # Return None if no item has ref_wave
        )

        # Initialize ref_mels conditionally
        # b[5] is ref_mel_tensor, use `self.max_mel_length`
        ref_mels = (
            torch.zeros((bsize, nmels, self.max_ref_mel_length)).float()
            if has_valid_ref_wave
            else torch.tensor([], dtype=torch.float)  # Return None if no item has ref_mel
        )

        # b[6] is ref_label (reference speaker ID) - not used
        # ref_labels = torch.zeros((batch_size)).long()

        # b[7] is wavfile path - not used anymore
        # paths = ["" for _ in range(batch_size)]

        # b[8] is raw wave tensor
        # Due to memory constraints, it is better to keep it as a list
        waves = [None for _ in range(bsize)]

        # Re-arrange batch data according to mel length
        for bid, (label, mel, text, ref_text, ref_wave, ref_mel, _, _, wave) in enumerate(batch):
            mel_size = mel.size(1)  # Get sizes of current item
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            ref_wave_size = ref_wave.size(0) if has_valid_ref_wave else 0
            ref_mel_size = ref_mel.size(1) if has_valid_ref_wave else 0

            # Fill tensors
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size

            # Only assign if ref_wave is not None and size > 0
            # Otherwise, it remains empty (as initialized)
            if ref_wave_size > 0:
                # Ensure we don't try to assign beyond the bounds of ref_waves
                ref_waves[bid, :, :ref_wave_size] = ref_wave

            # Only assign if ref_mel is not None and size > 0
            if ref_mel_size > 0:
                # Ensure we don't try to assign beyond the bounds of ref_mels
                ref_mels[bid, :, :ref_mel_size] = ref_mel
            # If ref_mel_size is 0, ref_mels[bid] remains empty (as initialized)

            # ref_labels[bid] = ref_label  # not used anymore
            waves[bid] = wave
            # spkenc_segment_begs[bid] = self._get_segment_start_for_speaker_encoder(
            #     wave,
            #     spkenc_segment_len,
            # )

        return (
            waves,  # List of raw waveform tensors [T_samples] (or None)
            texts,  # Padded input phoneme IDs [B, T_text]
            input_lengths,  # Input phoneme lengths [B]
            ref_texts,  # Padded OOD text phoneme IDs [B, T_ref_text]
            ref_lengths,  # OOD texts phoneme lengths [B]
            mels,  # Padded mel spectrograms [B, n_mels, T_mel]
            output_lengths,  # Mel spectrogram lengths [B]
            ref_waves,  # Padded reference waveforms [B, 1, max_ref_wave_length]
            ref_mels,  # Padded reference mel spectrograms [B, n_mels, max_mel_length]
        )


class BalancedSpeakerSampler(Sampler):
    """A PyTorch Sampler that ensures balanced speaker representation across mini-batches.
    This sampler distributes data samples to maintain approximately equal representation
    of different speakers within each batch, which is particularly useful for training
    speaker-aware models like text-to-speech systems.
    The sampler supports distributed training (DDP) by partitioning speakers across
    multiple processes and provides deterministic shuffling through epoch-based seeding.
    Args:
        dataset: Dataset object containing data_list where each item has speaker_id at index 2
        batch_size (int): Number of samples per mini-batch
        drop_last (bool): Whether to drop the last incomplete batch
        seed (int, optional): Base random seed for reproducibility. Defaults to 42.
        rank (int, optional): Process rank for distributed training. Defaults to 0.
        world_size (int, optional): Total number of processes in distributed training. Defaults to 1.
    Raises:
        ValueError: If batch_size < 1 or world_size < 1
        TypeError: If batch_size is not an integer
    Attributes:
        batch_size (int): Number of samples per batch
        drop_last (bool): Whether to drop incomplete final batch
        base_seed (int): Base seed for random number generation
        rank (int): Current process rank
        world_size (int): Total number of distributed processes
        epoch (int): Current training epoch (updated via set_epoch)
        spk2idx (dict): Mapping from speaker IDs to lists of sample indices
        speakers (list): List of all unique speaker IDs
    Example:
        >>> sampler = BalancedSpeakerSampler(dataset, batch_size=32, drop_last=True)
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in dataloader:
        ...         # Training code here
        ...         pass
    Note:
        Call set_epoch() at the beginning of each training epoch to ensure
        proper shuffling and reproducibility across epochs and distributed processes."""

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last,
        seed=3407,
        rank=0,
        world_size=1,
    ):
        super().__init__(None)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0  # Updated via ``set_epoch`` from the training loop.

        # Build mapping {speaker_id: [indices]}
        self.spk2idx = defaultdict(list)
        for index, data in enumerate(dataset.data_list):
            spk_id = data[2]  # data[2] is speaker_id
            self.spk2idx[spk_id].append(index)
        self.speakers = list(self.spk2idx.keys())

        # Pre‑allocate RNG.  We *reseed* it every epoch for determinism.
        self._rng = random.Random()

    # ---------------------------------------------------------------------
    # Epoch property with setter
    # ---------------------------------------------------------------------
    @property
    def epoch(self):
        """Current training epoch number."""
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        """Set the current epoch for deterministic sampling.

        This method is typically called by the DataLoader to ensure proper
        shuffling behavior across training epochs when using distributed training
        or epoch-based sampling strategies.

        Args:
            value (int): The current epoch number.
        """
        self._epoch = value

    # ---------------------------------------------------------------------
    # Legacy compatibility method
    # ---------------------------------------------------------------------
    def set_epoch(self, epoch):  # noqa: D401 (non‑imperative)
        """Set the current epoch for the dataset.

        This method is typically called by the DataLoader to ensure proper
        shuffling behavior across training epochs when using distributed training
        or epoch-based sampling strategies.

        Args:
            epoch (int): The current epoch number.
        """
        self.epoch = epoch

    # ------------------------------------------------------------------
    # Sampler core
    # ------------------------------------------------------------------
    def __iter__(self):  # noqa: D401
        """Iterate over batches of data indices with speaker-balanced sampling.
        This iterator implements a sophisticated batching strategy that:
        1. Uses deterministic seeding based on epoch and rank for reproducibility
        2. Shuffles samples within each speaker and shuffles speaker order
        3. Distributes speakers across multiple ranks for distributed training
        4. Rotates through active speakers to create balanced batches
        5. Ensures each batch contains samples from different speakers when possible
        The algorithm maintains fairness by cycling through speakers and only
        removing them from the active pool when they're exhausted. This prevents
        any single speaker from dominating the batches.
        Yields:
            List[int]: Batches of data indices, each of size `batch_size`
                        (except possibly the last batch if `drop_last=False`)
        Note:
            - Uses NumPy's permutation for efficient shuffling of large lists (>32 items)
            - Falls back to Python's random.shuffle for smaller lists
            - Supports distributed data parallel (DDP) training via rank-based partitioning
            - Reshuffles active speakers after each full rotation to maintain randomness
        """

        # ------------------------------------------------------------------
        # (1) Deterministic seed per epoch & rank
        # ------------------------------------------------------------------
        epoch_seed = self.base_seed + self.epoch + self.rank * 10_000
        self._rng.seed(epoch_seed)
        np_rng = np.random.default_rng(epoch_seed)

        # ------------------------------------------------------------------
        # (2) Shuffle order *inside* each speaker + order of speakers
        # ------------------------------------------------------------------
        for idx_list in self.spk2idx.values():
            # NumPy shuffle is ~2× faster for long lists than pure Python.
            if len(idx_list) > 32:
                idx_list[:] = np_rng.permutation(idx_list).tolist()
            else:
                self._rng.shuffle(idx_list)

        # Partition speakers among ranks in DDP (simple round‑robin split)
        speakers_this_rank = self.speakers[self.rank :: self.world_size]
        self._rng.shuffle(speakers_this_rank)

        # Cursor tracks how many clips have been consumed per speaker.
        cursor = {spk: 0 for spk in speakers_this_rank}

        batch = []
        active_spk = [spk for spk in speakers_this_rank if cursor[spk] < len(self.spk2idx[spk])]

        while active_spk:
            # Rotate through the list, reshuffling every full pass.
            for spk in list(active_spk):
                pos = cursor[spk]
                if pos >= len(self.spk2idx[spk]):
                    continue  # Speaker exhausted, handled later.

                batch.append(self.spk2idx[spk][pos])
                cursor[spk] += 1

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            # Remove exhausted speakers and reshuffle the remainder.
            active_spk = [s for s in active_spk if cursor[s] < len(self.spk2idx[s])]
            self._rng.shuffle(active_spk)

        # Tail batch (if allowed)
        if batch and not self.drop_last:
            yield batch

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        """
        Calculate the number of batches available for this dataset rank.
        Returns the total number of batches that will be produced by this dataset
        instance, taking into account distributed training settings. The calculation
        considers the portion of data assigned to the current rank and applies
        batch size division with optional dropping of incomplete batches.
        Returns:
            int: Number of batches available for iteration. If drop_last is True,
                 returns only complete batches. Otherwise, includes the final
                 incomplete batch if present.
        """

        total_clips = sum(len(v) for v in self.spk2idx.values())
        # Only the portion of data assigned to *this* rank counts.
        total_clips = math.ceil(total_clips / self.world_size)
        if self.drop_last:
            return total_clips // self.batch_size
        return math.ceil(total_clips / self.batch_size)


def build_dataloader(
    path_list,
    root_path,
    text_cleaner,
    validation=False,
    ood_data=None,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config=None,
    dataset_config=None,
    use_speaker_sampler=False,
):
    """Builds and returns a PyTorch DataLoader.
    The DataLoader is configured for loading audio and text data pairs
    from a list of file paths, using a custom dataset and collate function.
    Args:
        path_list (list): List of file paths to the data samples or metadata.
        root_path (str): Root directory containing the actual data files.
        text_cleaner (callable): Function or object to clean/preprocess text data.
        validation (bool, optional): If True, configure DataLoader for validation
            (no shuffling, no drop_last). Defaults to False.
        ood_data (any, optional): Out-of-distribution data, passed to the dataset.
            Defaults to None.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        num_workers (int, optional): Number of subprocesses for data loading.
            Defaults to 1.
        device (str, optional): Target device ('cpu' or 'cuda'). Affects pin_memory.
            Defaults to "cpu".
        collate_config (dict, optional): Configuration dictionary for the Collater.
            Defaults to an empty dict.
        dataset_config (dict, optional): Configuration dictionary for the FilePathDataset.
            Defaults to an empty dict.
        use_speaker_sampler (bool, optional): If True, use BalancedSpeakerSampler
            to ensure one sample per speaker per batch. Defaults to False.
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader instance.
    """
    collate_config = collate_config or {}
    dataset_config = dataset_config or {}

    # Propagate selected params from dataset_config to collate_config if missing
    if "sr" not in collate_config and "sr" in dataset_config:
        collate_config["sr"] = dataset_config["sr"]
    if "hop_length" not in collate_config and "hop_length" in dataset_config["spect_params"]:
        collate_config["hop_length"] = dataset_config["spect_params"]["hop_length"]
    if "max_ref_mel_length" not in collate_config and "max_ref_mel_length" in dataset_config:
        collate_config["max_ref_mel_length"] = dataset_config["max_ref_mel_length"]

    dataset = FilePathDataset(
        path_list,
        root_path,
        text_cleaner,
        ood_data=ood_data,
        validation=validation,
        **dataset_config,
    )
    # Create collate function with provided configuration
    collate_fn = Collater(**collate_config)

    # Dataloader for speaker-balanced sampling
    if use_speaker_sampler and not validation:
        batch_sampler = BalancedSpeakerSampler(
            dataset,
            batch_size=batch_size,
            drop_last=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(device != "cpu"),
        )
        return dataloader

    # Dataloader for regular sampling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )
    return dataloader
