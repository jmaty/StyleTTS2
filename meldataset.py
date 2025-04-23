# coding: utf-8
import os.path as osp
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader

from logger import get_logger

# Setup logger
logger = get_logger(__name__)

np.random.seed(1)
random.seed(1)


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
        self.data_augmentation = data_augmentation and (not validation)  # not used
        self.max_mel_length = 192
        self.root_path = root_path  # Set up path to waveform directory

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
        # Check if the OOD file is provided for loading.
        # If not, return an empty list
        if ood_file is None:
            return []

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
        # Load the waveform, phonetic string, and speaker ID
        wave, text_tensor, speaker_id = self._load_tensor(data)

        # Process audio: output [n_mels, n_frames]
        mel_normalized = self.audio_processor(wave)

        # acoustic_feature = mel_tensor.squeeze()
        acoustic_feature = mel_normalized
        length_feature = acoustic_feature.size(1)
        # Ensure feature tensor with even length
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]

        # Get reference sample of max length `self.max_mel_length` (192)
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])  # ref_label is speaker ID

        # Randomly select a phonetic sentence from the OOD texts if available
        if self.ptexts:
            ref_ph_string = self.ptexts[np.random.randint(0, len(self.ptexts) - 1)]
            # Encode phonetic string as a list of phoneme IDs with padding
            ref_ph_ids = torch.LongTensor(self.text_cleaner(ref_ph_string, pad=True))
        else:
            # Use empty tensor if no OOD texts are provided
            ref_ph_ids = torch.LongTensor([])  # Empty tensor

        return (
            speaker_id,  # speaker ID
            acoustic_feature,  # mel spectrogram of input waveform
            text_tensor,  # phoneme IDs of input text
            ref_ph_ids,  # phone IDs of OOD text with padding or empty tensor
            ref_mel_tensor,  # reference mel vector of the given speaker
            ref_label,  # reference speaker ID
            data[0],  # wavfile
            wave,  # raw waveform tensor
        )

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
        # Load waveform directly into a tensor
        wave, sr = torchaudio.load(osp.join(self.root_path, wave_path))
        # Handle stereo audio by taking the first channel
        if wave.shape[0] > 1:
            wave = wave[0, :].unsqueeze(0)  # Keep it as a 2D tensor [1, n_samples]
        # Resample if necessary
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            wave = resampler(wave)
            logger.warning("%s: sampling rate is %d, resampling to %d", wave_path, sr, self.sr)

        # Add padding to the waveform tensor
        silence_beg_tensor = torch.zeros((1, self.silence_beg), dtype=wave.dtype)
        silence_end_tensor = torch.zeros((1, self.silence_end), dtype=wave.dtype)
        wave = torch.cat([silence_beg_tensor, wave, silence_end_tensor], dim=1)

        return (
            wave.squeeze(0),  # raw waveform as 1D tensor
            torch.LongTensor(self.text_cleaner(ph_string, pad=True)),  # phone IDs with padding
            int(speaker_id),  # speaker ID
        )

    def _load_data(self, data):
        """Takes a tuple containing the relative path to a waveform file,
        a string of phonemes, and a speaker ID. It loads the waveform, processes
        it into a mel spectrogram, and crops it to a maximum length if necessary.
        Args:
            data (tuple): A tuple containing:
                - wave_path (str): Relative path to the waveform file.
                - ph_string (str): String representation of phonemes.
                - speaker_id (str or int): The speaker identifier.
        Returns:
            tuple: A tuple containing:
                - mel_tensor (torch.Tensor): The processed mel spectrogram.
                - speaker_id (int): The integer speaker identifier.
        Raises:
            Warning: Logs a warning if the mel spectrogram length exceeds
                the maximum length and crops it."""
        # Loads the reference audio waveform
        wave, _, speaker_id = self._load_tensor(data)
        # mel_tensor = preprocess(wave).squeeze()
        # Process audio
        # Output: [n_mels, n_frames]
        mel_tensor = self.audio_processor(wave)

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            # Randomly crop a segment of the mel spectrogram with length self.max_mel_length
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start : random_start + self.max_mel_length]

        return mel_tensor, speaker_id


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

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave

    def __call__(self, batch):
        """Collate function for creating batches of data.
        This method takes a list of data samples (returned by __getitem__)
        and collates them into a batch suitable for model training or inference.
        It sorts the batch by mel spectrogram length (descending), determines
        maximum lengths for padding within the batch, initializes padded tensors,
        and populates these tensors with data from the batch samples. Reference
        mel spectrograms are padded or cropped to a fixed maximum length defined
        in the class configuration (`self.max_mel_length`).
        Args:
            batch (list): A list of tuples, where each tuple represents a data
                sample and contains:
                - speaker_id (int): Speaker identifier.
                - acoustic_feature (torch.Tensor): Mel spectrogram [n_mels, T_mel].
                - text_tensor (torch.Tensor): Input phoneme IDs [T_text].
                - ref_ph_ids (torch.Tensor): OOD text phoneme IDs [T_ref_text].
                - ref_mel_tensor (torch.Tensor): Reference mel spectrogram
                  [n_mels, T_ref_mel].
                - ref_label (int): Reference speaker ID (not used in the returned batch).
                - wavfile_path (str): Path to the original audio file.
                - wave_tensor (torch.Tensor or None): Raw waveform tensor.
        Returns:
            tuple: A tuple containing the following batched and padded tensors:
            - waves (list): List of raw waveform tensors (torch.Tensor or None)
              from the batch.
            - texts (torch.Tensor): Padded input phoneme IDs [B, max_text_length].
            - input_lengths (torch.Tensor): Lengths of input phoneme sequences [B].
            - ref_texts (torch.Tensor): Padded OOD text phoneme IDs
              [B, max_rtext_length].
            - ref_lengths (torch.Tensor): Lengths of OOD text phoneme sequences [B].
            - mels (torch.Tensor): Padded mel spectrograms
              [B, n_mels, max_mel_length].
            - output_lengths (torch.Tensor): Lengths of mel spectrograms [B].
            - ref_mels (torch.Tensor): Padded/cropped reference mel spectrograms
              [B, n_mels, self.max_mel_length].
        """

        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # Sort batch by acoustic feature (mel) length (descending)
        # b[1] is acoustic_feature from __getitem__
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]  # Get indices for sorting
        batch = [batch[bid] for bid in batch_indexes]  # Sort the batch

        # Determine max lengths for padding within this batch
        nmels = batch[0][1].size(0)
        # Max length of acoustic_feature in this batch
        max_mel_length = max([b[1].shape[1] for b in batch])
        # b[2] is text_tensor
        max_text_length = max([b[2].shape[0] for b in batch])
        # b[3] is ref_ph_ids (OOD text)
        max_rtext_length = max([b[3].shape[0] for b in batch])

        # Initialize padded tensors
        # b[0] is speaker_id (integer)
        labels = torch.zeros((batch_size)).long()
        # b[1] is acoustic_feature (mel spectrogram)
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        # b[2] is text_tensor (input phoneme IDs)
        texts = torch.zeros((batch_size, max_text_length)).long()
        # b[3] is ref_ph_ids (OOD text phoneme IDs)
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        # b[4] is ref_mel_tensor, use self.max_mel_length (fixed size from config)
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        # b[5] is ref_label (reference speaker ID) - not used
        # ref_labels = torch.zeros((batch_size)).long()

        # b[6] is wavfile path
        paths = ["" for _ in range(batch_size)]
        # b[7] is raw wave tensor
        waves = [None for _ in range(batch_size)]

        # Rearrange batch data according to mel length
        for bid, (label, mel, text, ref_text, ref_mel, _, path, wave) in enumerate(batch):
            mel_size = mel.size(1)  # Get sizes of current item
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            ref_mel_size = ref_mel.size(1)  # Actual size before padding/cropping

            # Fill tensors
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path

            ref_mels[bid, :, :ref_mel_size] = ref_mel
            # ref_labels[bid] = ref_label  # not used
            waves[bid] = wave

        return (
            waves,  # List of raw waveform tensors (or None)
            texts,  # Padded input phoneme IDs [B, T_text]
            input_lengths,  # Input phoneme lengths [B]
            ref_texts,  # Padded OOD text phoneme IDs [B, T_ref_text]
            ref_lengths,  # OOD texts phoneme lengths [B]
            mels,  # Padded mel spectrograms [B, n_mels, T_mel]
            output_lengths,  # Mel spectrogram lengths [B]
            ref_mels,  # Padded reference mel spectrograms [B, n_mels, max_mel_length]
        )


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
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader instance.
    """
    collate_config = collate_config or {}
    dataset_config = dataset_config or {}

    dataset = FilePathDataset(
        path_list,
        root_path,
        text_cleaner,
        ood_data=ood_data,
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
