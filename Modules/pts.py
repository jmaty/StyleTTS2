import random as python_random
import re
from collections import OrderedDict

import librosa
import numpy as np
import torch
import yaml
from munch import munchify
from scipy.io.wavfile import write

from logger import get_logger
from meldataset import AudioProcessor
from models import build_model, load_ASR_models, load_F0_models
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from text_utils import TextCleaner
from utils import length_to_mask, log_norm
from Utils.PLBERT.util import load_plbert

# Setup logger
logger = get_logger(__name__)


class PTS:
    def __init__(
        self,
        config,
        model,
        t=0.7,
        alpha=0.3,
        beta=0.7,
        diffusion_steps=10,
        embedding_scale=1.0,
        speech_rate=1.0,
        use_glob_noise=False,
        fix_noise_in_ph_string=False,
    ):
        """
        Initialize the PTS (Phonetic Text to Speech) synthesizer.
        This class handles the setup and configuration for text-to-speech synthesis
        using a pre-trained StyleTTS2 model.
        Parameters:
            config: The configuration object or path to the configuration file
            model: The pre-trained model or path to the model checkpoint
            t (float, optional): The temperature parameter for synthesis. Default: 0.7
                                 Weight for convex combination of two styles (of neighboring sentences).
                                 t=1.0 means only the style of current sentences is used.
                                 t=0.0 means only the style of previous sentences is used.
                                 t=(0,1) means the style is a convex combination of both.
            alpha (float, optional): Controls the influence of the style embedding. Default: 0.3
                                     Parameter for timbre similarity with reference speaker.
                                     Higher values set the style more suitable to text
                                     but less similar to the reference speaker.
            beta (float, optional): Controls the influence of the content embedding. Default: 0.7
                                    Parameter for prosody (emotions).
                                    Higher values set the style more suitable to text
                                    but less similar to the reference speaker.
            diffusion_steps (int, optional): Number of diffusion steps. Default: 10
            embedding_scale (float, optional): Scaling factor for the embeddings. Default: 1.0
            speech_rate (float, optional): Controls the rate of synthesized speech. Default: 1.0
            use_glob_noise (bool, optional): Whether to use global noise for all synthesis operations. Default: False
            fix_noise_in_ph_string (bool, optional): Whether to use the same noise for phonetic strings. Default: False

        Note:
            If `use_glob_noise` is True, `fix_noise_in_ph_string` is automatically set to True.
        """
        self._model = None
        self._config = None
        self._sampler = None

        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.diffusion_steps = diffusion_steps
        self.embedding_scale = embedding_scale
        self.speech_rate = speech_rate

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", self.device)

        # Set up model
        self.setup_config(config)
        self.setup_model(model)

        self.text_cleaner = TextCleaner(
            self._config.data_params.symbol_dict_path,
            pad=self._config.data_params.pad,
        )

        symbol_count = len(self.text_cleaner)
        logger.debug("Number of symbols: %s", symbol_count)
        assert symbol_count == 81, f"Number of symbols must be 81 but it is {symbol_count}"

        # Generate global noise if specified
        self.glob_noise = self.generate_noise() if use_glob_noise else None
        # In case of global noise, noise within phonetic string is always fixed;
        # otherwise, it is optional according to `fix_noise_in_ph_string`
        self.fix_noise_in_ph_string = fix_noise_in_ph_string if not use_glob_noise else True
        logger.debug("Using global noise: %s", use_glob_noise)
        logger.debug("Fix noise in phonetic string: %s", fix_noise_in_ph_string)

        # Create audio processor
        self.audio_processor = AudioProcessor(
            n_mels=self._config.model_params.n_mels,
            n_fft=self._config.preprocess_params.spect_params.n_fft,
            win_length=self._config.preprocess_params.spect_params.win_length,
            hop_length=self._config.preprocess_params.spect_params.hop_length,
            mean=self._config.preprocess_params.mean,
            std=self._config.preprocess_params.std,
        )

    def setup_config(self, config):
        # Determine if config is a path or a pre-loaded configuration
        if isinstance(config, str):
            # It's a path
            logger.info("Initializing PTS with config from: %s", config)
            with open(config, encoding="utf-8") as file:
                self._config = munchify(yaml.safe_load(file))
        else:
            # It's a pre-loaded configuration
            logger.info("Initializing PTS with provided configuration object")
            # Ensure it's a Munch object (if it's a dict, convert it)
            self._config = (
                config if isinstance(config, munchify({}).__class__) else munchify(config)
            )

    def setup_model(self, model):
        # Determine if model is a path or a pre-loaded model
        if isinstance(model, str):
            # It's a path
            logger.info("Loading model from path: %s", model)
            self._build_model()
            self._load_model_params(model)
        else:  # It's a pre-loaded model
            self.model = model
            logger.info("Model loaded from pre-loaded object")
        # Set up the diffusion sampler
        self._setup_sampler()

    def to_eval(self):
        """Set all model components to evaluation mode"""
        if self._model is None:
            logger.warning("Cannot set to evaluation mode: Model is not initialized")
            return
        logger.info("Setting model to evaluation mode")
        _ = [self._model[key].eval() for key in self._model]

    def to_device(self):
        """Move all model components to the appropriate device"""
        if self._model is None:
            logger.warning("Cannot move to device: Model is not initialized")
            return
        logger.info("Moving model to device: %s", self.device)
        _ = [self._model[key].to(self.device) for key in self._model]

    def _build_model(self):
        """
        Builds and initializes the StyleTTS2 model with its required components.
        This method loads the necessary pre-trained models:
        - Text aligner (ASR model) for aligning text with audio
        - Pitch extractor (F0 model) for extracting pitch information
        - PLBERT for linguistic feature extraction
        Then constructs the StyleTTS2 model using these components and the configuration
        parameters from self.config.model_params. After building the model, it sets
        the model to evaluation mode and moves it to the appropriate device.
        Returns:
            None: The model is stored as self.model
        """
        logger.info("Building StyleTTS2 model...")

        # Load models
        logger.info("Loading ASR model from %s", self._config.ASR_path)
        text_aligner = load_ASR_models(self._config.ASR_path, self._config.ASR_config)

        logger.info("Loading F0 model from %s", self._config.F0_path)
        pitch_extractor = load_F0_models(self._config.F0_path)

        plbert = load_plbert(self._config.PLBERT_dir)

        # Build StyleTTS2 model
        logger.info("Constructing StyleTTS2 model with components")
        self._model = build_model(self._config.model_params, text_aligner, pitch_extractor, plbert)

        self.to_eval()
        self.to_device()
        logger.info("Model building complete")

    def _load_model_params(self, model_path):
        """
        Load model parameters from a specified path.
        This method loads model parameters from a saved checkpoint file,
        mapping them to the CPU, and applies a module prefix hack to ensure
        compatibility with the current model structure.
        Args:
            model_path (str): Path to the model checkpoint file.
        Returns:
            None: The method loads parameters into the model but doesn't return anything.
        Note:
            This method assumes the checkpoint contains parameters under the 'net' key
            and uses an internal method _hack_module_prefix to modify parameter names if needed.
        """
        logger.info("Loading model parameters from %s", model_path)
        params = torch.load(model_path, map_location="cpu")

        # Reduced model does not have 'net' key but the original full model has 'net' key
        # => handle both cases
        if "net" in params:
            # Original full model
            logger.debug("Full model with 'net' key in model parameters loaded")
            params = params["net"]
        logger.debug("Reduced model with only-inference parameters loaded")

        self._hack_module_prefix(params)
        logger.info("Model parameters loaded successfully")
        self.to_eval()

    def _hack_module_prefix(self, params):
        """
        Handles loading state dictionaries into model components, with a fallback mechanism
        for models saved with torch.nn.DataParallel.
        This method attempts to load parameters from the provided state dictionary into
        corresponding model components. If the direct loading fails (typically due to key mismatches),
        it attempts to remove the 'module.' prefix from keys, which is added when models are
        saved after training with torch.nn.DataParallel.
        Parameters
        ----------
        params : dict
            A dictionary containing state dictionaries for model components,
            where keys correspond to model component names.
        Returns
        -------
        None
            The method updates the model components in-place.
        Notes
        -----
        - The method prints confirmation messages for each successfully loaded component.
        - Uses OrderedDict for maintaining the order of parameters during the prefix removal process.
        - Performs strict=False loading in the fallback case to allow for partial state dict loading.
        """
        logger.warning("Hacking model parameters with module prefix handling")
        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except Exception:
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    try:
                        self.model[key].load_state_dict(new_state_dict, strict=False)
                    except Exception as exc:
                        logger.error("Failed to load parameters for %s: %s}", key, str(exc))
            else:
                logger.debug("No parameters found for component: %s => not used in inference", key)
        logger.info("Model parameters hacked successfully")

    def generate_noise(self):
        """Generate noise

        Returns:
            tensor: Noise for diffusion.
        """
        return torch.randn(1, 1, 256, device=self.device)

    def _setup_sampler(self):
        """Setup diffusion sampler."""
        # Access the original model using .module if wrapped by DDP/FSDP
        try:
            diffusion_model = self.model.diffusion.module
        except AttributeError:
            # Model is not wrapped (e.g., single GPU or CPU)
            diffusion_model = self.model.diffusion

        self._sampler = DiffusionSampler(
            diffusion_model.diffusion,  # access the inner diffusion attribute on the original model
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

    @property
    def sampler(self):
        """Get the diffusion sampler."""
        return self._sampler

    def __call__(
        self,
        ph_strings,
        ref_s=None,
        spk_emb=None,
    ):
        """
        Generate wavs from phonetic strings.

        Args:
            ph_strings (list): List of phoneme strings to be converted to speech.
            ref_s (torch.Tensor): Reference speaker style embedding or path to a wav file.
                If a path is provided, the style embedding will be computed from the wav file.
                If None, the model will not use speaker style embedding (the case of a single speaker model).
                If a tensor is provided, it should be of shape (1, 256) or (1, 256, 1).
                The first 128 dimensions are for timbre and the last 128 dimensions are for prosody.
                If ref_s is None, the model will not use speaker style embedding (the case of a single speaker model).

        Note:
            - This method assumes that the phonetic strings are well-formed
              and that the model is properly initialized.
            - The method will process each phonetic string, split them to phonetic sentences and
              generate the corresponding audio waveform.
            - The method uses the `text_cleaner` to process the phonetic sentences
              (making them compatible with pre-trained PL-BERT and converting them to IDs)
              before passing them to the model.
            - The `text_cleaner` should be initialized with the correct symbol dictionary
              and padding options.
            - The method uses the `infer` method to generate audio waveform for an input
              phonetic sentence.
            - It handles the generation of noise for diffusion sampling and manages the previous
              style embedding for each phonetic sentence.
            - The method also trims the generated audio to remove silence at the beginning and end
              (using the `offset_beg` and `offset_end` properties of the class).

        Returns:
            list: List of generated audio waveforms in numpy format.
        """
        # Initialize previous style and wavs
        wavs = []
        s_prev = None

        if isinstance(ref_s, str) and isinstance(spk_emb, str):
            # If ref_s is a path, compute style embedding.
            # Otherwise, both ref_s and spk_emb are assumed to be
            # reference speaker style embedding tensors or None
            ref_s = self.compute_style(ref_s, spk_emb, top_db=30)

        logger.info("Generating wavs from phoneme strings: %s", ph_strings)

        # Iterate over phoneme strings
        for ph_string in ph_strings:
            logger.debug("Processing phoneme string: %s", ph_string)

            if self.glob_noise is not None:
                # Use the same noise for the entire document (across phonetic strings)
                noise = self.glob_noise
            elif self.fix_noise_in_ph_string:
                # Use the same noise within a phonetic string (one phonetic line)
                noise = self.generate_noise()
            else:
                # New noise will be generated for each sentence
                noise = None

            # Iterate over sentences in the phonetic string
            for ph_sent in re.findall(r"[^.!?]*[.!?]", ph_string):
                if not ph_sent.strip():  # skip empty phonetic string
                    continue
                logger.debug("Phonetic sentence: %s", ph_sent)

                # Add padding and tokenize phonetic sentence
                ph_ids = self.text_cleaner(ph_sent, pad=True)

                # Perform inference => generate wav
                wav, s_prev = self.infer(
                    torch.tensor(ph_ids, dtype=torch.long, device=self.device).unsqueeze(0),
                    noise=noise,
                    s_prev=s_prev,
                    ref_s=ref_s,
                )

                # Collect wavs (without silence forced in training)
                wavs.append(wav[self.offset_beg : -self.offset_end])
                logger.debug("Phonetic sentence waveform generated")

        return wavs

    def infer(
        self,
        ph_ids,
        noise=None,
        s_prev=None,
        ref_s=None,
    ):
        """
        Perform inference with the StyleTTS2 model.

        Args:
            ph_ids (torch.Tensor): Tensor of phoneme IDs.
            noise (torch.Tensor): Noise tensor for diffusion.
            s_prev (torch.Tensor): Previous style embedding.
            ref_s (torch.Tensor): Reference speaker embedding.

        Returns:
            torch.Tensor: Generated audio waveform.
            torch.Tensor: Style embedding.
        """
        with torch.no_grad():
            logger.debug("Infering from phoneme IDs")
            logger.debug("Computing phonetic features")
            # Prepare input lengths and masks
            input_lengths = torch.tensor([ph_ids.shape[-1]], dtype=torch.long, device=self.device)
            text_mask = length_to_mask(input_lengths)

            # Phonetic features encoded from phonetic IDs (tokens) only
            t_en = self.model.text_encoder(ph_ids, input_lengths, text_mask)
            # Contextual phonetic features encoded by PL-BERT catching
            # linguistic context of the whole sentence
            bert_dur = self.model.bert(ph_ids, attention_mask=(~text_mask).int())
            # Transformed (and compressed) BERT-encoded linguistic features
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            # Generate the waveform from phonetic features
            return self.infer_from_ph_features(
                input_lengths,
                text_mask,
                t_en,
                bert_dur,
                d_en,
                noise,
                s_prev,
                ref_s,
            )
        # returns the generated waveform and predicted style embedding

    def infer_from_ph_features(
        self,
        input_lengths,
        text_mask,
        t_en,
        bert_en,
        d_en,
        noise=None,
        s_prev=None,
        ref_s=None,
    ):
        """Perform inference with the StyleTTS2 model using pre-computed text features.
        Args:
            t_en (torch.Tensor): Phonetic encoded features from phonetic IDs (tokens) only.
            bert_en (torch.Tensor): PL-BERT encoded phonetic features.
            d_en (torch.Tensor): # Transformed (and compressed) BERT-encoded linguistic features.
            noise (torch.Tensor): Noise tensor for diffusion sampling.
            s_prev (torch.Tensor): Previous style embedding.
            ref_s (torch.Tensor): Reference speaker embedding.
        """
        with torch.no_grad():
            logger.debug("Infering from phonetic features")
            # Sampling from the diffusion model
            # - generate style embedding from contextual PL-BERT based features
            # - represent timbre and prosody
            # - `ref_s` is 256-dimensional tensor
            if ref_s is None:
                # No reference speaker style embedding, typically for a single speaker model
                logger.debug("No reference speaker style embedding provided")
                s_pred = self._sampler(
                    self.generate_noise() if noise is None else noise,  # noise for diffusion
                    embedding=bert_en[0].unsqueeze(0),
                    embedding_scale=self.embedding_scale,
                    num_steps=self.diffusion_steps,
                ).squeeze(0)
            else:
                logger.debug("Reference speaker style embedding provided: %s", ref_s.shape)
                s_pred = self._sampler(
                    self.generate_noise() if noise is None else noise,  # noise for diffusion
                    embedding=bert_en[0].unsqueeze(0),
                    embedding_scale=self.embedding_scale,
                    features=ref_s,  # reference from the same speaker as the embedding
                    num_steps=self.diffusion_steps,
                ).squeeze(0)

            # Combine styles
            if s_prev is not None:
                logger.debug("Combining styles with previous style embedding")
                # convex combination of previous and current styles
                s_pred = self.t * s_pred + (1 - self.t) * s_prev

            s = s_pred[:, 128:]  # prosodic features: 128 (style) + 512 (speaker embedding)
            ref = s_pred[:, :128]  # timbre features

            # If reference speaker style embedding  `ref_s` is provided,
            # combine it with the generated style
            # - `alpha` controls the influence of the reference timbre
            #   (higher = more similar to the generated style,
            #   lower = more similar to the reference style)
            # - `beta` controls the influence of the reference prosody
            #   (higher = more similar to the generated style,
            #   lower = more similar to the reference style)
            if ref_s is not None:
                logger.debug("Combining styles with reference speaker style embedding")
                ref = self.alpha * ref + (1 - self.alpha) * ref_s[:, :128]
                s = self.beta * s + (1 - self.beta) * ref_s[:, 128:]
                s_pred = torch.cat([ref, s], dim=-1)

            # Style-conditioned phonetic features
            # - enriches linguistic features with style information
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            # xLSTM processed phonetic features
            x = self.model.predictor.lstm(d)
            x_mod = self.model.predictor.prepare_projection(x)  # 640 -> 512

            # Duration prediction: number of frames for each phoneme
            duration = self.model.predictor.duration_proj(x_mod)
            duration = torch.sigmoid(duration).sum(axis=-1) / self.speech_rate
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5  # add silence at the end

            # Create phoneme-audio alignment target matrix: [number of phones, number of frames]
            # - each phoneme contains sequence of 1 at the positions of its frames
            # - `pred_aln_trg[i,j] = 1` means phoneme i shall be pronunced at frame j
            # - sum of each row is the number of frames for each phoneme
            # - sum of each column is constant 1 => each frame is assigned to one phoneme
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                # For each phoneme, set the number of its frames to 1
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # Encode prosody: encoded prosodic audio-aligned features
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model.decoder.type == "hifigan":
                en_new = torch.zeros_like(en)
                en_new[:, :, 0] = en[:, :, 0]
                en_new[:, :, 1:] = en[:, :, 0:-1]
                en = en_new

            # Predict F0 and normalization (loudness)
            f0_pred, n_pred = self.model.predictor.F0Ntrain(en, s)
            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            # Decode the waveform
            # - `asr` is the phonetic features aligned with the audio frames (content)
            # - `f0_pred` is the predicted F0 (pitch) features aligned with the audio frames
            # - `n_pred` is the predicted normalization (loudness) features aligned with the audio frames
            # - `ref` is the style embedding (timbre) aligned with the audio frames
            out = self.model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))

            # Weird pulse at the end of the model, need to be fixed later
            # (without silence forced in training)
            return out.squeeze().cpu().numpy()[self.offset_beg : -self.offset_end], s_pred
            # return out.squeeze().cpu().numpy()[..., :-50], s_pred

    def reconstruct(self, mel_gt, en, spk_emb, p_en=None):
        """Reconstruct the waveform from the mel spectrogram.
        This method uses the decoder of the model to generate the waveform
        Args:
            mel_gt (torch.Tensor): Mel spectrogram.
            en (torch.Tensor): Encoded phonetic audio-aligned features.
            p_en (torch.Tensor): Predicted phonetic audio-aligned features.

        Returns:
            torch.Tensor: Reconstructed waveform.
        """
        with torch.no_grad():
            if p_en is not None:
                # Predict duration-related features from ground truth mel spectrogram
                s_dur = self.model.predictor_encoder(mel_gt.unsqueeze(1))
                # Predict F0 and norm
                f0, n = self.model.predictor.F0Ntrain(p_en, s_dur)
            else:
                # Extract real F0
                f0, _, _ = self.model.pitch_extractor(mel_gt.unsqueeze(1))
                f0 = f0.unsqueeze(0)
                # Extract real norm
                n = log_norm(mel_gt.unsqueeze(1)).squeeze(1)

            # Encode style from ground truth mel spectrogram
            acoust_style = self.model.acoustic_style_encoder(spk_emb)
            pros_style = self.model.prosodic_style_encoder(mel_gt.unsqueeze(1))
            style = torch.cat([acoust_style, pros_style], dim=1)
            # Decode
            y_pred = self.model.decoder(en, f0, n, style)

        # Return the waveform without silence at the beginning and end
        return y_pred.cpu().numpy().squeeze()[self.offset_beg : -self.offset_end]

    def compute_style(self, wavpath, spk_emb_path, top_db=30):
        """Compute style embedding from a waveform.

        Args:
            wavpath (torch.Tensor): Waveform numpy array or path to waveform file.
            top_db (int): Threshold for trimming silence. Default: 30.
        Note:
            If wav is a path, it will be loaded using librosa.
            The waveform will be trimmed for silence and resampled to the sampling rate
            specified in the configuration.
        Raises:
            ValueError: If wav is neither a numpy array nor a path to a wav file.
        Returns:
            torch.Tensor: Style embedding.
        """
        # TODO: Change loading wav to torchaudio
        with torch.no_grad():
            logger.debug("Computing style from wav file: %s", wavpath)
            wav, sr = librosa.load(wavpath, sr=self._config.preprocess_params.sr)

            if top_db is not None:
                # Trim silence
                logger.debug("Trimming silence from wav (%d dB", top_db)
                wav, _ = librosa.effects.trim(wav, top_db=top_db)
            if sr != self._config.preprocess_params.sr:
                # Resample if necessary
                logger.debug("Resampling wav from %d to %d", sr, self._config.preprocess_params.sr)
                wav = librosa.resample(wav, sr, self._config.preprocess_params.sr)

            wave_tensor = torch.from_numpy(wav).float()
            mel_tensor = self.audio_processor(wave_tensor).to(self.device)

            # Load speaker embedding
            spk_emb = torch.load(spk_emb_path)

            # Compute style embedding
            ref_acoust_style = self.model.acoustic_style_encoder(spk_emb)  # style = timbre
            ref_pros_style = self.model.prosodic_style_encoder(
                mel_tensor.unsqueeze(1)
            )  # style = prosody

        return torch.cat([ref_acoust_style, ref_pros_style], dim=1)

    def save_wav(self, wav, path):
        """Save wavs to a single wav file.

        Args:
            wav (list): list of waveform numpy arrays or torch tensors.
            path (string): Output wav file path
        """
        if isinstance(wav, (list, tuple)):
            # Assuming wav is a list/tuple of torch tensors
            if all(isinstance(w, torch.Tensor) for w in wav):
                wav_numpy = torch.cat(wav, dim=0).cpu().numpy()
            # Original numpy handling (kept for reference or if input is numpy)
            elif all(isinstance(w, np.ndarray) for w in wav):
                wav_numpy = np.concatenate(wav)
            else:
                # Handle mixed types or raise error
                raise TypeError(
                    "Input 'wav' must be a list/tuple of PyTorch tensors or NumPy arrays."
                )
        elif isinstance(wav, np.ndarray):
            wav_numpy = wav  # Already a numpy array
        elif isinstance(wav, torch.Tensor):
            wav_numpy = wav.cpu().numpy()
        else:
            raise TypeError("Input 'wav' must be a list/tuple, NumPy array, or PyTorch tensor.")

        # Save audio
        # torchaudio.save(path, wav_tensor.cpu().float(), self._config.preprocess_params.sr)
        write(path, self._config.preprocess_params.sr, wav_numpy)

    @property
    def model(self):
        """Get the model dictionary."""
        return self._model

    @model.setter
    def model(self, model):
        """
        Set the model dictionary and automatically put it in evaluation mode.

        Args:
            model: Dictionary of model components
        """
        self._model = model
        if self._model is not None:
            logger.debug("Model set, automatically switching to evaluation mode")
            self.to_eval()

    @property
    def offset_beg(self):
        """Get the beginning offset for audio generation."""
        return self._config.preprocess_params.silence_beg

    @property
    def offset_end(self):
        """Get the end offset for audio generation."""
        return self._config.preprocess_params.silence_end

    @property
    def t(self):
        """Get the `t` parameter for convex combination of styles."""
        return self._t

    @t.setter
    def t(self, value):
        """Set the `t` parameter for convex combination of styles."""
        if not 0 <= value <= 1:
            raise ValueError("t must be between 0 and 1.")
        self._t = value

    @property
    def alpha(self):
        """Get the alpha parameter for timbre similarity."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """Set the alpha parameter for timbre similarity."""
        if not 0 <= value <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        self._alpha = value

    @property
    def beta(self):
        """Get the beta parameter for prosody."""
        return self._beta

    @beta.setter
    def beta(self, value):
        """Set the beta parameter for prosody."""
        if not 0 <= value <= 1:
            raise ValueError("beta must be between 0 and 1.")
        self._beta = value

    @property
    def diffusion_steps(self):
        """Get the number of diffusion steps."""
        return self._diffusion_steps

    @diffusion_steps.setter
    def diffusion_steps(self, value):
        """Set the number of diffusion steps."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("diffusion_steps must be a positive integer.")
        self._diffusion_steps = value

    @property
    def embedding_scale(self):
        """Get the embedding scale."""
        return self._embedding_scale

    @embedding_scale.setter
    def embedding_scale(self, value):
        """Set the embedding scale."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("embedding_scale must be a positive number.")
        self._embedding_scale = value

    @property
    def speech_rate(self):
        """Get the speech rate."""
        return self._speech_rate

    @speech_rate.setter
    def speech_rate(self, value):
        """Set the speech rate."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("speech_rate must be a positive number.")
        self._speech_rate = value


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
