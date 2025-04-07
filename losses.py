from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torchaudio
import whisper
from transformers import AutoModel, WhisperConfig, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=window,
        )

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std

        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=None,
        hop_sizes=None,
        win_lengths=None,
        window=torch.hann_window,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """


def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_dg = torch.median((dr - dg))
        l_rel = torch.mean((((dr - dg) - m_dg) ** 2)[dr < dg + m_dg])
        loss += tau - F.relu(tau - l_rel)
    return loss


def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_dg = torch.median((dr - dg))
        l_rel = torch.mean((((dr - dg) - m_dg) ** 2)[dr < dg + m_dg])
        loss += tau - F.relu(tau - l_rel)
    return loss


class GeneratorLoss(torch.nn.Module):
    """Computes the total generator loss using MPD (Multi-Period Discriminator)
    and MSD (Multi-Scale Discriminator).

    This class implements the generator loss calculation for adversarial training
    of audio generation models. It combines feature matching loss, generator adversarial loss,
    and relative loss components.

    Args:
        mpd (torch.nn.Module): Multi-Period Discriminator module
        msd (torch.nn.Module): Multi-Scale Discriminator module

    Methods:
        forward(y, y_hat): Computes the total generator loss
            Args:
                y (torch.Tensor): Ground truth audio waveform
                y_hat (torch.Tensor): Generated audio waveform
            Returns:
                torch.Tensor: Mean of combined generator losses including feature matching,
                             adversarial, and relative losses
    """

    def __init__(self, mpd, msd):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y, y_hat):
        """
        Computes the combined generator loss for the HiFi-GAN model.

        This method calculates multiple loss components:
        - Feature matching loss from Multi-Period Discriminator (MPD)
        - Feature matching loss from Multi-Scale Discriminator (MSD)
        - Generator adversarial loss from MPD
        - Generator adversarial loss from MSD
        - Relative logistic loss between real and generated samples

        Args:
            y (Tensor): Ground truth audio waveform
            y_hat (Tensor): Generated audio waveform

        Returns:
            Tensor: Mean of the combined generator loss including feature matching,
                   adversarial, and relative logistic components
        """
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)

        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + generator_TPRLS_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel

        return loss_gen_all.mean()


class DiscriminatorLoss(torch.nn.Module):
    """It combines the following components:
    - MPD (Multi-Period Discriminator) loss
    - MSD (Multi-Scale Discriminator) loss
    - Relative loss using TPRLS (Two-Path Relative Loss Strategy)

    Methods:
        forward(y, y_hat): Calculates the total discriminator loss
                y (torch.Tensor): Real audio samples
                y_hat (torch.Tensor): Generated audio samples
            Returns:
                torch.Tensor: Mean of combined discriminator losses
    """

    def __init__(self, mpd, msd):
        """Initialize MultiScale and MultiPeriod discriminators.
        Args:
            mpd (nn.Module): Multi-period discriminator module
            msd (nn.Module): Multi-scale discriminator module
        """
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y, y_hat):
        """
        Forward pass for the discriminator loss calculation.
        Args:
            y (torch.Tensor): Ground truth waveform.
            y_hat (torch.Tensor): Generated/predicted waveform.
        Returns:
            torch.Tensor: Mean discriminator loss combining MPD (Multi-Period Discriminator),
                         MSD (Multi-Scale Discriminator), and relative losses.
        Details:
            - Computes MPD (Multi-Period Discriminator) loss using real and generated samples
            - Computes MSD (Multi-Scale Discriminator) loss using real and generated samples
            - Calculates relative loss using TPRLS (Two-Path Regularization Loss Strategy)
            - Combines all losses into final discriminator loss
        """
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        d_loss = loss_disc_s + loss_disc_f + loss_rel

        return d_loss.mean()


# #####################
# MIXED PRECISION
# #####################


class WhisperEncoderOnly(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)

    def forward(self, input_features, attention_mask=None):
        return self.encoder(input_features, attention_mask)


class SLMLoss(torch.nn.Module, ABC):
    """
    Abstract base class for speech language model losses.
    This provides a common interface for different speech representation models
    like WavLM, Whisper, etc. used for loss calculation in speech synthesis.
    """

    def __init__(self, wd, model_sr, slm_sr=16000):
        super().__init__()
        self.wd = wd
        self.model_sr = model_sr
        self.slm_sr = slm_sr
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    @abstractmethod
    def forward(self, wav, y_rec):
        """Forward pass for processing audio through the speech language model."""
        pass

    @abstractmethod
    def generator(self, y_rec):
        """Calculate generator loss."""
        pass

    @abstractmethod
    def discriminator(self, wav, y_rec):
        """Calculate discriminator loss."""
        pass

    @abstractmethod
    def discriminator_forward(self, wav):
        """Forward pass through discriminator for evaluation."""
        pass


class WhisperLoss(SLMLoss):
    """
    WhisperLoss class for speech representation learning using Whisper embeddings.
    This class implements a loss function based on the Whisper model for tasks such as
    voice conversion, speech synthesis, or audio generation. It extends the SLMLoss base class
    and provides functionality to calculate various losses (generator, discriminator, feature matching)
    using Whisper's encoder representations.
    The class uses a Whisper encoder model to extract audio embeddings and compares them
    between original and reconstructed audio samples. It supports both adversarial training
    through generator and discriminator losses, as well as feature matching for representation
    alignment.
    Attributes:
        slm (WhisperEncoderOnly): The Whisper encoder model used for feature extraction
        wd: Waveform discriminator model
    Methods:
        forward: Main method to compute various losses based on the specified mode
        generator: Computes generator adversarial loss for audio reconstruction
        discriminator: Computes discriminator loss to distinguish real from generated audio
        discriminator_forward: Performs forward pass through discriminator for inference
    Example:
        loss_fn = WhisperLoss("openai/whisper-large-v2", wd=discriminator, model_sr=24000)
        loss = loss_fn(original_audio, generated_audio)
    """

    def __init__(self, model_name, wd, model_sr, slm_sr=16000):
        """
        Initialize WhisperLoss with a pre-trained Whisper model.
        Parameters:
        ----------
        model_name : str
            The name or path of the pre-trained Whisper model to load.
        wd : float
            Waveform discriminator model.
        model_sr : int
            Sampling rate of the model input.
        slm_sr : int, optional
            Sampling rate for the Whisper model, default is 16000.
        Notes:
        -----
        This initializes a WhisperEncoderOnly model for use as a semantic loss component,
        converting the model to bfloat16 precision for efficiency.
        """
        super().__init__(wd, model_sr, slm_sr)
        print(f"Using Whisper for SLM loss: {model_name}")
        # # Load the Whisper large-v2 model encoder-only configuration and
        # # set it to be non-decoder
        # config = WhisperConfig.from_pretrained("Respair/Whisper_Large_v2_Encoder_Block")
        # config.is_encoder_decoder = False
        # config.use_cache = False

        # Load the full model and keep only the encoder
        full_model = WhisperEncoderOnly.from_pretrained(
            model_name,
            # config=config,
            torch_dtype=torch.bfloat16,
        )
        full_model.config.is_encoder_decoder = False
        full_model.config.use_cache = False

        # Initialize the encoder-only model with the same configuration
        model = WhisperEncoderOnly(full_model.config)
        # Load encoder weights from the full model
        model.encoder.load_state_dict(full_model.encoder.state_dict())
        del full_model  # Free up memory

        # Set the number of mel bins based on the model config
        self.n_mels = model.config.num_mel_bins
        self.slm = model.to(torch.bfloat16)

    def forward(self, wav, y_rec):
        """
        Computes various losses for training speech models using embeddings from whisper model.
        This method computes different losses depending on the mode specified:
        - When `generator=True`: Calculates generator loss using discriminator outputs on reconstructed audio
        - When `discriminator=True`: Computes discriminator loss by comparing real vs generated audio embeddings
        - When `discriminator_forward=True`: Forward pass through discriminator for real audio only
        - Default mode: Calculates feature matching loss between real and reconstructed audio embeddings
        Parameters
        ----------
        wav : torch.Tensor
            Input waveform tensor (original/real audio)
        y_rec : torch.Tensor
            Reconstructed waveform tensor (generated audio)
        generator : bool, default=False
            Whether to compute generator loss
        discriminator : bool, default=False
            Whether to compute discriminator loss
        discriminator_forward : bool, default=False
            Whether to only perform forward pass through discriminator
        Returns
        -------
        torch.Tensor
            Loss value depending on the mode:
            - Generator loss when generator=True
            - Discriminator loss when discriminator=True
            - Discriminator outputs when discriminator_forward=True
            - Feature matching loss by default
        """
        wav = wav.squeeze(1)
        y_rec = y_rec.squeeze(1)

        wav = whisper.pad_or_trim(wav)
        wav = whisper.log_mel_spectrogram(wav, n_mels=self.n_mels)

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec, n_mels=self.n_mels)

        with torch.no_grad():
            wav_embeddings = self.slm.encoder(
                wav.to(torch.bfloat16), output_hidden_states=True
            ).hidden_states

            y_rec_embeddings = self.slm.encoder(
                y_rec.to(torch.bfloat16), output_hidden_states=True
            ).hidden_states

        floss = 0
        for er, eg in zip(
            [e.to(torch.float32) for e in wav_embeddings],
            [e.to(torch.float32) for e in y_rec_embeddings],
        ):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        """Calculate generator loss for adversarial training.
        This function processes reconstructed audio samples to calculate the generator loss in
        an adversarial training framework.

            y_rec (Tensor): Reconstructed audio tensor with shape [batch_size, 1, length]

            Tensor: Generator loss value as a float32 scalar tensor. The loss encourages the
            generator to produce samples that the discriminator will classify as real.

            1. Removes channel dimension from input tensor
            2. Processes the reconstructed audio with Whisper's pad_or_trim and log_mel_spectrogram
            5. Passes embeddings through the discriminator (wd)
            6. Calculates generator loss using (1 - D(G(z)))² formulation
        """
        y_rec = y_rec.squeeze(1)

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec, n_mels=self.n_mels)

        with torch.no_grad():
            y_rec_embeddings = self.slm.encoder(
                y_rec.to(torch.bfloat16), output_hidden_states=True
            ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings.to(torch.float32))
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen.to(torch.float32)

    def discriminator(self, wav, y_rec):
        """
        This function processes original (wav) and reconstructed (y_rec) audio samples through a
        speech language model (SLM) to extract embeddings, then evaluates them with a discriminator
        to calculate the adversarial loss.

        Args:
            wav (Tensor): Original audio tensor to discriminate, with shape [batch_size, 1, length]
            y_rec (Tensor): Reconstructed audio tensor, with shape [batch_size, 1, length]

        Returns:
            Tensor: Discriminator loss value as a float32 scalar tensor. This loss combines the real
            sample loss (r_loss) and generated sample loss (g_loss), encouraging the discriminator
            to assign high scores to real samples and low scores to generated ones.

        Process:
            1. Removes channel dimension from input tensors
            2. Processes both samples with Whisper's pad_or_trim and log_mel_spectrogram
            3. Extracts embeddings from all hidden states of the SLM encoder
            4. Formats embeddings by stacking, transposing and flattening
            5. Computes discriminator outputs for both real and generated samples
            6. Calculates adversarial loss using mean squared error formulation
        """
        wav = wav.squeeze(1)
        y_rec = y_rec.squeeze(1)

        wav = whisper.pad_or_trim(wav)
        wav = whisper.log_mel_spectrogram(wav, n_mels=self.n_mels)

        y_rec = whisper.pad_or_trim(y_rec)
        y_rec = whisper.log_mel_spectrogram(y_rec, n_mels=self.n_mels)

        with torch.no_grad():
            wav_embeddings = self.slm.encoder(
                wav.to(torch.bfloat16), output_hidden_states=True
            ).hidden_states
            y_rec_embeddings = self.slm.encoder(
                y_rec.to(torch.bfloat16), output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings.to(torch.float32))
        y_d_gs = self.wd(y_rec_embeddings.to(torch.float32))

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean().to(torch.float32)

    def discriminator_forward(self, wav):
        # Squeeze the channel dimension if it's unnecessary
        wav = wav.squeeze(1)  # Adjust this line if the channel dimension is not at dim=1

        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_16 = whisper.pad_or_trim(wav_16)
            wav_16 = whisper.log_mel_spectrogram(wav_16, n_mels=self.n_mels)

            wav_embeddings = self.slm.encoder(
                wav_16.to(torch.bfloat16), output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings.to(torch.float32))

        return y_d_rs


class WavLMLoss(SLMLoss):
    """
    WavLMLoss class for computing losses based on WavLM embeddings.

    This class extends SLMLoss to provide loss calculations using WavLM model
    embeddings for speech synthesis evaluation. It can be used to calculate
    feature matching losses between original and synthesized audio, as well as
    adversarial losses when using WavLM in a GAN-based training setup.

    The class handles resampling between the model's sample rate and the WavLM
    expected sample rate (default 16kHz), and provides methods for both feature
    matching and adversarial training.

    Attributes:
        slm: The WavLM model used for audio embedding extraction
        wd: Waveform discriminator model

    Methods:
        forward: Calculates feature matching loss between original and reconstructed audio
        generator: Computes generator loss for adversarial training
        discriminator: Calculates discriminator loss between real and fake samples
        discriminator_forward: Forward pass through discriminator using WavLM embeddings
    """

    def __init__(self, model_name, wd, model_sr, slm_sr=16000):
        """
        Initialize WavLM loss component.

        Parameters
        ----------
        model_name : str
            The name or path of the pre-trained model to be loaded.
        wd : float
            Waveform discriminator model
        model_sr : int
            Sample rate of the model input.
        slm_sr : int, optional
            Sample rate for the speech language model, defaults to 16000 Hz.
        """
        super().__init__(wd, model_sr, slm_sr)
        print(f"Using WavLM for SLM loss: {model_name}")
        self.slm = AutoModel.from_pretrained(model_name)

    def forward(self, wav, y_rec):
        """
        Calculates a feature matching loss between the original and reconstructed waveforms.
        The method resamples both waveforms to 16kHz and computes embeddings using a
        speech language model (slm). The loss is calculated as the mean absolute difference
        between corresponding embedding layers.
        Args:
            wav (torch.Tensor): The original audio waveform.
            y_rec (torch.Tensor): The reconstructed audio waveform.
            generator (bool, optional): Unused parameter. Defaults to False.
            discriminator (bool, optional): Unused parameter. Defaults to False.
            discriminator_forward (bool, optional): Unused parameter. Defaults to False.
        Returns:
            torch.Tensor: The mean feature matching loss across all embedding layers.
        """
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.slm(input_values=wav_16, output_hidden_states=True).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.slm(
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        """
        Compute the generator loss for adversarial training.
        This method calculates the generator component of GAN loss using discriminator outputs.
        It first resamples the reconstructed audio, extracts WavLM embeddings, and passes
        them through the discriminator to compute how well the generator fools the discriminator.
        Args:
            y_rec (torch.Tensor): Reconstructed audio waveform from the generator.
        Returns:
            torch.Tensor: The generator loss value as a scalar tensor, calculated as mean((1 - D(G(x)))²).
            Lower values indicate the generator is better at fooling the discriminator.
        """
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.slm(input_values=y_rec_16, output_hidden_states=True).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec):
        """
        Calculates the discriminator loss between original and reconstructed audio waveforms.
        This method extracts embeddings from both the original and reconstructed waveforms
        using a WavLM model, processes them through the waveform discriminator, and computes
        the adversarial loss that helps distinguish between real and generated samples.
        Args:
            wav (Tensor): The original audio waveform (ground truth).
            y_rec (Tensor): The reconstructed/generated audio waveform.
        Returns:
            Tensor: The mean discriminator loss, combining the real sample loss (r_loss)
                    and generated sample loss (g_loss).
        """
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.slm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.slm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        """
        Forward pass through the discriminator using WavLM embeddings.
        This method processes an input waveform through the WavLM model to extract
        embeddings, which are then passed through the discriminator. The gradient
        calculation is disabled during this process.
        Args:
            wav (torch.Tensor): The input waveform tensor.
        Returns:
            torch.Tensor: The discriminator's output predictions based on WavLM embeddings.
        """
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.slm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs


def create_slm_loss(config, wd, model_sr):
    """
    Factory function to create the appropriate speech language model loss based on config.

    Args:
        config (dict): Configuration dictionary with model settings
        wd: Waveform discriminator model
        model_sr (int): Model sample rate

    Returns:
        SLMLoss: An instance of the appropriate loss class
    """
    model_name = config.model.lower()
    slm_sr = config.get("sr", 16000)

    if "wavlm" in model_name:
        return WavLMLoss(model_name, wd, model_sr, slm_sr)
    if "whisper" in model_name:
        return WhisperLoss(model_name, wd, model_sr, slm_sr)

    raise ValueError(f"Unsupported model name: {model_name}. Use 'wavlm' or 'whisper'.")
