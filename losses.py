import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel


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
    """Computes the total generator loss using MPD (Multi-Period Discriminator) and MSD (Multi-Scale Discriminator).

    This class implements the generator loss calculation for adversarial training of audio generation models.
    It combines feature matching loss, generator adversarial loss, and relative loss components.

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


class WavLMLoss(torch.nn.Module):
    """WavLMLoss module for comparing and discriminating audio embeddings using WavLM model.
    This class implements a loss module that uses the WavLM model to extract embeddings from audio
    and compute various losses for training speech synthesis models.
    Args:
        model (str): Path or identifier for the pretrained WavLM model
        wd (nn.Module): Discriminator module for WavLM embeddings
        model_sr (int): Sample rate of the input audio
        slm_sr (int, optional): Target sample rate for WavLM model. Defaults to 16000.
    Methods:
        forward(wav, y_rec): Computes feature matching loss between original and reconstructed audio
        generator(y_rec): Computes generator loss using discriminator predictions
        discriminator(wav, y_rec): Computes discriminator loss for real and generated samples
        discriminator_forward(wav): Forward pass through discriminator for real samples only
    The class provides functionality for:
    - Feature matching between original and reconstructed audio using WavLM embeddings
    - Adversarial training with a discriminator operating on WavLM embeddings
    - Resampling audio to match WavLM's expected sample rate
    """

    def __init__(self, model, wd, model_sr, slm_sr=16000):
        """Initialize the model with specified parameters.
        Args:
            model (str): Path or identifier for the pre-trained WavLM model.
            wd (float): Weight decay parameter for optimization.
            model_sr (int): Sample rate of the input audio for the model.
            slm_sr (int, optional): Target sample rate for speech language model. Defaults to 16000.
        """
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def forward(self, wav, y_rec):
        """Forward pass for feature loss calculation.
        This method computes the feature loss between original and reconstructed audio
        using WavLM embeddings. It resamples both signals to 16kHz and extracts
        embeddings using the WavLM model.
        Args:
            wav (Tensor): Original input waveform
            y_rec (Tensor): Reconstructed waveform
        Returns:
            Tensor: Mean feature loss calculated as L1 distance between original
                    and reconstructed WavLM embeddings across all layers
        Note:
            Both input tensors should be audio waveforms with same sampling rate.
            The method handles resampling to 16kHz internally.
        """
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
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
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
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
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
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
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs
