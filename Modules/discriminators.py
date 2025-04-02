import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

from .utils import get_padding

LRELU_SLOPE = 0.1


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    # real = x_stft[..., 0]
    # imag = x_stft[..., 1]

    return torch.abs(x_stft).transpose(2, 1)


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        use_spectral_norm=False,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ]
        )

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        y = y.squeeze(1)
        y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=None,
        hop_sizes=None,
        win_lengths=None,
        window="hann_window",
    ):
        super().__init__()
        # Set default values using None
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        self.discriminators = nn.ModuleList(
            [
                SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
                SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
                SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))
                ),
                norm_f(
                    Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))
                ),
                norm_f(
                    Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))
                ),
                norm_f(
                    Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """
    WavLM-based discriminator module for audio processing.
    This discriminator takes WavLM embeddings as input and processes them through
    a series of convolutional layers to determine if a given audio sample is real or generated.
    Args:
        slm_hidden (int, optional): Hidden dimension size of the WavLM model. Defaults to 768.
        slm_layers (int, optional): Number of layers in the WavLM model. Defaults to 13.
        initial_channel (int, optional): Initial number of channels for the convolutional layers. Defaults to 64.
        use_spectral_norm (bool, optional): Whether to use spectral normalization instead of weight normalization. Defaults to False.
    Inputs:
        x (Tensor): WavLM embeddings concatenated from all layers, with shape (batch_size, slm_hidden * slm_layers, time_steps)
    Returns:
        Tensor: Flattened discriminator output, with shape (batch_size, time_steps)
    Notes:
        - Feature maps from intermediate layers are collected but not returned in the current implementation
        - Uses leaky ReLU with a slope of 0.1 for activations
    """

    def __init__(self, slm_hidden=768, slm_layers=13, initial_channel=64, use_spectral_norm=False):
        """
        Initialize the discriminator model.
        This discriminator processes encoded style language model (SLM) features through a series
        of convolutional layers to determine if the input is real or generated.
        Parameters
        ----------
        slm_hidden : int, optional
            Hidden dimension size of the style language model, by default 768
        slm_layers : int, optional
            Number of layers from the style language model to use, by default 13
        initial_channel : int, optional
            Initial number of channels for the convolutional layers, by default 64
        use_spectral_norm : bool, optional
            Whether to use spectral normalization instead of weight normalization, by default False
        """
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.pre = norm_f(Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
                norm_f(
                    nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)
                ),
                norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
            ]
        )

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Forward pass for the discriminator.
        Args:
            x (torch.Tensor): Input tensor to the discriminator.
        Returns:
            torch.Tensor: Flattened output tensor after applying convolutions and activation functions.
        Note:
            This method applies a pre-processing step, followed by a series of convolutional layers
            with leaky ReLU activations. It collects feature maps during processing, applies a final
            convolution, and then flattens the result.
        """
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x
