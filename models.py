# coding:utf-8

import copy
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from munch import munchify, Munch
from torch.nn.utils import spectral_norm, weight_norm
from xlstm import mLSTMBlockConfig, mLSTMLayerConfig, xLSTMBlockStack, xLSTMBlockStackConfig

from logger import get_logger
from Modules.diffusion.diffusion import AudioDiffusionConditional
from Modules.diffusion.modules import StyleTransformer1d, Transformer1d
from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavDiscriminator,
)
from Modules.hifigan import Decoder as HifiDecoder
from Modules.istftnet import Decoder as ISTFTDecoder
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

# Setup logger
logger = get_logger(__name__)


class LearnedDownSample(nn.Module):
    """
    A learnable downsampling module that applies different types of convolutions based on the specified layer type.

    This module implements three different downsampling strategies:
    - "none": Identity mapping (no downsampling)
    - "timepreserve": Downsamples only in the frequency dimension while preserving time
    - "half": Downsamples both time and frequency dimensions by half

    Args:
        layer_type (str): Type of downsampling to apply. Must be one of ["none", "timepreserve", "half"]
        dim_in (int): Number of input channels

    Raises:
        RuntimeError: If layer_type is not one of the supported options

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_in, height, width)

    Returns:
        torch.Tensor: Downsampled tensor with dimensions depending on layer_type:
            - "none": Same shape as input
            - "timepreserve": (batch_size, dim_in, height//2, width)
            - "half": (batch_size, dim_in, height//2, width//2)
    """

    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1
                )
            )
        else:
            raise RuntimeError(
                f"Got unexpected donwsampletype {self.layer_type}, expected is [none, timepreserve, half]"
            )

    def forward(self, x):
        return self.conv(x)


class LearnedUpSample(nn.Module):
    """
    Learned upsampling module for neural networks.

    This module provides different types of learned upsampling operations using transposed convolutions,
    allowing for trainable upsampling instead of fixed interpolation methods.

    Args:
        layer_type (str): Type of upsampling to perform. Options are:
            - "none": Identity operation (no upsampling)
            - "timepreserve": Upsamples only in the time dimension (height) by factor of 2,
                             preserving the frequency dimension using depthwise convolution
            - "half": Upsamples both dimensions by factor of 2 using depthwise convolution
        dim_in (int): Number of input channels

    Raises:
        RuntimeError: If layer_type is not one of the supported options

    Forward:
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in, height, width)

        Returns:
            torch.Tensor: Upsampled tensor with dimensions depending on layer_type:
                - "none": Same shape as input
                - "timepreserve": (batch_size, dim_in, height*2, width)
                - "half": (batch_size, dim_in, height*2, width*2)
    """

    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 1),
                stride=(2, 1),
                groups=dim_in,
                output_padding=(1, 0),
                padding=(1, 0),
            )
        elif self.layer_type == "half":
            self.conv = nn.ConvTranspose2d(
                dim_in,
                dim_in,
                kernel_size=(3, 3),
                stride=(2, 2),
                groups=dim_in,
                output_padding=1,
                padding=1,
            )
        else:
            raise RuntimeError(
                f"Got unexpected upsampletype {self.layer_type}, expected is [none, timepreserve, half]"
            )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """
    A downsampling module that applies different pooling strategies based on the specified layer type.

    This module provides three different downsampling approaches:
    - "none": No downsampling, returns input unchanged
    - "timepreserve": Downsamples only the frequency dimension while preserving time
    - "half": Downsamples both dimensions by half

    Args:
        layer_type (str): The type of downsampling to apply. Must be one of:
            - "none": No downsampling
            - "timepreserve": Average pool with kernel (2, 1) to preserve time dimension
            - "half": Average pool with kernel 2x2, padding input if needed

    Forward Args:
        x (torch.Tensor): Input tensor to be downsampled

    Returns:
        torch.Tensor: Downsampled tensor according to the specified layer_type

    Raises:
        RuntimeError: If layer_type is not one of the supported values
    """

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                f"Got unexpected donwsampletype {self.layer_type}, expected is [none, timepreserve, half]"
            )


class UpSample(nn.Module):
    """
    A neural network module for upsampling tensors with different strategies.

    This module provides flexible upsampling functionality with three different modes:
    - 'none': No upsampling, returns input tensor unchanged
    - 'timepreserve': Upsamples only the spatial dimension while preserving time dimension (scale factor 2x1)
    - 'half': Upsamples both dimensions equally (scale factor 2x2)

    Args:
        layer_type (str): The type of upsampling to perform. Must be one of 'none', 'timepreserve', or 'half'.

    Raises:
        RuntimeError: If an unsupported layer_type is provided.

    Example:
        >>> upsampler = UpSample('timepreserve')
        >>> output = upsampler(input_tensor)
    """

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.interpolate(x, scale_factor=(2, 1), mode="nearest")
        elif self.layer_type == "half":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            raise RuntimeError(
                f"Got unexpected upsampletype {self.layer_type}, expected is [none, timepreserve, half]"
            )


class ResBlk(nn.Module):
    """
    Residual Block with optional normalization and downsampling.

    A convolutional residual block that implements skip connections with learnable shortcuts
    when input and output dimensions differ. Supports various downsampling strategies and
    optional instance normalization.

    Args:
        dim_in (int): Number of input channels
        dim_out (int): Number of output channels
        actv (nn.Module, optional): Activation function. Defaults to nn.LeakyReLU(0.2)
        normalize (bool, optional): Whether to apply instance normalization. Defaults to False
        downsample (str, optional): Downsampling strategy. Defaults to "none"

    Attributes:
        actv: Activation function used in the residual path
        normalize: Flag indicating whether normalization is applied
        downsample: Downsampling module for the shortcut path
        downsample_res: Learned downsampling module for the residual path
        learned_sc: Flag indicating whether a learned shortcut connection is needed
        conv1: First convolutional layer with spectral normalization
        conv2: Second convolutional layer with spectral normalization
        norm1: First instance normalization layer (if normalize=True)
        norm2: Second instance normalization layer (if normalize=True)
        conv1x1: 1x1 convolutional layer for learned shortcut (if dim_in != dim_out)

    Returns:
        torch.Tensor: Output tensor with unit variance scaling (divided by sqrt(2))

    Note:
        The output is scaled by 1/sqrt(2) to maintain unit variance in the residual network.
    """

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample="none"):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AcousticStyleEncoder(nn.Module):
    """
    Acoustic Style Encoder that combines internal style encoding with external speaker embeddings.

    This module provides flexible style encoding capabilities with three operational modes:
    - 'internal': Uses only the internal style encoder based on acoustic features
    - 'external': Uses only external speaker embeddings
    - 'mix': Combines both internal and external representations with learnable or fixed weights

    The encoder supports both learnable and fixed gating mechanisms for mixing internal
    and external style representations, allowing for adaptive or predetermined blending
    strategies during training and inference.

    Attributes:
        spk_proj (nn.Module): Projects speaker embeddings to style dimension when needed
        style_encoder (StyleEncoder): Internal style encoder for acoustic features
        gate_param (nn.Parameter or torch.Tensor): Mixing weight parameter for blending modes

    Properties:
        style_dim (int): Output dimension of the style embeddings
        mode (str): Current operational mode ('internal', 'external', or 'mix')
        learnable_gate (bool): Whether the mixing gate is trainable

    Methods:
        forward(x, spk_emb, is_warmup): Main forward pass with mode-dependent processing
        forward_internal(x): Internal style encoding only
        forward_external(spk_emb): External speaker embedding processing only
        forward_mix(x, spk_emb, is_warmup): Mixed internal and external processing
    """

    def __init__(
        self,
        dim_in=48,
        spk_emb_dim=512,
        style_dim=128,
        max_conv_dim=384,
        mode="internal",
        mix_weight=0.0,
        learnable_gate=True,
    ):
        """Initialize the Acoustic Style Encoder.
        Args:
            dim_in (int, optional): Input dimension for the projection layer. Defaults to 48.
            spk_emb_dim (int, optional): Dimension of the external speaker embedding.
                Defaults to 512.
            style_dim (int, optional): Output dimension for the projection layer,
                representing the style embedding dimension. Defaults to 128.
            max_conv_dim (int, optional): Maximum dimension for the convolutional layers
                in the style encoder. Defaults to 384.
            mix_weight (float, optional): Initial value for the trainable mix weight.
                Defaults to 0.0 (0.5 after sigmoid activation).
            learnable_gate (bool, optional): If True, the gate is a learnable parameter with
                initialization close to 0.5 (after sigmoid activation when `mix_weight` is 0.0)
                to allow for a balanced mix between the internal style encoder output and the
                external speaker embedding.
                If False, it is a fixed value `mix_weight`. Defaults to True.
        """
        super().__init__()
        self._learnable_gate = learnable_gate
        self._mode = mode
        self._style_dim = style_dim

        # self.spk_proj = nn.Sequential(
        #     nn.Linear(spk_emb_dim, style_dim),  # Reduce the dimension of the speaker embedding
        #     nn.LayerNorm(style_dim),  # Stabilize the statistics
        # )
        # Reduce the dimension of the speaker embedding
        if mode != "internal" and spk_emb_dim > style_dim:
            self.spk_proj = nn.Linear(spk_emb_dim, style_dim)
        else:
            self.spk_proj = nn.Identity()

        self.style_encoder = StyleEncoder(
            dim_in=dim_in,
            style_dim=style_dim,
            max_conv_dim=max_conv_dim,
        )

        if learnable_gate:
            # Initialize the gate parameter as a trainable parameter
            # The gate parameter is initialized to a value close to 0.5 (after sigmoid activation)
            # to allow for a balanced mix between the internal style encoder output and
            # the external speaker embedding. This allows the model to learn the optimal mix
            # during training
            self.gate_param = nn.Parameter(
                torch.full(
                    (style_dim,),
                    mix_weight,
                    dtype=torch.float32,
                )
            )
        else:
            # If not learnable, register a buffer to hold the untrainable gate parameter
            # This will not be updated during training, but can still be used in the forward pass
            if mode == "external":
                w = 1
            elif mode == "internal":
                w = 0
            else:
                w = mix_weight
            self.register_buffer(
                "gate_param",
                torch.full(
                    (style_dim,),
                    w,
                    dtype=torch.float32,
                ),
            )

    def forward(self, x, spk_emb=None, is_warmup=False):
        """
        Forward pass of the module.
        Args:
            x (torch.Tensor): Internal style embedding.
            spk_emb (torch.Tensor): External speaker embedding.
        Returns:
            torch.Tensor: The output style tensor.
        """
        if self._mode == "internal" or spk_emb is None:
            # Internal style encoding is used
            return self.forward_internal(x)
        elif self._mode == "external":
            # External speaker embedding is provided, project it and normalize
            return self.forward_external(spk_emb)

        # Both internal and external style encodings are used => style will be mixed
        return self.forward_mix(x, spk_emb, is_warmup)

    def forward_internal(self, x):
        """
        Forward pass for internal style encoding only.
        Args:
            x (torch.Tensor): Internal style embedding.
        Returns:
            torch.Tensor: The output style tensor from the internal style encoder.
        """
        return self.style_encoder(x)

    def forward_external(self, spk_emb):
        """
        Forward pass for external speaker embedding only.
        Args:
            spk_emb (torch.Tensor): External speaker embedding.
        Returns:
            torch.Tensor: The output style tensor from the external speaker embedding projection.
        """
        if spk_emb is None:
            raise ValueError("External speaker embedding is required when mode is 'external'.")
        # normalized = F.normalize(projected, p=2, dim=-1)
        return self.spk_proj(spk_emb)

    def forward_mix(self, x, spk_emb, is_warmup=False):
        """
        Forward pass for mixing internal and external style embeddings.
        Args:
            x (torch.Tensor): Internal style embedding.
            spk_emb (torch.Tensor): External speaker embedding.
            is_warmup (bool, optional): If True, only use the internal style encoder output.
        Returns:
            torch.Tensor: The output style tensor after mixing.
        """
        style_intern = self.forward_internal(x)
        if is_warmup:
            # During warmup, use the internal style encoder output only
            return style_intern
        style_extern = self.forward_external(spk_emb)
        # Setup gate parameter:
        # - If learnable, use sigmoid activation to ensure it is between 0 and 1
        #   with a default value of `mix_weight=0` being 0.5 after sigmoid activation.
        # - If not learnable, use the fixed value of `mix_weight`.
        g = torch.sigmoid(self.gate_param) if self._learnable_gate else self.gate_param
        # Fusion: (1 - g) * style_intern + g * style_extern
        # g: shape (style_dim,) or (batch, style_dim) (broadcasted to match batch)
        # style_intern: shape (batch, style_dim)
        # style_extern: shape (batch, style_dim)
        # Broadcasting ensures elementwise mixing per style dimension.
        return (1 - g) * style_intern + g * style_extern

    @property
    def style_dim(self):
        """
        Returns the style dimension of the encoder.
        This is the output dimension of the style encoder.
        """
        return self._style_dim

    @property
    def mode(self):
        """
        Returns the mode of the style encoder.
        This indicates whether the encoder is using 'internal', 'external', or 'mix' mode.
        """
        return self._mode

    @property
    def learnable_gate(self):
        """
        Returns whether the gate parameter is learnable.
        If True, the gate parameter is a trainable parameter.
        If False, the gate parameter is a fixed value.
        """
        return self._learnable_gate


class StyleEncoder(nn.Module):
    """
    A neural network module for encoding style information from spectrograms.

    The StyleEncoder processes 2D input spectrograms through a series of convolutional layers
    with residual blocks and downsampling to extract style embeddings. The architecture
    consists of shared convolutional layers followed by an unshared linear projection.

    Args:
        dim_in (int, optional): Initial input dimension for the first convolutional layer.
            Defaults to 48.
        style_dim (int, optional): Output dimension of the style embedding. Defaults to 48.
        max_conv_dim (int, optional): Maximum number of convolutional channels to prevent
            excessive memory usage. Defaults to 384.

    Architecture:
        - Initial 3x3 convolution with spectral normalization
        - 4 residual blocks with downsampling and channel doubling (up to max_conv_dim)
        - LeakyReLU activation
        - 5x5 convolution with spectral normalization
        - Adaptive average pooling to 1x1
        - Final LeakyReLU and linear projection to style_dim

    Forward:
        Args:
            x (torch.Tensor): Input spectrogram tensor of shape (batch_size, 1, height, width)

        Returns:
            torch.Tensor: Style embedding tensor of shape (batch_size, style_dim)
    """

    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample="half"))
            dim_in = dim_out

        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.LeakyReLU(0.2))

        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        # normalized = F.normalize(s, p=2, dim=-1)
        return s


class LinearNorm(torch.nn.Module):
    """
    A linear transformation layer with Xavier uniform weight initialization.

    This module wraps a standard PyTorch Linear layer and applies Xavier uniform
    initialization to the weights based on the specified activation function gain.

    Args:
        in_dim (int): Size of input features.
        out_dim (int): Size of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias.
            Defaults to True.
        w_init_gain (str, optional): Name of the nonlinearity used to calculate the gain
            for Xavier initialization. Defaults to "linear".

    Attributes:
        linear_layer (torch.nn.Linear): The underlying linear transformation layer.

    Example:
        >>> linear = LinearNorm(256, 128, w_init_gain='relu')
        >>> output = linear(input_tensor)
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class Discriminator2d(nn.Module):
    """
    A 2D discriminator network for adversarial training.

    This discriminator processes 2D spectral representations (e.g., spectrograms) and outputs
    discrimination scores along with intermediate features. It uses spectral normalization
    and residual blocks with downsampling to progressively reduce spatial dimensions while
    increasing feature depth.

    Args:
        dim_in (int, optional): Initial number of input channels/features. Defaults to 48.
        num_domains (int, optional): Number of output domains for discrimination. Defaults to 1.
        max_conv_dim (int, optional): Maximum number of convolutional channels. Defaults to 384.
        repeat_num (int, optional): Number of residual blocks to repeat. Defaults to 4.

    Architecture:
        - Initial 2D convolution from 1 to dim_in channels
        - Sequence of ResBlk layers with progressive downsampling and channel doubling
        - Final layers: LeakyReLU -> Conv2d -> LeakyReLU -> AdaptiveAvgPool2d -> Conv2d
        - All convolutions use spectral normalization for training stability

    Methods:
        get_feature(x): Returns final output and all intermediate features from each layer.
        forward(x): Returns squeezed output and intermediate features for adversarial loss computation.

    Returns:
        tuple: (discrimination_scores, intermediate_features)
            - discrimination_scores: Tensor of shape (batch,) for single domain or (batch, num_domains)
            - intermediate_features: List of tensors from each network layer
    """

    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x)
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features


class ResBlk1d(nn.Module):
    """
    A 1D residual block module for neural networks.

    This module implements a residual block with 1D convolutions, featuring optional
    normalization, downsampling, and dropout for regularization. The block follows
    the residual learning framework where the output is the sum of a shortcut
    connection and a residual path, scaled for unit variance.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        actv (nn.Module, optional): Activation function. Defaults to nn.LeakyReLU(0.2).
        normalize (bool, optional): Whether to apply instance normalization. Defaults to False.
        downsample (str, optional): Downsampling method. Can be "none" or other values
            for downsampling. Defaults to "none".
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.2.

    Attributes:
        actv (nn.Module): Activation function used in the block.
        normalize (bool): Flag indicating whether normalization is applied.
        downsample_type (str): Type of downsampling applied.
        learned_sc (bool): Whether a learned shortcut connection is used (when input
            and output dimensions differ).
        dropout_p (float): Dropout probability.
        pool (nn.Module): Pooling layer for downsampling or identity.
        conv1 (nn.Conv1d): First 1D convolution layer with weight normalization.
        conv2 (nn.Conv1d): Second 1D convolution layer with weight normalization.
        norm1 (nn.InstanceNorm1d, optional): First instance normalization layer.
        norm2 (nn.InstanceNorm1d, optional): Second instance normalization layer.
        conv1x1 (nn.Conv1d, optional): 1x1 convolution for learned shortcut connection.

    Returns:
        torch.Tensor: Output tensor with the same spatial dimensions as input
            (or downsampled if downsample != "none"), with dim_out channels,
            scaled by 1/sqrt(2) for unit variance.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
        dropout_p=0.2,
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p

        if self.downsample_type == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1)
            )

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == "none":
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)

        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class LayerNorm(nn.Module):
    """
    Custom LayerNorm module that applies layer normalization along the channel dimension.

    This implementation transposes the input tensor to apply standard layer normalization
    and then transposes back to maintain the original tensor shape. It's designed to work
    with tensors where the channel dimension is not the last dimension.

    Args:
        channels (int): Number of channels (features) to normalize over
        eps (float, optional): Small value added to denominator for numerical stability.
                              Defaults to 1e-5.

    Attributes:
        channels (int): Number of channels
        eps (float): Epsilon value for numerical stability
        gamma (nn.Parameter): Learnable scale parameter initialized to ones
        beta (nn.Parameter): Learnable shift parameter initialized to zeros

    Forward Args:
        x (torch.Tensor): Input tensor to normalize

    Returns:
        torch.Tensor: Layer normalized tensor with same shape as input
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    """
    Text encoder module for sequence-to-sequence models using CNN and xLSTM layers.

    This module processes text token sequences through embedding, convolutional layers, and
    transformer-like xLSTM blocks to generate encoded representations suitable for downstream
    tasks like text-to-speech synthesis.

    Args:
        channels (int): Number of channels/dimensions for embeddings and hidden representations
        kernel_size (int): Kernel size for convolutional layers
        depth (int): Number of convolutional layers to stack
        n_symbols (int): Size of the vocabulary/symbol set for embedding layer
        actv (nn.Module, optional): Activation function. Defaults to nn.LeakyReLU(0.2)

    Attributes:
        embedding (nn.Embedding): Token embedding layer mapping symbols to dense vectors
        prepare_projection (LinearNorm): Linear layer projecting to reduced dimensionality for xLSTM
        post_projection (LinearNorm): Linear layer projecting back to original dimensionality
        cfg (xLSTMBlockStackConfig): Configuration for xLSTM block stack
        cnn (nn.ModuleList): Stack of convolutional layers with normalization and dropout
        lstm (xLSTMBlockStack): xLSTM transformer-like blocks for sequence modeling

    Methods:
        forward(x, input_lengths, m): Full forward pass with masking for training
        inference(x): Simplified forward pass for inference without masking
        length_to_mask(lengths): Utility method to create boolean masks from sequence lengths

    Note:
        The model uses xLSTM (extended LSTM) architecture which combines advantages of
        LSTMs and Transformers. Masking is applied throughout to handle variable-length
        sequences properly.
    """

    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)  # [n_symbols, channels]

        self.prepare_projection = LinearNorm(channels, channels // 2)
        self.post_projection = LinearNorm(channels // 2, channels)
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            # slstm_block=sLSTMBlockConfig(
            #     slstm=sLSTMLayerConfig(
            #         backend="cuda",
            #         num_heads=4,
            #         conv1d_kernel_size=4,
            #         bias_init="powerlaw_blockdependent",
            #     ),
            #     feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            # ),
            context_length=channels,
            num_blocks=8,
            embedding_dim=channels // 2,
            # slstm_at=[1],
        )

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
                for _ in range(depth)
            ]
        )

        self.lstm = xLSTMBlockStack(self.cfg)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()

        x = self.prepare_projection(x)
        x = self.lstm(x)
        x = self.post_projection(x)

        x = x.transpose(-1, -2)

        x.masked_fill_(m, 0.0)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.lstm(x)
        return x

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    """
    Adaptive Instance Normalization 1D module.

    This module applies adaptive instance normalization to 1D input tensors using style vectors.
    It first normalizes the input using instance normalization without learnable parameters,
    then applies style-dependent affine transformation with learned gamma and beta parameters.

    Args:
        style_dim (int): Dimensionality of the input style vector.
        num_features (int): Number of features/channels in the input tensor to be normalized.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
        s (torch.Tensor): Style vector of shape (batch_size, style_dim).

    Returns:
        torch.Tensor: Style-modulated normalized tensor with the same shape as input x.
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    """
    A 1D upsampling module that can either pass input unchanged or upsample by factor of 2.

    This module provides conditional upsampling functionality based on the specified layer type.
    When layer_type is "none", the input is returned unchanged. For any other layer_type value,
    the input is upsampled by a factor of 2 using nearest neighbor interpolation.

    Args:
        layer_type (str): Type of upsampling to perform. If "none", no upsampling is applied.
                         Any other value will trigger 2x upsampling.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, length) to be upsampled.

    Returns:
        torch.Tensor: Output tensor. Same shape as input if layer_type is "none", otherwise
                      upsampled by factor of 2 in the last dimension.
    """

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
    """
    Adaptive Instance Normalization Residual Block for 1D convolutions.

    This module implements a residual block with Adaptive Instance Normalization (AdaIN)
    for style transfer in 1D signals. It combines residual connections with style-based
    normalization to enable style conditioning in neural networks.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        style_dim (int, optional): Dimensionality of the style vector. Defaults to 64.
        actv (nn.Module, optional): Activation function. Defaults to nn.LeakyReLU(0.2).
        upsample (str, optional): Upsampling type. Can be "none" or other upsampling modes.
            Defaults to "none".
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.

    Attributes:
        actv (nn.Module): Activation function used in the block.
        upsample_type (str): Type of upsampling applied.
        upsample (UpSample1d): Upsampling layer.
        learned_sc (bool): Whether to use learned shortcut connection when input and output
            dimensions differ.
        dropout (nn.Dropout): Dropout layer for regularization.
        pool (nn.Module): Pooling/transpose convolution layer for upsampling.
        conv1 (nn.Conv1d): First convolution layer.
        conv2 (nn.Conv1d): Second convolution layer.
        norm1 (AdaIN1d): First adaptive instance normalization layer.
        norm2 (AdaIN1d): Second adaptive instance normalization layer.
        conv1x1 (nn.Conv1d, optional): 1x1 convolution for shortcut connection when
            input and output dimensions differ.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_in, sequence_length).
        s (torch.Tensor): Style vector of shape (batch_size, style_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim_out, sequence_length).
            The output sequence length may change depending on the upsampling configuration.
    """

    def __init__(
        self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample="none", dropout_p=0.0
    ):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module that applies style-conditional normalization.

    This module performs layer normalization with learnable affine parameters (gamma and beta)
    that are predicted from a style vector, enabling style-dependent feature normalization.

    Args:
        style_dim (int): Dimensionality of the input style vector.
        channels (int): Number of channels in the input feature tensor.
        eps (float, optional): Small value added to denominator for numerical stability.
            Defaults to 1e-5.

    Forward Args:
        x (torch.Tensor): Input feature tensor of shape (batch_size, channels, seq_len).
        s (torch.Tensor): Style vector of shape (batch_size, style_dim).

    Returns:
        torch.Tensor: Style-conditioned normalized tensor with same shape as input x.

    Note:
        The input tensor undergoes multiple transpose operations to ensure proper dimension
        alignment for layer normalization and style conditioning operations.
    """

    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    """A neural network module for predicting prosodic features including duration, F0, and energy.

    This module uses xLSTM (extended Long Short-Term Memory) blocks to process text and style
    inputs for prosody prediction in text-to-speech synthesis. It can operate in two modes:
    duration prediction mode and F0/energy prediction mode.

    The architecture consists of:
    - Text encoder for processing input text with style conditioning
    - xLSTM blocks for sequence modeling
    - Separate prediction heads for duration, F0, and normalized energy
    - AdaIN residual blocks for style-conditioned feature processing

    Attributes:
        cfg (xLSTMBlockStackConfig): Configuration for the main xLSTM stack
        cfg_pred (xLSTMBlockStackConfig): Configuration for the prediction xLSTM stack
        text_encoder (DurationEncoder): Encoder for text input with style conditioning
        lstm (xLSTMBlockStack): Main xLSTM processing stack
        prepare_projection (nn.Linear): Linear layer for feature projection
        duration_proj (LinearNorm): Projection layer for duration prediction
        shared (xLSTMBlockStack): Shared xLSTM stack for F0/energy prediction
        f0 (nn.ModuleList): Sequential AdaIN residual blocks for F0 prediction
        n (nn.ModuleList): Sequential AdaIN residual blocks for energy prediction
        f0_proj (nn.Conv1d): Final projection layer for F0 output
        n_proj (nn.Conv1d): Final projection layer for energy output

        style_dim (int): Dimension of the style embedding
        d_hid (int): Hidden dimension size for the model
        nlayers (int): Number of layers in the text encoder
        max_dur (int, optional): Maximum duration value for prediction. Defaults to 50.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()

        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            context_length=d_hid,
            num_blocks=8,
            embedding_dim=d_hid + style_dim,
        )

        self.cfg_pred = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            context_length=4096,
            num_blocks=8,
            embedding_dim=d_hid + style_dim,
        )

        # self.shared = Hopfield(input_size=d_hid + style_dim,
        #                             hidden_size=d_hid // 2,
        #                             num_heads=32,
        #                             # scaling=.75,
        #                             add_zero_association=True,
        #                             batch_first=True)

        # if you want to use hopfield, just comment out the block above, then hash the "self.shared below"

        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )

        self.lstm = xLSTMBlockStack(self.cfg)
        self.prepare_projection = nn.Linear(d_hid + style_dim, d_hid)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = xLSTMBlockStack(self.cfg_pred)

        self.f0 = nn.ModuleList()
        self.f0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.f0.append(
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout)
        )
        self.f0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.n = nn.ModuleList()
        self.n.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.n.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.n.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.f0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.n_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths=None, alignment=None, mask=None, compute_f0=False):
        """Forward pass of the model.
        This method performs one of two main operations based on the `compute_f0` flag:
        1.  If `compute_f0` is True: It predicts F0 (fundamental frequency) and
            normalized energy using `self.F0Ntrain(texts, style)` and returns these
            predictions.
        2.  If `compute_f0` is False: It predicts token durations and an aligned
            encoded representation. This involves:
            - Encoding input `texts` and `style` via `self.text_encoder`, potentially
              using `text_lengths` and `mask`.
            - Processing the encoded output (`d`) through `self.lstm` and
              `self.prepare_projection`. The mask is also utilized here.
            - Predicting durations using `self.duration_proj` on the processed tensor
              after transposing and permuting it. Dropout is applied during training.
            - Calculating an aligned encoded representation `en` by matrix multiplying
              the transposed encoded text (`d`) with the provided `alignment`.
        Args:
            texts (torch.Tensor): Input text sequences.
            style (torch.Tensor): Style embedding or information.
            text_lengths (torch.Tensor, optional): Lengths of the input text sequences.
                Used by `self.text_encoder`. Defaults to None.
            alignment (torch.Tensor, optional): Pre-computed alignment matrix, used for
                calculating `en` if `compute_f0` is False. Defaults to None.
            mask (torch.Tensor, optional): Mask for the input text sequences.
                Used by `self.text_encoder` and for subsequent processing steps if
                `compute_f0` is False. Defaults to None.
            compute_f0 (bool, optional): If True, computes and returns F0 and
                normalized energy. Otherwise, predicts duration and `en`.
                Defaults to False.
        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
            - If `compute_f0` is True: The output of `self.F0Ntrain(texts, style)`.
            - If `compute_f0` is False: A tuple `(duration, en)` where:
                - `duration` (torch.Tensor): Predicted durations for each token,
                  with the last dimension squeezed. Shape: (batch_size, text_seq_len).
                - `en` (torch.Tensor): Encoded representation weighted by `alignment`.
                  Shape depends on `d` and `alignment`.
        """
        if compute_f0:
            # Predict F0 and norm energy and return
            return self.F0Ntrain(texts, style)

        # Predict duration and alignment

        # Problem is here
        d = self.text_encoder(texts, style, text_lengths, mask)

        # batch_size = d.shape[0]
        # text_size = d.shape[1]

        # # predict duration
        # input_lengths = text_lengths.cpu().numpy()

        # x = nn.utils.rnn.pack_padded_sequence(
        #     d, input_lengths, batch_first=True, enforce_sorted=False)
        x = d  # this dude can handle variable seq len so no need for padding
        mask = mask.to(text_lengths.device).unsqueeze(1)

        x = self.lstm(x)  # no longer using lstm
        x = self.prepare_projection(x)

        # x, _ = nn.utils.rnn.pad_packed_sequence(
        #     x, batch_first=True)

        # x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        # x_pad[:, :x.shape[1], :] = x
        # x = x_pad.to(x.device)

        x = x.transpose(-1, -2)
        x = x.permute(0, 2, 1)
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))

        en = d.transpose(-1, -2) @ alignment

        return duration.squeeze(-1), en

    # Train F0 and norm energy
    def F0Ntrain(self, x, s):
        x = self.shared(x.transpose(-1, -2))
        x = self.prepare_projection(x)

        f0 = x.transpose(-1, -2)
        for block in self.f0:
            f0 = block(f0, s)
        f0 = self.f0_proj(f0)

        n = x.transpose(-1, -2)
        for block in self.n:
            n = block(n, s)
        n = self.n_proj(n)

        return f0.squeeze(1), n.squeeze(1)

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


class DurationEncoder(nn.Module):
    """
    A neural network module for encoding duration information in text-to-speech synthesis.

    This encoder uses bidirectional LSTM layers with adaptive layer normalization to process
    text features along with style embeddings to predict duration patterns for speech synthesis.

    Args:
        sty_dim (int): Dimension of the style embedding vector.
        d_model (int): Hidden dimension of the model.
        nlayers (int): Number of LSTM layers to stack.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Attributes:
        lstms (nn.ModuleList): List of LSTM layers and AdaLayerNorm modules.
        dropout (float): Dropout probability used during training.
        d_model (int): Hidden dimension of the model.
        sty_dim (int): Dimension of the style embedding vector.

    Methods:
        forward(x, style, text_lengths, m):
            Forward pass for training with text lengths and masking.

        inference(x, style):
            Inference pass for generating duration predictions.

        length_to_mask(lengths):
            Utility method to create attention masks from sequence lengths.
    """

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)

        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)

        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)

                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad.to(x.device)

        return x.transpose(-1, -2)

    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask


def load_F0_models(path):
    """
    Load a JDCNet fundamental frequency (F0) model from a specified path.
    This function loads the pretrained F0 model from a checkpoint file, initializes
    the model with appropriate parameters, and loads the state dictionary from the
    checkpoint. The model is set to training mode after loading.
    Parameters
    ----------
    path : str
        The file path to the saved F0 model checkpoint.
    Returns
    -------
    f0_model : JDCNet
        The loaded fundamental frequency model instance ready for use.
    Notes
    -----
    The model expects the checkpoint to have a 'net' key containing the state dictionary.
    """
    logger.info("Loading F0 model from %s", path)
    # f0_model = JDCNet(num_class=1, seq_len=192)
    f0_model = JDCNet(num_class=1)  # `seq_len` is not used in the model definition
    params = torch.load(path, map_location="cpu")["net"]
    f0_model.load_state_dict(params)
    _ = f0_model.train()

    return f0_model


def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    """
    Load an Automatic Speech Recognition (ASR) model using the specified model path and configuration.
    Parameters
    ----------
    ASR_MODEL_PATH : str
        Path to the saved ASR model weights.
    ASR_MODEL_CONFIG : str
        Path to the YAML configuration file for the ASR model.
    Returns
    -------
    asr_model : ASRCNN
        The loaded ASR model instance set to training mode.
    Notes
    -----
    The function performs the following steps:
    1. Loads the model configuration from the specified YAML file
    2. Initializes an ASRCNN model with the loaded configuration
    3. Loads the model weights from the specified path
    4. Sets the model to training mode before returning
    The ASRCNN class should be imported before calling this function.
    """

    def _load_config(path):
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model_config = config["model_params"]
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(params)
        return model

    logger.info("Loading ASR model from %s", ASR_MODEL_PATH)
    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=None):
    """
    Load model and optimizer states from a checkpoint file.
    This function handles loading model parameters with special handling for
    DataParallel modules that might have key name inconsistencies between
    training stages.
    Parameters
    ----------
    model : dict
        Dictionary of model components to load
    optimizer : torch.optim.Optimizer
        Optimizer to load state
    path : str
        Path to the checkpoint file
    load_only_params : bool, default=True
        If True, only loads model parameters without optimizer state,
        and resets epoch/iters to 0. If False, loads optimizer state
        and continues from saved epoch/iters.
    ignore_modules : list, optional
        List of module names to ignore during loading
    Returns
    -------
    tuple
        (model, optimizer, epoch, iters) - The loaded model, optimizer,
        current epoch, and iteration count
    Notes
    -----
    This function includes handling for inconsistent key names between first
    and second training stages as noted in StyleTTS2 GitHub issues.
    """
    # Modified to deal with inconsistent key names between first and second training stages
    # => see https://github.com/yl4579/StyleTTS2/issues/254,
    # https://github.com/yl4579/StyleTTS2/issues/21#issue-1962579727
    # https://github.com/pytorch/pytorch/issues/9176#issuecomment-403570715

    if ignore_modules is None:
        ignore_modules = []
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    for key in model:
        if key in params and key not in ignore_modules:
            logger.info("%s loaded", key)
            try:
                model[key].load_state_dict(params[key], strict=True)
            except RuntimeError:  # DataParallel module. mismatch
                state_dict = params[key]
                new_state_dict = OrderedDict()
                # print(f'{key} key length: {len(model[key].state_dict().keys())}, state_dict length: {len(state_dict.keys())}')
                # print("model", len(model[key].state_dict().items()))
                # print("state", len(state_dict.items()))
                for k_m, _ in model[key].state_dict().items():
                    k_fix, v_c = None, None
                    if k_m in state_dict:
                        v_c = state_dict[k_m]
                        k_fix = k_m[7:]
                    if k_fix:
                        new_state_dict[k_fix] = v_c
                        # print(f'=> {k_m} => {k_fix}')
                model[key].load_state_dict(new_state_dict, strict=False)
    # Set to eval mode
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        # advance start epoch or we'd re-train and rewrite the last epoch file
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters


class StyleTTS2:
    """
    StyleTTS2 model class that encapsulates all components of the StyleTTS2 text-to-speech system.

    This class provides a unified interface to access various components of the StyleTTS2 model,
    including text encoding, style encoding, prosody prediction, diffusion model, and waveform
    generation. It allows for easy access to model components using both dot notation and
    dictionary-like access.

    Attributes
    ----------
    model : Munch
        A Munch object containing all model components.
    """

    def __init__(self, args, text_aligner, pitch_extractor, bert):
        """
        Initializes the StyleTTS2 model with the provided arguments and components.
        This constructor sets up the model parameters and builds the model components.

        Args:
            args (object): Configuration object containing model hyperparameters.
            text_aligner (nn.Module): Module that aligns text with audio features.
            pitch_extractor (nn.Module): Module that extracts pitch information from audio.
            bert (nn.Module): Pre-trained BERT model for extracting contextual text embeddings.
        """
        self._model = Munch()
        self._params = args
        self._build(args, text_aligner, pitch_extractor, bert)

    @property
    def model(self):
        return self._model

    @property
    def multispeaker(self):
        return self._params.multispeaker

    @property
    def slm(self):
        return self._params.slm

    @model.setter
    def model(self, value):
        try:
            self._model = munchify(value)
        except (TypeError, AttributeError) as exc:
            raise TypeError("Model must be a dictionary or Munch-compatible object") from exc

    def __getattr__(self, item):
        """
        Enable dot notation access to model components.
        This method is called when an attribute is not found in the instance.
        It delegates to the Munch object to provide dot notation access.
        """
        try:
            return self._model[item]
        except (AttributeError, KeyError) as exc:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'"
            ) from exc

    def __getitem__(self, key):
        """
        Enable dictionary-like access to model components using [] operator.
        Args:
            key: Key to access in the model dictionary
        Returns:
            The model component associated with the key
        """
        return self._model[key]

    def __setitem__(self, key, value):
        """
        Enable dictionary-like assignment to model components using [] operator.
        Args:
            key: Key to set in the model dictionary
            value: Value to assign to the key
        """
        self._model[key] = value

    def __delitem__(self, key):
        """
        Enable dictionary-like deletion of model components using [] operator.
        Args:
            key: Key to delete from the model dictionary
        """
        del self._model[key]

    def __iter__(self):
        """
        Enable iteration over model components.
        This allows for 'for k in model:' syntax to iterate over model component names.
        Returns:
            Iterator over the keys in self._model
        """
        return iter(self._model)

    def __contains__(self, key):
        """
        Enable membership testing using the 'in' operator.
        This allows for 'key in model' syntax to check if a component exists.
        Args:
            key: Key to check for existence in the model dictionary
        Returns:
            bool: True if the key exists in self._model, False otherwise
        """
        return key in self._model

    def _build(self, args, text_aligner, pitch_extractor, bert):
        """
        Builds the StyleTTS2 model components.
        This function constructs and configures all neural network components required for the
        StyleTTS2 TTS system, including text encoding, style encoding, prosody prediction,
        diffusion model, and waveform generation.
        Parameters
        ----------
        args : object
            Configuration object containing model hyperparameters:
            - hidden_dim: Dimension of hidden layers
            - style_dim: Dimension of style vectors
            - n_mels: Number of mel spectrogram bins
            - n_layer: Number of layers in various components
            - max_dur: Maximum duration for prosody prediction
            - dropout: Dropout rate for predictor
            - dim_in: Input dimension for style encoders
            - n_token: Number of tokens in the vocabulary
            - multispeaker: Boolean flag for multispeaker model configuration
            - diffusion: Configuration for diffusion model parameters
            - slm: Configuration for SLM discriminator parameters
            - decoder: Configuration for the decoder:
            - type: Either "istftnet" or "hifigan"
            - resblock_kernel_sizes: Kernel sizes for residual blocks
            - upsample_rates: Rates for upsampling
            - upsample_initial_channel: Initial channel count for upsampling
            - resblock_dilation_sizes: Dilation sizes for residual blocks
            - upsample_kernel_sizes: Kernel sizes for upsampling
            - gen_istft_n_fft: FFT size for ISTFT (for istftnet only)
            - gen_istft_hop_size: Hop size for ISTFT (for istftnet only)
        text_aligner : nn.Module
            Module that aligns text with audio features
        pitch_extractor : nn.Module
            Module that extracts pitch information from audio
        bert : nn.Module
            Pre-trained BERT model for extracting contextual text embeddings
        Returns
        -------
        nets : Munch
            A Munch object containing all model components:
            - bert: BERT model for text embedding
            - bert_encoder: Linear projection of BERT embeddings
            - predictor: Prosody predictor module
            - decoder: Mel-spectrogram decoder (ISTFTNet or HifiGAN)
            - text_encoder: Text encoding module
            - predictor_encoder: Style encoder for prosody prediction
            - style_encoder: Style encoder for acoustic features
            - diffusion: Audio diffusion model for generating waveforms
            - text_aligner: Module for aligning text with audio
            - pitch_extractor: Module for extracting pitch information
            - mpd: Multi-Period Discriminator for adversarial training
            - msd: Multi-Resolution Spectrogram Discriminator
            - wd: Waveform Discriminator for SLM
        """
        assert args.decoder.type in ["istftnet", "hifigan"], "Decoder type unknown"

        if args.decoder.type == "istftnet":
            decoder = ISTFTDecoder(
                dim_in=args.hidden_dim,
                # TODO: 2x means both acoustic and prosodic styles are computed
                # (originally only acoustic)
                style_dim=args.style_dim,
                resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
                upsample_rates=args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
                gen_istft_n_fft=args.decoder.gen_istft_n_fft,
                gen_istft_hop_size=args.decoder.gen_istft_hop_size,
            )
        else:
            decoder = HifiDecoder(
                dim_in=args.hidden_dim,
                # TODO: 2x means both acoustic and prosodic styles are computed
                # (originally only acoustic)
                style_dim=args.style_dim,
                resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
                upsample_rates=args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
            )

        text_encoder = TextEncoder(
            channels=args.hidden_dim,
            kernel_size=5,
            depth=args.n_layer,
            n_symbols=args.n_token,
        )

        prosodic_predictor = ProsodyPredictor(
            style_dim=args.style_dim,
            d_hid=args.hidden_dim,
            nlayers=args.n_layer,
            max_dur=args.max_dur,
            dropout=args.dropout,
        )

        # Acoustic style encoder
        acoustic_style_encoder = AcousticStyleEncoder(
            dim_in=args.dim_in,
            spk_emb_dim=args.spk_emb_dim,
            style_dim=args.style_dim,
            max_conv_dim=args.max_conv_dim,
            # Initial external/internal speaker embedding fusion weight
            mix_weight=args.mix_weight,
            learnable_gate=args.learnable_gate,
            mode=args.mode,
        )

        # Prosodic style encoder
        prosodic_style_encoder = StyleEncoder(
            dim_in=args.dim_in,
            style_dim=args.style_dim,
            max_conv_dim=args.max_conv_dim,
        )

        # define diffusion model
        if args.multispeaker:
            transformer = StyleTransformer1d(
                channels=args.style_dim * 2,
                context_embedding_features=bert.config.hidden_size,
                context_features=args.style_dim * 2,
                **args.diffusion.transformer,
            )
        else:
            transformer = Transformer1d(
                channels=args.style_dim * 2,
                context_embedding_features=bert.config.hidden_size,
                **args.diffusion.transformer,
            )

        diffusion = AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=bert.config.max_position_embeddings,
            embedding_features=bert.config.hidden_size,
            # Conditional dropout of batch elements
            embedding_mask_proba=args.diffusion.embedding_mask_proba,
            channels=args.style_dim * 2,
            context_features=args.style_dim * 2,
        )

        diffusion.diffusion = KDiffusion(
            net=diffusion.unet,
            sigma_distribution=LogNormalDistribution(
                mean=args.diffusion.dist.mean, std=args.diffusion.dist.std
            ),
            # a placeholder, will be changed dynamically when start training diffusion model
            sigma_data=args.diffusion.dist.sigma_data,
            dynamic_threshold=0.0,
        )
        diffusion.diffusion.net = transformer
        diffusion.unet = transformer

        self._model = munchify(
            {
                "bert": bert,
                "bert_encoder": nn.Linear(bert.config.hidden_size, args.hidden_dim),
                "prosodic_predictor": prosodic_predictor,
                "decoder": decoder,
                "text_encoder": text_encoder,
                "prosodic_style_encoder": prosodic_style_encoder,
                "acoustic_style_encoder": acoustic_style_encoder,
                "diffusion": diffusion,
                "text_aligner": text_aligner,
                "pitch_extractor": pitch_extractor,
                "mpd": MultiPeriodDiscriminator(),
                "msd": MultiResSpecDiscriminator(),
                # slm discriminator head
                "wd": WavDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
            }
        )

    def to(self, device="cpu"):
        """
        Move model parameters to the specified device.
        Args:
            device (str or torch.device): Device to move the model to (e.g., 'cuda' or 'cpu')
        Returns:
            StyleTT2: Self with model parameters moved to the specified device
        """
        device = torch.device(device)  # Convert once
        for key, module in self._model.items():  # .items() is faster
            if hasattr(module, "to"):
                self._model[key] = module.to(device, non_blocking=True)  # non_blocking for CUDA
        return self

    def set_mode(self, mode="train", components=None):
        """
        Set the model to training or evaluation mode.
        Args:
            mode (str): Mode to set the model to ('train' or 'eval')
            components (list, optional): List of component names (keys) to set mode for.
                                         If None, all components are set. Defaults to None.
        Returns:
            StyleTT2: Self with model components set to the specified mode
        """
        if mode not in ["train", "eval"]:
            raise ValueError("Mode must be either 'train' or 'eval'")

        # If no specific components specified, use all components
        if components is None:
            components = self._model.keys()

        method_name = mode
        for key in components:
            if key in self._model:
                module = self._model[key]
                if hasattr(module, method_name):
                    try:
                        getattr(module, method_name)()
                        logger.debug("Component '%s' set to %s mode", key, mode)
                    except (RuntimeError, TypeError, AttributeError) as e:
                        logger.warning("Failed to set %s to %s mode: %s", key, mode, e)
                else:
                    logger.debug("Module '%s' does not have a .%s() method", key, method_name)
            else:
                logger.warning("Component '%s' not found in model", key)

        return self

    def clone(self, freeze=False, eval_mode=False):
        """
        Clone the model with optional freezing parameters, and evaluation mode.
        Args:
            freeze (bool, optional): If True, sets requires_grad=False on all parameters.
                Defaults to False.
            eval_mode (bool, optional): If True, sets the model to evaluation mode.
                Defaults to False.
        Returns:
            StyleTT2: A new instance of StyleTT2 with cloned model components.
        """
        model_clone = copy.deepcopy(self._model)
        if freeze:
            for param in model_clone.parameters():
                param.requires_grad = False
        if eval_mode:
            model_clone.eval()
        return model_clone

    def is_warmup(self, epoch):
        return self._params.mode == "mix" and epoch < self._params.warmup_epochs

    def save(
        self,
        optimizer,
        epoch,
        iters,
        loss,
        basename,
        save_dir,
        max_saved_models=None,
        use_epoch_in_name=True,
    ):
        """
        Save a model checkpoint to disk with automatic cleanup of old checkpoints.

        Args:
            optimizer: The optimizer object whose state will be saved
            epoch (int): Current training epoch number
            iters (int): Current iteration number
            loss (float): Validation loss value to save
            basename (str): Base name for the checkpoint file
            save_dir (str): Directory where the checkpoint will be saved
            max_saved_models (int, optional): Maximum number of checkpoints to keep.
                If specified, older checkpoints will be automatically deleted.
            use_epoch_in_name (bool, optional): Whether to include epoch number in filename.
                Defaults to True.

        Returns:
            str: Full filepath of the saved checkpoint

        Note:
            - Creates save_dir if it doesn't exist
            - Skips saving if checkpoint file already exists
            - Automatically removes oldest checkpoints when max_saved_models limit is exceeded
            - Checkpoint contains model state_dict, optimizer state, iteration count, loss,
              and epoch
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Prepare model state for saving
        state_dict = {
            "net": {key: self._model[key].state_dict() for key in self._model},
            "optimizer": optimizer.state_dict(),
            "iters": iters,
            "val_loss": loss,
            "epoch": epoch,
        }

        # Save the model
        filename = f"{basename}_{epoch:05d}.pth" if use_epoch_in_name else f"{basename}.pth"
        filepath = os.path.join(save_dir, filename)
        if os.path.isfile(filepath):
            # Skip saving model when already exists
            logger.warning("Model %s already exists => skipping", filepath)
            return filepath
        torch.save(state_dict, filepath)
        logger.info("New model saved to %s", filepath)

        if max_saved_models:
            # Get list of all saved models and sort by epoch number
            saved_models = sorted(
                [
                    f
                    for f in os.listdir(save_dir)
                    if f.startswith(f"{basename}") and f.endswith(".pth")
                ],
                key=lambda x: int(x.split("_")[2].split(".")[0]),
            )

            # Remove old models if exceeding max_saved_models
            while len(saved_models) > max_saved_models:
                old_model = saved_models.pop(0)
                os.remove(os.path.join(save_dir, old_model))
                logger.info("Old model %s removed", old_model)

        # Return saved model's filepath
        return filepath

    def load(self, path, optimizer, load_only_params=True, ignore_modules=None):
        """
        Load model state from a checkpoint file.
        This method handles inconsistent key names between first and second training stages
        by attempting strict loading first, then falling back to a key-fixing approach
        for DataParallel module mismatches.
        Args:
            path (str): Path to the checkpoint file to load from.
            optimizer: The optimizer object to load state into (if load_only_params=False).
            load_only_params (bool, optional): If True, only load model parameters and ignore
                optimizer state and training metadata. Defaults to True.
            ignore_modules (list, optional): List of module keys to skip during loading.
                Defaults to None.
        Returns:
            tuple: A tuple containing (optimizer, epoch, iters) where:
                - optimizer: The optimizer object (potentially with loaded state)
                - epoch (int): Starting epoch number (0 if load_only_params=True, otherwise state["epoch"] + 1)
                - iters (int): Starting iteration count (0 if load_only_params=True, otherwise state["iters"])
        Note:
            - Sets the model to evaluation mode after loading
            - Handles DataParallel module key mismatches by removing the "module." prefix
            - Logs successful loading of each module
            - Uses non-strict loading as fallback for key mismatches
        """
        # Modified to deal with inconsistent key names between first and second training stages
        # => see https://github.com/yl4579/StyleTTS2/issues/254,
        # https://github.com/yl4579/StyleTTS2/issues/21#issue-1962579727
        # https://github.com/pytorch/pytorch/issues/9176#issuecomment-403570715

        if ignore_modules is None:
            ignore_modules = []
        state = torch.load(path, map_location="cpu")
        params = state["net"]
        for key in self._model:
            if key in params and key not in ignore_modules:
                logger.info("%s loaded", key)
                try:
                    self._model[key].load_state_dict(params[key], strict=True)
                except RuntimeError:  # DataParallel module. mismatch
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    # print(f'{key} key length: {len(model[key].state_dict().keys())}, state_dict length: {len(state_dict.keys())}')
                    # print("model", len(model[key].state_dict().items()))
                    # print("state", len(state_dict.items()))
                    for k_m, _ in self._model[key].state_dict().items():
                        k_fix, v_c = None, None
                        if k_m in state_dict:
                            v_c = state_dict[k_m]
                            k_fix = k_m[7:]
                        if k_fix:
                            new_state_dict[k_fix] = v_c
                            # print(f'=> {k_m} => {k_fix}')
                    self._model[key].load_state_dict(new_state_dict, strict=False)
        # Set to eval mode
        self.set_mode("eval")

        if not load_only_params:
            # advance start epoch or we'd re-train and rewrite the last epoch file
            epoch = state["epoch"] + 1
            iters = state["iters"]
            optimizer.load_state_dict(state["optimizer"])
        else:
            epoch = 0
            iters = 0

        return optimizer, epoch, iters
