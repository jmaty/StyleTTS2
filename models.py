# coding:utf-8

import copy
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from munch import Munch
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


class StyleEncoder(nn.Module):
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
        return s


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class Discriminator2d(nn.Module):
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
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
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
    f0_model = JDCNet(num_class=1)
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


def build_model(args, text_aligner, pitch_extractor, bert):
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
    acoustic_style_encoder = StyleEncoder(
        dim_in=args.dim_in,
        style_dim=args.style_dim,
        max_conv_dim=args.max_conv_dim,
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

    nets = Munch(
        bert=bert,
        bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),
        prosodic_predictor=prosodic_predictor,
        decoder=decoder,
        text_encoder=text_encoder,
        prosodic_style_encoder=prosodic_style_encoder,
        acoustic_style_encoder=acoustic_style_encoder,
        diffusion=diffusion,
        text_aligner=text_aligner,
        pitch_extractor=pitch_extractor,
        mpd=MultiPeriodDiscriminator(),
        msd=MultiResSpecDiscriminator(),
        # slm discriminator head
        wd=WavDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
    )

    return nets


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


# JMa: Save model and delete old models
def save_checkpoint(
    model,
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
    Save model checkpoint to disk.
    Args:
        model (dict): Dictionary of network models to save
        optimizer: Optimizer whose state will be saved
        epoch (int): Current epoch number
        iters (int): Current iteration count
        loss (float): Current validation loss
        basename (str): Base filename for the saved model
        save_dir (str): Directory to save the model in
        max_saved_models (int, optional): Maximum number of saved models to keep.
            If exceeded, oldest models will be deleted. If None, all models are kept.
        use_epoch_in_name (bool, optional): Whether to include epoch number in filename.
            Defaults to True.
    Returns:
        str: Path to the saved checkpoint file
    Notes:
        - Creates save_dir if it doesn't exist
        - Skips saving if the exact file already exists
        - If max_saved_models is specified, maintains only the N most recent checkpoints
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare model state for saving
    state_dict = {
        "net": {key: model[key].state_dict() for key in model},
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
            [f for f in os.listdir(save_dir) if f.startswith(f"{basename}") and f.endswith(".pth")],
            key=lambda x: int(x.split("_")[2].split(".")[0]),
        )

        # Remove old models if exceeding max_saved_models
        while len(saved_models) > max_saved_models:
            old_model = saved_models.pop(0)
            os.remove(os.path.join(save_dir, old_model))
            logger.info("Old model %s removed", old_model)

    # Return saved model's filepath
    return filepath


def model2device(model, device="cpu"):
    """
    Move model parameters to the specified device.
    Args:
        model (dict): Dictionary of model components
        device (torch.device): Device to move the model to (e.g., 'cuda' or 'cpu')
    Returns:
        dict: Model with parameters moved to the specified device
    """
    device = torch.device(device)  # Convert once
    for key, module in model.items():  # .items() is faster
        if hasattr(module, "to"):
            model[key] = module.to(device, non_blocking=True)  # non_blocking for CUDA
    return model


def model2mode(model, mode="train", components=None):
    """
    Set model to training or evaluation mode.
    Args:
        model (dict): Dictionary of model components
        mode (str): Mode to set the model to ('train' or 'eval')
        components (list, optional): List of component names (keys) to set mode for.
                                   If None, all components are set. Defaults to None.
    Returns:
        dict: Model with the specified mode set
    """
    if mode not in ["train", "eval"]:
        raise ValueError("Mode must be either 'train' or 'eval'")

    # If no specific components specified, use all components
    if components is None:
        components = model.keys()

    method_name = mode
    for key in components:
        if key in model:
            module = model[key]
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

    return model


def clone_model(model, device=None, freeze=False, eval_mode=False):
    """
    Returns a deep copy of a PyTorch model.
    Args:
        model:      model to be cloned
        device:     torch.device (optional), target device for clone
        freeze:     bool, if True sets requires_grad=False on all parameters
        eval_mode:  bool, if True puts model in eval() mode
    Returns:
        model_clone: New instance, weights copied.
    """
    model_clone = copy.deepcopy(model)
    if device is not None:
        model_clone = model_clone.to(device)
    else:
        model_clone = model_clone.to(model.device)
    if freeze:
        for param in model_clone.parameters():
            param.requires_grad = False
    if eval_mode:
        model_clone.eval()
    return model_clone
