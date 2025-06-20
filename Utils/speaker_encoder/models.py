import logging
import numpy as np
import torch
from torch import nn
import torchaudio
import soundfile as sf
import librosa as lr

logger = logging.getLogger(__name__)


def rms_volume_norm(wav: np.ndarray, db_level: float = -27.0, **kwargs) -> np.ndarray:
    """Normalize the volume based on RMS of the signal.

    Args:
        wav (np.ndarray): Raw waveform.
        db_level (float): Target dB level in RMS. Defaults to -27.0.

    Returns:
        np.ndarray: RMS normalized waveform.
    """
    assert -99 <= db_level <= 0, " [!] db_level should be between -99 and 0"
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
    return wav * a


class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer(
            "filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class HASPSpeakerEncoder(nn.Module):
    """Implementation of the model H/ASP without batch normalization in speaker embedding.
    This model was proposed in: https://arxiv.org/abs/2009.14153
    Adapted from: https://github.com/idiap/coqui-ai-TTS
    (originally from https://github.com/clovaai/voxceleb_trainer)
    """

    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        audio_config=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        self.torch_spec = self.get_torch_mel_spectrogram_class(audio_config)

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        out_dim = num_filters[3] * outmap_size * 2  # ASP encoder type

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    @torch.inference_mode()
    def inference(self, x):
        return self.forward(x)

    def get_torch_mel_spectrogram_class(self, audio_config):
        return torch.nn.Sequential(
            PreEmphasis(audio_config["preemphasis"]),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config["sample_rate"],
                n_fft=audio_config["fft_size"],
                win_length=audio_config["win_length"],
                hop_length=audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=audio_config["num_mels"],
            ),
        )

    @torch.inference_mode()
    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        # map to the waveform size
        num_frames = num_frames * self.audio_config["hop_length"]

        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings

    def load_checkpoint(self, checkpoint_path: str, evaluate: bool = False):
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        try:
            self.load_state_dict(state["model"])
            logger.info("Model fully restored. ")
        except (KeyError, RuntimeError) as error:
            raise error

        if evaluate:
            self.eval()
            assert not self.training

    def compute_embedding_from_clip(self, wav_file):
        """Compute a embedding from a given audio file.

        Args:
            wav_file (str): Target file path.

        Returns:
            list: Computed embedding.
        """
        m_input = self.load_waveform(wav_file)
        embedding = self.compute_embedding(m_input.unsqueeze(0))
        return embedding[0].tolist()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal.

        Shapes:
            - x: :math:`(N, 1, T_{in})`
        """
        x.squeeze_(1)

        x = self.torch_spec(x)  # convert wav to mels

        x = (x + 1e-6).log()  # log_input

        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        # ASP -- Attentive Statistics Pooling -- https://arxiv.org/abs/1803.10963
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        # apply l2 norm
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    @staticmethod
    def load_waveform(wav_file):
        """Load a waveform from a given audio file.

        Args:
            wav_file (str): Target file path.

        Returns:
            Tensor: Loaded waveform.
        """
        waveform, org_sr = sf.read(wav_file, dtype="float32")
        if org_sr != 16000:
            waveform = lr.resample(waveform, orig_sr=org_sr, target_sr=16000)
        waveform = rms_volume_norm(waveform)
        waveform = torch.from_numpy(waveform)
        return waveform
