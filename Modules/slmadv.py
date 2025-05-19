import numpy as np
import torch
import torch.nn.functional as F

from logger import get_logger

# Setup logger
logger = get_logger(__name__)


class SLMAdversarialLoss(torch.nn.Module):
    """SLMAdversarialLoss implements adversarial training for Style Language Models.

    This class implements a custom loss module that combines generator and discriminator losses
    for adversarial training of speech synthesis models using Style Language Models.

    Args:
        model: The main model that contains bert, predictor and decoder components
        wl: WaveformLoss instance for computing waveform-based losses
        sampler: The sampler used for generating style embeddings
        min_len: Minimum length of mel-spectrogram segments
        max_len: Maximum length of mel-spectrogram segments
        batch_percentage (float, optional): Maximum percentage of batch to process. Defaults to 0.5.
        skip_update (int, optional): Number of iterations to skip discriminator updates. Defaults to 10.
        sig (float, optional): Sigma parameter for Gaussian smoothing. Defaults to 1.5.
        hop_len (int, optional): Hop length for mel-spectrogram extraction. Defaults to 300.

    Methods:
        forward(iters, y_rec_gt, y_rec_gt_pred, waves, mel_input_length, ref_text,
               ref_lengths, use_ind, s_trg, ref_s=None):
            Computes the adversarial losses for a training step.
    """

    def __init__(
        self,
        model,
        wl,
        sampler,
        min_len,
        max_len,
        batch_percentage=0.5,
        skip_update=10,
        sig=1.5,
        hop_len=300,
    ):
        super().__init__()
        self.model = model
        self.wl = wl
        self.sampler = sampler

        self.min_len = min_len
        self.max_len = max_len
        self.batch_percentage = batch_percentage

        self.sig = sig
        self.skip_update = skip_update

        self.hop_len = hop_len

    def forward(
        self,
        iters,
        y_rec_gt,
        y_rec_gt_pred,
        waves,
        mel_input_length,
        ref_text,
        ref_lengths,
        use_ind,
        s_trg,
        ref_s=None,
    ):
        """Forward pass of the SLMAdv model.

        This method performs the forward pass through the SLMAdv (Style-based Latent Model Adversarial) training pipeline.
        It handles generation of style predictions, duration modeling, and adversarial training of generator/discriminator.

        Args:
            iters (int): Current training iteration number
            y_rec_gt (torch.Tensor): Ground truth reconstructed audio
            y_rec_gt_pred (torch.Tensor): Predicted reconstructed audio
            waves (torch.Tensor): Input waveform audio samples
            mel_input_length (torch.Tensor): Length of input mel-spectrograms
            ref_text (torch.Tensor): Reference text embeddings
            ref_lengths (torch.Tensor): Lengths of reference sequences
            use_ind (bool): Whether to use in-domain texts
            s_trg (torch.Tensor): Target style embeddings
            ref_s (torch.Tensor, optional): Reference style embeddings. Defaults to None.

        Returns:
            tuple or None: Returns either:
                - None if batch size is too small
                - Tuple of (discriminator_loss, generator_loss, predicted_audio)
                  where:
                  - discriminator_loss (torch.Tensor): Loss for discriminator
                  - generator_loss (torch.Tensor): Loss for generator
                  - predicted_audio (numpy.ndarray): Generated audio waveform
        """
        # text_mask = length_to_mask(ref_lengths).to(ref_text.device)
        # bert_dur = self.model.bert(ref_text, attention_mask=(~text_mask).int())
        # d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        text_mask = length_to_mask(ref_lengths).to(ref_text.device)
        d_en = self.model.text_encoder(ref_text, ref_lengths, text_mask)
        bert_dur = d_en.transpose(1, 2)

        if use_ind and np.random.rand() < 0.5:
            # Teacher forcing for the style component to
            # use target style to stabilize training
            s_preds = s_trg
        else:
            # Generate style predictions
            num_steps = np.random.randint(3, 5)
            if ref_s is not None:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(ref_text.device),
                    embedding=bert_dur,
                    embedding_scale=1,
                    features=ref_s,  # reference from the same speaker as the embedding
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)
            else:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(ref_text.device),
                    embedding=bert_dur,
                    embedding_scale=1,
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)

        s_dur = s_preds[:, 128:]
        # s = s_preds[:, :128]

        # Predict durations
        d, _ = self.model.predictor(
            d_en,
            s_dur,
            ref_lengths,
            torch.randn(ref_lengths.shape[0], ref_lengths.max(), 2).to(ref_text.device),
            text_mask,
        )

        bib = 0
        output_lengths = []
        attn_preds = []

        # Differentiable duration modeling
        for _s2s_pred, _text_length in zip(d, ref_lengths):
            _s2s_pred_org = _s2s_pred[:_text_length, :]
            _s2s_pred = torch.sigmoid(_s2s_pred_org)
            _dur_pred = _s2s_pred.sum(axis=-1)

            l = int(torch.round(_s2s_pred.sum()).item())
            t = torch.arange(0, l).expand(l)

            t = torch.arange(0, l).unsqueeze(0).expand((len(_s2s_pred), l)).to(ref_text.device)
            loc = torch.cumsum(_dur_pred, dim=0) - _dur_pred / 2

            h = torch.exp(-0.5 * torch.square(t - (l - loc.unsqueeze(-1))) / (self.sig) ** 2)

            out = torch.nn.functional.conv1d(
                _s2s_pred_org.unsqueeze(0),
                h.unsqueeze(1),
                padding=h.shape[-1] - 1,
                groups=int(_text_length),
            )[..., :l]
            attn_preds.append(F.softmax(out.squeeze(), dim=0))

            output_lengths.append(l)

        max_len = max(output_lengths)

        with torch.no_grad():
            t_en = self.model.text_encoder(ref_text, ref_lengths, text_mask)

        s2s_attn = torch.zeros(len(ref_lengths), int(ref_lengths.max()), max_len).to(
            ref_text.device
        )
        for bib, (r, o, a) in enumerate(zip(ref_lengths, output_lengths, attn_preds)):
            s2s_attn[bib, :r, :o] = a

        asr_pred = t_en @ s2s_attn

        # Predict aligned pitch features
        _, p_pred = self.model.predictor(d_en, s_dur, ref_lengths, s2s_attn, text_mask)

        mel_len = max(int(min(output_lengths) / 2 - 1), self.min_len // 2)
        mel_len = min(mel_len, self.max_len // 2)

        # --- Pre-allocate Segment Extraction ---
        # Compute the batch size based on the given percentage of the original batch size,
        # ensuring it is at least 1
        # original_batch_size = len(waves)
        bsize = max(1, int(self.batch_percentage * len(waves)))
        # Calculate fixed waveform segment length
        wav_len = (mel_len * 2) * self.hop_len

        # Pre-allocate tensors with the calculated fixed length
        en = torch.empty(
            bsize, asr_pred.shape[1], mel_len, device=asr_pred.device, dtype=asr_pred.dtype
        )
        p_en = torch.empty(
            bsize, p_pred.shape[1], mel_len, device=p_pred.device, dtype=p_pred.dtype
        )
        wav_gt = torch.empty(bsize, wav_len, device=p_pred.device, dtype=torch.float)
        # Predicted styles: 'voice' style (128) + prosodic style (128)
        sp = s_preds[:bsize, :]

        # Iterate through the batch samples
        for bidx in range(bsize):
            mel_len_pred = output_lengths[bidx]
            mel_len_gt = int(mel_input_length[bidx].item() / 2)
            # Skip too short mel-spectrogram segments
            if mel_len_gt <= mel_len or mel_len_pred <= mel_len:
                continue

            # Randomly select a start point for features within valid range
            beg = np.random.randint(0, mel_len_pred - mel_len)
            # Extract features
            en[bidx] = asr_pred[bidx, :, beg : beg + mel_len]
            p_en[bidx] = p_pred[bidx, :, beg : beg + mel_len]

            # Randomly select a start point for ground truth segments
            beg = np.random.randint(0, mel_len_gt - mel_len)
            beg_idx = (beg * 2) * self.hop_len
            end_idx = beg_idx + wav_len
            wav_gt[bidx] = waves[bidx][beg_idx:end_idx]
        # --- End of Pre-allocate Segment Extraction ---

        # # --- Original Segment Extraction ---
        # # Get clips
        # en, p_en, sp, wav = [], [], [], []
        # for bib, (o, m, s, w) in enumerate(zip(output_lengths, mel_input_length, s_preds, waves)):
        #     mel_length_pred = o
        #     mel_length_gt = int(m.item() / 2)
        #     # Skip too short mel-spectrogram segments
        #     if mel_length_gt <= mel_len or mel_length_pred <= mel_len:
        #         continue

        #     sp.append(s)

        #     random_start = np.random.randint(0, mel_length_pred - mel_len)
        #     en.append(asr_pred[bib, :, random_start : random_start + mel_len])
        #     p_en.append(p_pred[bib, :, random_start : random_start + mel_len])

        #     # Get ground truth clips
        #     random_start = np.random.randint(0, mel_length_gt - mel_len)
        #     y = w[(random_start * 2) * self.hop_len : ((random_start + mel_len) * 2) * self.hop_len]
        #     # wav.append(torch.from_numpy(y).to(ref_text.device))
        #     wav.append(y.to(ref_text.device))

        #     if len(wav) >= self.batch_percentage * len(waves):  # prevent OOM due to longer lengths
        #         logger.debug(
        #             "Prevent OOM due to longer lengths: %d >= %d",
        #             len(wav),
        #             self.batch_percentage * len(waves),
        #         )
        #         break

        # # if len(sp) <= 1: # Originally, batch size >=2 supported
        # if len(sp) < 1:  # JMa: Can we use only 1 sample for SLM adversarial loss training?
        #     logger.warning("No samples left after filtering, skipping SLMADV batch")
        #     return None

        # sp = torch.stack(sp)
        # wav = torch.stack(wav).float()
        # en = torch.stack(en)
        # p_en = torch.stack(p_en)
        # # --- End of Original Segment Extraction ---

        f0_fake, n_fake = self.model.predictor.F0Ntrain(p_en, sp[:, 128:])
        y_pred = self.model.decoder(en, f0_fake, n_fake, sp[:, :128])

        # discriminator loss
        if (iters + 1) % self.skip_update == 0:
            if np.random.randint(0, 2) == 0:
                wav = y_rec_gt_pred
                use_rec = True
            else:
                wav = wav_gt
                use_rec = False

            crop_size = min(wav.size(-1), y_pred.size(-1))
            if use_rec:  # use reconstructed (shorter lengths), do length invariant regularization
                if wav.size(-1) > y_pred.size(-1):
                    real_gp = wav[:, :, :crop_size]
                    out_crop = self.wl.discriminator_forward(real_gp.detach().squeeze(1))
                    out_org = self.wl.discriminator_forward(wav.detach().squeeze(1))
                    loss_reg = F.l1_loss(out_crop, out_org[..., : out_crop.size(-1)])

                    if np.random.randint(0, 2) == 0:
                        d_loss = self.wl.discriminator(
                            real_gp.detach().squeeze(1), y_pred.detach().squeeze(1)
                        ).mean()
                    else:
                        d_loss = self.wl.discriminator(
                            wav.detach().squeeze(1), y_pred.detach().squeeze(1)
                        ).mean()
                else:
                    real_gp = y_pred[:, :, :crop_size]
                    out_crop = self.wl.discriminator_forward(real_gp.detach().squeeze(1))
                    out_org = self.wl.discriminator_forward(y_pred.detach().squeeze(1))
                    loss_reg = F.l1_loss(out_crop, out_org[..., : out_crop.size(-1)])

                    if np.random.randint(0, 2) == 0:
                        d_loss = self.wl.discriminator(
                            wav.detach().squeeze(1), real_gp.detach().squeeze(1)
                        ).mean()
                    else:
                        d_loss = self.wl.discriminator(
                            wav.detach().squeeze(1), y_pred.detach().squeeze(1)
                        ).mean()

                # regularization (ignore length variation)
                d_loss += loss_reg

                out_gt = self.wl.discriminator_forward(y_rec_gt.detach().squeeze(1))
                out_rec = self.wl.discriminator_forward(y_rec_gt_pred.detach().squeeze(1))

                # regularization (ignore reconstruction artifacts)
                d_loss += F.l1_loss(out_gt, out_rec)

            else:
                d_loss = self.wl.discriminator(
                    wav.detach().squeeze(1), y_pred.detach().squeeze(1)
                ).mean()
        else:
            d_loss = 0

        # generator loss
        gen_loss = self.wl.generator(y_pred.squeeze(1))
        gen_loss = gen_loss.mean()

        return d_loss, gen_loss, y_pred.detach().cpu().numpy()


def length_to_mask(lengths):
    """
    Creates a boolean mask tensor based on sequence lengths.

    This function generates a mask where True values indicate positions beyond the sequence length
    for each item in the batch.

    Args:
        lengths (torch.Tensor): A 1D tensor containing sequence lengths for each item in the batch.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, max_length) where True values
                     indicate positions beyond each sequence's length.

    Example:
        >>> lengths = torch.tensor([2, 3, 1])
        >>> mask = length_to_mask(lengths)
        >>> print(mask)
        tensor([[False, False,  True],
                [False, False, False],
                [False,  True,  True]])
    """
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask
