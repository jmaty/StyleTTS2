import argparse
import os
import os.path as osp
import random
import time
import warnings

import numpy as np
import nvidia_smi
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from monotonic_align import mask_from_lens
from munch import munchify

from logger import add_logging_args, get_logger, setup_logging
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, create_slm_loss
from meldataset import build_dataloader
from models import StyleTTS2, load_ASR_models, load_F0_models
from Modules.pts import PTS
from optimizers import build_optimizer
from text_utils import TextCleaner
from utils import (
    get_data_path_list,
    length_to_mask,
    log_norm,
    maximum_path,
    nccl_warmup,
    set_random_seed,
    h100_fix,
)
from Utils.PLBERT.util import load_plbert

warnings.simplefilter("ignore")

h100_fix()  # Fix for H100 GPU


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="StyleTTS2 stage 1 training")
    parser.add_argument("config_path", type=str, help="path to config")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="number of workers")
    add_logging_args(parser)  # --log-level, --log-file
    args = parser.parse_args()

    # Load config
    with open(args.config_path, encoding="utf-8") as fr:
        cfg = munchify(yaml.safe_load(fr))

    wb_logger = None  # WandB logger

    # Set up logging
    set_random_seed(cfg.seed)
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # must be before Accelerator
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        project_dir=log_dir,
        split_batches=True,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # NCCL warm-up
    nccl_warmup(device=getattr(acc, "device", None), local_rank=local_rank)

    # Initialize W&B first (it may add logging handlers); we'll override logging next
    if acc.is_main_process:
        wb_logger = wandb.init(
            # Set the wandb project where this run will be logged.
            project="StyleTTS2_cs",
            # Set run name
            name=f"{osp.basename(log_dir)}",
            config=cfg,
            dir=log_dir,
        )

    # Uniform logging (main process only)
    log_file = args.log_file or osp.join(log_dir, "train.log")
    setup_logging(args.log_level, log_file, accelerator=acc)
    logger = get_logger(__name__)

    # Set up device
    device = acc.device

    # Init NVLM
    n_gpus, max_vram, total_vram = 0, 0, 0
    if acc.is_main_process:
        nvidia_smi.nvmlInit()
        n_gpus = nvidia_smi.nvmlDeviceGetCount()
        max_vram = 0  # Track maximum VRAM usage
        # Get total VRAM of the first GPU
        total_vram = (
            nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(0)).total >> 30
        )
        logger.info("NVLM initialized")

    # Set up epochs
    epochs = cfg.epochs.stage1
    tma_epoch = cfg.epochs.tma

    # Set up data parameters
    test_audio_dir = os.path.join(cfg.log_dir, cfg.data_params.test_audio_dir)

    # Set up text cleaner and pre-processing function
    text_cleaner = TextCleaner(cfg.data_params.symbol_dict_path, pad=cfg.data_params.pad)
    if acc.is_main_process:
        logger.debug("Number of symbols: %d", len(text_cleaner))
    assert len(text_cleaner) == 81, f"Number of symbols must be 81 but it is {len(text_cleaner)}"
    assert (
        cfg.model_params.n_token == 81
    ), f"Number of tokens must be 81 but it is {cfg.model_params.n_token}"

    # Load utility models
    with acc.main_process_first():
        # Load pretrained ASR model
        text_aligner = load_ASR_models(cfg.ASR_path, cfg.ASR_config)
        # Load pretrained F0 model
        pitch_extractor = load_F0_models(cfg.F0_path)
        # Load BERT model
        plbert = load_plbert(cfg.PLBERT_dir)

    # Initialize StyleTTS2 model
    model = StyleTTS2(cfg.model_params, text_aligner, pitch_extractor, plbert)
    logger.info("StyleTTS2 model built with %s", model.keys())

    # Extract BERT size from ALBERT config
    bert_size = model.bert.config.max_position_embeddings

    # Prepare model for distributed training
    for k in model:
        model[k] = acc.prepare(model[k])

    # Build combined optimizer and schedulers
    optimizer = build_optimizer(
        {k: list(model[k].parameters()) for k in model},  # modules to optimize
        cfg.optimizer_params,
    )

    # Prepare optimizers and schedulers for distributed training - safe variant
    optimizer.optimizers = {k: acc.prepare(v) for k, v in optimizer.optimizers.items()}
    optimizer.schedulers = {k: acc.prepare(v) for k, v in optimizer.schedulers.items()}

    # Load data
    train_list, val_list = get_data_path_list(cfg.data_params.train_data, cfg.data_params.val_data)

    # Set up dataset parameters (from config)
    dataset_config = {
        "sr": cfg.preprocess_params.sr,
        "min_length": cfg.data_params.min_length,
        # limit max length of the input sequence to the max length of the BERT model
        "max_length": bert_size,  # ALBERT config
        "silence_beg": cfg.preprocess_params.silence_beg,
        "silence_end": cfg.preprocess_params.silence_end,
        "n_mels": cfg.model_params.n_mels,
        "spect_params": cfg.preprocess_params.spect_params,
        "max_ref_mel_length": cfg.preprocess_params.max_ref_mel_length,
        "use_ref_sample": False,
    }

    # Prepare dataloaders
    logger.info("Building training dataloader...")
    train_dataloader = build_dataloader(
        train_list,
        cfg.data_params.root_path,
        text_cleaner,
        validation=False,
        ood_data=None,  # OOD data not used for 1st stage training
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        device=device,
        dataset_config=dataset_config,
        use_speaker_sampler=bool(model.multispeaker),
    )
    logger.info("Building validation dataloader...")
    val_dataloader = build_dataloader(
        val_list,
        cfg.data_params.root_path,
        text_cleaner,
        ood_data=None,  # OOD data not used for validation
        batch_size=cfg.batch_size,
        validation=True,
        num_workers=0,
        device=device,
        dataset_config=dataset_config,
        use_speaker_sampler=False,
    )
    if acc.is_main_process:  # Přidat tuto podmínku
        wb_logger.summary["n_train_samples"] = len(train_dataloader.dataset)
        wb_logger.summary["n_valid_samples"] = len(val_dataloader.dataset)
        wb_logger.summary["n_ood_texts"] = train_dataloader.dataset.number_ood_texts()

    # Prepare dataloaders for accelerated training
    train_dataloader, val_dataloader = acc.prepare(train_dataloader, val_dataloader)

    # Number of iteration-steps per epoch (per process)
    steps_per_epoch = len(train_dataloader)
    # Number of update-steps per epoch (accounts for grad accumulation)
    updates_per_epoch = int(np.ceil(steps_per_epoch / max(1, cfg.grad_accum_steps)))

    # Load model weights
    with acc.main_process_first():
        if cfg.get("pretrained_model", "") != "":
            optimizer, start_epoch, iters = model.load(
                cfg["pretrained_model"],
                optimizer,
                load_only_params=cfg.get("load_only_params", True),
            )
            # advance start epoch or we'd re-train and rewrite the last epoch file
            # start_epoch += 1
            logger.info("Loading pre-trained model: %s", cfg.pretrained_model)
            logger.info("Starting epoch:            %d", start_epoch)
            logger.info("Starting iterations:       %d", iters)
            logger.info("")
        else:
            start_epoch = 0
            iters = 0
        logger.info("")

    # In case not distributed computing
    try:
        n_down = model.text_aligner.module.n_down
    except AttributeError:
        logger.warning("Distributed computing NOT used")
        n_down = model.text_aligner.n_down

    # Wrapped losses for compatibility with mixed precision
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = create_slm_loss(model.slm, model.wd, cfg.preprocess_params.sr).to(device)

    # Create test audio dir under log/eval dir
    if (cfg.data_params.save_val_audio or cfg.data_params.save_test_audio) and not os.path.exists(
        test_audio_dir
    ):
        os.makedirs(test_audio_dir, exist_ok=True)

    # Create phoneme-to-speech object for synthesizing validation sentences
    # - use global noise for speed
    pts = PTS(cfg, model, use_glob_noise=True)

    best_loss = float("inf")  # best test loss

    if acc.is_main_process:
        logger.info(" > Start training cycles:")
        logger.info(" | > Random seed:         %s", cfg.seed)
        logger.info(" | > Experiment label:    %s", cfg.label)
        logger.info(" | > Starting epoch:      %d", start_epoch)
        logger.info(" | > Total epochs:        %d", epochs)
        logger.info(" | > Steps per epoch:     %d", steps_per_epoch)
        logger.info(" | > Updates per epoch:   %d", updates_per_epoch)
        logger.info(" | > Input iterations:    %d", iters)
        logger.info(" | > Train data:          %s", cfg.data_params.train_data)
        logger.info(" | > Valid data:          %s", cfg.data_params.val_data)
        logger.info(" | > Pretrained model:    %s", cfg.pretrained_model)
        logger.info(" | > Text aligner:        %s", cfg.ASR_path)
        logger.info(" | > F0 model:            %s", cfg.F0_path)
        logger.info(" | > PL-BERT:             %s", cfg.PLBERT_dir)
        logger.info(" | > Batch size:          %d", cfg.batch_size)
        logger.info(" | > Grad. accum. steps:  %d", cfg.grad_accum_steps)
        logger.info(" | > Effect. batch size:  %d", cfg.batch_size * cfg.grad_accum_steps)
        logger.info(" | > Max len:             %d", cfg.max_len)
        logger.info(" | > SLM loss:            %s", cfg.model_params.slm.model)
        logger.info(" | > Acoust style dim:    %d", cfg.model_params.style_dim)
        logger.info(" | > Pros. style dim:     %d", cfg.model_params.style_dim)
        logger.info("")

    # === Start of training loop ==============================================
    model.set_mode("eval")

    # Counter of optimizer.step() calls accross the training
    updates = 0

    # Iterate through the defined number of epochs
    for epoch in range(start_epoch, epochs):
        logger.debug("> ----- Epoch %d/%d -----", epoch + 1, epochs)
        running_loss = 0
        start_time = time.time()
        train_dataloader.batch_sampler.epoch = epoch  # Set epoch for the sampler
        updates_at_epoch_start = updates
        updates_at_last_log = updates  # Track updates for loss averaging

        # Models in train mode from the beginning
        train_components = [
            "decoder",
            "text_encoder",
            "acoustic_style_encoder",
        ]

        # Models in train mode based on the epoch
        if epoch >= tma_epoch:
            train_components.extend(["msd", "mpd", "text_aligner"])

        # Set models to train mode
        model.set_mode("train", train_components)

        # JMa: Zero gradients of all optimizers at each epoch start
        optimizer.zero_grad()

        # Train loop for each epoch
        for batch_idx, batch in enumerate(train_dataloader):
            waves = batch[0]  # Keep ground truth audio
            # Keep individual batch tensors
            (
                phonemes,  # Padded input phoneme IDs [B, T_text]
                ph_inp_lens,  # Input phoneme lengths [B]
                _,  # OOD texts not used in 1st stage training
                _,  # OOD phoneme lengths not used in 1st stage training
                mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                mel_inp_len,  # Mel spectrogram lengths [B]
                _,  # Reference mel spectrograms not used in 1st stage
            ) = batch[1:]

            # Generate masks for text and mel spectrograms
            with torch.no_grad():
                # `2**n_down` scaling ensures the mask aligns with the downsampled feature dimension
                mel_mask = length_to_mask(mel_inp_len // (2**n_down)).to(mel_inp_len.device)
                ph_mask = length_to_mask(ph_inp_lens).to(phonemes.device)

            # Align text and audio (mel)
            _, s2s_pred, d_algn = model.text_aligner(mels, mel_mask, phonemes)
            # Refine attention matrix
            d_algn = d_algn.transpose(-1, -2)
            d_algn = d_algn[..., 1:]
            d_algn = d_algn.transpose(-1, -2)

            # Create attention mask
            with torch.no_grad():
                attn_mask = (
                    (~mel_mask)
                    .unsqueeze(-1)
                    .expand(mel_mask.shape[0], mel_mask.shape[1], ph_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask.float()
                    * (~ph_mask)
                    .unsqueeze(-1)
                    .expand(ph_mask.shape[0], ph_mask.shape[1], mel_mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1  # Convert to boolean tensor

            # Apply attention mask to the attention matrix
            # d_algn.masked_fill_(attn_mask, 0.0)  # Apply attention mask to the attention matrix
            d_algn = d_algn.masked_fill(attn_mask, 0.0)

            with torch.no_grad():
                # Create monotonic attention
                mask_st = mask_from_lens(d_algn, ph_inp_lens, mel_inp_len // (2**n_down))
                d_algn_mono = maximum_path(d_algn, mask_st)

            # Encode
            h_ph = model.text_encoder(phonemes, ph_inp_lens, ph_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                h_algn = h_ph @ d_algn
            else:
                h_algn = h_ph @ d_algn_mono

            # Get clips
            # TODO: not to divide by 2?
            # TODO: get max (+ padding) instead of min?

            # --- Pre-allocate tensors ---

            mel_inp_len_all = acc.gather(mel_inp_len)  # for balanced load
            mel_len_gt = min([int(mel_inp_len_all.min().item() / 2 - 1), cfg.max_len // 2])
            # Early check for segment length:
            # - mel_len_gt * 2 is the length of the original mel spectrogram
            # - multiplication by 2 is due to the downsampling factor between mel and text aligner
            if mel_len_gt * 2 < 80:
                logger.warning(
                    "Segment is too short (%d frames, %d samples)=> skipping batch %d.",
                    mel_len_gt * 2,
                    (mel_len_gt * 2) * cfg.preprocess_params.spect_params.hop_length,
                    batch_idx,
                )
                continue
            mel_len_st = int(mel_inp_len.min().item() / 2 - 1)

            bsize = mel_inp_len.shape[0]  # Use current batch size
            # Calculate fixed waveform segment length
            wav_len = (mel_len_gt * 2) * cfg.preprocess_params.spect_params.hop_length

            # Pre-allocate tensors with the calculated fixed length
            ph_algn = torch.empty(
                bsize,
                h_algn.shape[1],
                mel_len_gt,
                device=device,
                dtype=h_algn.dtype,
            )
            mel_gt = torch.empty(
                bsize,
                mels.shape[1],
                mel_len_gt * 2,
                device=device,
                dtype=mels.dtype,
            )
            mel_st = torch.empty(
                bsize,
                mels.shape[1],
                mel_len_st * 2,
                device=device,
                dtype=mels.dtype,
            )
            wav_gt = torch.empty(bsize, wav_len, device=device, dtype=torch.float)

            # Iterate through the batch samples
            for bidx in range(bsize):
                # Mel-spectrogram length (dividing by 2 due to a downsampling factor?)
                mel_len = int(mel_inp_len[bidx].item() / 2)

                # --- Segment for en, mel_gt, wav_gt ---
                # Randomly select a start point for the mel spectrogram within valid range
                beg_gt = np.random.randint(0, mel_len - mel_len_gt)

                # Extract text-audio aligned encoded features and assign to tensor
                ph_algn[bidx] = h_algn[bidx, :, beg_gt : beg_gt + mel_len_gt]
                # Extract ground-truth mel spectrogram and assign to tensor
                mel_gt[bidx] = mels[bidx, :, (beg_gt * 2) : ((beg_gt + mel_len_gt) * 2)]
                # Extract corresponding ground-truth audio and assign to tensor
                beg_idx_wav = (beg_gt * 2) * cfg.preprocess_params.spect_params.hop_length
                end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                # --- Segment for mel_st ---
                # Style reference (better to be different from the GT)
                beg_st = np.random.randint(0, mel_len - mel_len_st)
                # Extract style reference melspec for style conditioning and assign to tensor
                mel_st[bidx] = mels[bidx, :, (beg_st * 2) : ((beg_st + mel_len_st) * 2)]

            # Detach tensors to avoid unnecessary gradient tracking
            # `ph_algn` is not detached as it is used for gradient computation
            mel_gt = mel_gt.detach()
            mel_st = mel_st.detach()
            wav_gt = wav_gt.detach()

            # --- End of Pre-allocated tensors ---

            with torch.no_grad():
                # Get the pitch and norm of the ground truth samples
                norm_real = log_norm(mel_gt.unsqueeze(1)).squeeze(1).detach()
                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))

            # Style encoding:
            # - if not multispeaker, use the ground truth mel spectrogram
            # - if multispeaker, use other (style reference) mel spectrogram
            mel4style = mel_st.unsqueeze(1) if model.multispeaker else mel_gt.unsqueeze(1)

            # Only (acoustic) style encoder is trained within 1st stage training
            style = model.acoustic_style_encoder(mel4style)

            # Reconstruct the audio from the text-audio aligned encoded features, predicted style,
            # and ground truth pitch and norm
            y_rec = model.decoder(ph_algn, f0_real, norm_real, style)

            # --- Discriminator loss ---
            if epoch >= tma_epoch:
                # Compute decoder's discriminator loss
                loss_disc = dl(wav_gt.detach().unsqueeze(1).float(), y_rec.detach()).mean()

                # Use Accelerate's accumulate context manager for proper gradient accumulation
                with acc.accumulate(model.mpd, model.msd):
                    # JMa: Compute gradients only for discriminators
                    acc.backward(
                        loss_disc,
                        inputs=list(model.mpd.parameters()) + list(model.msd.parameters()),
                    )

                    # Gradient clipping
                    if cfg.grad_clip:
                        acc.clip_grad_norm_(model.mpd.parameters(), cfg.grad_clip)
                        acc.clip_grad_norm_(model.msd.parameters(), cfg.grad_clip)

                    optimizer.step("msd")
                    optimizer.step("mpd")
                    optimizer.zero_grad("msd")
                    optimizer.zero_grad("mpd")
            else:
                loss_disc = 0

            # --- Generator loss ---
            loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())

            if epoch >= tma_epoch:  # Start TMA training
                # Seq2seq loss measures the difference between the predicted and
                # ground truth text tokens
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, phonemes, ph_inp_lens):
                    loss_s2s += F.cross_entropy(
                        _s2s_pred[:_text_length], _text_input[:_text_length]
                    )
                loss_s2s /= phonemes.size(0)

                # Monotonic attention loss measures the difference between the
                # predicted and ground truth attention weights
                loss_mono = F.l1_loss(d_algn, d_algn_mono) * 10

                # Generator loss measures the difference between the
                # predicted and ground truth waveforms
                loss_gen_all = gl(wav_gt.detach().unsqueeze(1).float(), y_rec).mean()
                # SLM loss to ensure the generated audio follows natural speech patterns
                loss_slm = wl(wav_gt.detach(), y_rec).mean()

                # Final generator loss is a weighted sum of the above losses
                loss_gen = (
                    cfg.loss_params.lambda_mel * loss_mel
                    + cfg.loss_params.lambda_mono * loss_mono
                    + cfg.loss_params.lambda_s2s * loss_s2s
                    + cfg.loss_params.lambda_gen * loss_gen_all
                    + cfg.loss_params.lambda_slm * loss_slm
                )

            else:
                loss_s2s = 0
                loss_mono = 0
                loss_gen_all = 0
                loss_slm = 0
                loss_gen = loss_mel

            # Prepare models for gradient accumulation
            models_to_accumulate = [model.decoder, model.acoustic_style_encoder, model.text_encoder]
            if epoch >= tma_epoch:
                models_to_accumulate.append(model.text_aligner)

            # Use Accelerate's accumulate context manager for proper gradient accumulation
            with acc.accumulate(*models_to_accumulate):
                # JMa: Compute gradients only for generator
                inputs = (
                    list(model.decoder.parameters())
                    + list(model.acoustic_style_encoder.parameters())
                    + list(model.text_encoder.parameters())
                )
                if epoch >= tma_epoch:
                    inputs += list(model.text_aligner.parameters())
                acc.backward(loss_gen, inputs=inputs)

                # Gradient clipping
                if cfg.grad_clip:
                    acc.clip_grad_norm_(model.text_encoder.parameters(), cfg.grad_clip)
                    acc.clip_grad_norm_(model.acoustic_style_encoder.parameters(), cfg.grad_clip)
                    acc.clip_grad_norm_(model.decoder.parameters(), cfg.grad_clip)
                    if epoch >= tma_epoch:
                        acc.clip_grad_norm_(model.text_aligner.parameters(), cfg.grad_clip)

                optimizer.step("text_encoder")
                optimizer.step("acoustic_style_encoder")
                optimizer.step("decoder")

                if epoch >= tma_epoch:
                    optimizer.step("text_aligner")
                    # JMa: pitch extractor should not be updated, see:
                    # https://github.com/yl4579/StyleTTS2/issues/10#issuecomment-1783701686
                    # optimizer.step('pitch_extractor')

                # Zero all gradients
                optimizer.zero_grad()

            # Accumulate mean mel-spectrogram loss (over all GPUs) across batches for logging
            # Only accumulate when gradients are synced (actual update step)
            if acc.sync_gradients:
                running_loss += acc.gather(loss_mel).mean().item()
                updates += 1  # Increment update counter only when gradients are actually applied

            iters += 1  # Increment iteration counter

            # Log training progress
            if (batch_idx + 1) % cfg.log_interval == 0:
                # Calculate average loss based on number of actual updates since last log
                num_updates_since_log = updates - updates_at_last_log
                avg_loss_mel = running_loss / max(1, num_updates_since_log)
                curr_updates = min(updates - updates_at_epoch_start, updates_per_epoch)
                logger.info(
                    "Epoch [%3d/%d], Step [%4d/%d], Upd [%4d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    steps_per_epoch,
                    curr_updates,
                    updates_per_epoch,
                    avg_loss_mel,
                    loss_gen_all,
                    loss_disc,
                    loss_mono,
                    loss_s2s,
                    loss_slm,
                )

                if acc.is_main_process:
                    # Check current VRAM usage
                    curr_vrams = [
                        nvidia_smi.nvmlDeviceGetMemoryInfo(
                            nvidia_smi.nvmlDeviceGetHandleByIndex(device_idx)
                        ).used
                        for device_idx in range(n_gpus)
                    ]
                    # Update max VRAM usage
                    curr_vram = max(curr_vrams) >> 30  # Convert bytes to GB
                    max_vram = max(max_vram, curr_vram)

                    wb_logger.log(
                        {
                            "train/mel_loss": avg_loss_mel,
                            "train/gen_loss": loss_gen_all,
                            "train/disc_loss": loss_disc,
                            "train/mono_loss": loss_mono,
                            "train/s2s_loss": loss_s2s,
                            "train/slm_loss": loss_slm,
                            "train/curr_vram": curr_vram,
                            "train/max_vram": max_vram,
                            "train/epoch": epoch,
                        },
                        step=iters,
                    )

                    logger.info(
                        "Max VRAM usage: %d/%d GB (%.2f%%)",
                        max_vram,
                        total_vram,
                        max_vram / total_vram * 100,
                    )
                    logger.info("Time elapsed: %.2f seconds", time.time() - start_time)

                running_loss = 0  # Reset running loss for next log interval
                updates_at_last_log = updates  # Update checkpoint for next interval

        # === Start of validation part ==============================================

        # Validation
        loss_test = 0
        # Set all models to eval mode
        model.set_mode("eval")

        with torch.no_grad():
            iters_test = 0
            for _, batch in enumerate(val_dataloader):
                # optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                (
                    phonemes,  # Padded input phoneme IDs [B, T_text]
                    ph_inp_lens,  # Input phoneme lengths [B]
                    _,  # OOD texts not used in 1st stage training
                    _,  # OOD phoneme lengths not used in 1st stage training
                    mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                    mel_inp_len,  # Mel spectrogram lengths [B]
                    _,  # Reference mel spectrograms not used in 1st stage
                ) = batch
                # Current batch size
                bsize = mel_inp_len.shape[0]

                with torch.no_grad():
                    mel_mask = length_to_mask(mel_inp_len // (2**n_down)).to(device)
                    _, s2s_pred, d_algn = model.text_aligner(mels, mel_mask, phonemes)

                    d_algn = d_algn.transpose(-1, -2)
                    d_algn = d_algn[..., 1:]
                    d_algn = d_algn.transpose(-1, -2)

                    ph_mask = length_to_mask(ph_inp_lens).to(phonemes.device)
                    attn_mask = (
                        (~mel_mask)
                        .unsqueeze(-1)
                        .expand(mel_mask.shape[0], mel_mask.shape[1], ph_mask.shape[-1])
                        .float()
                        .transpose(-1, -2)
                    )
                    attn_mask = (
                        attn_mask.float()
                        * (~ph_mask)
                        .unsqueeze(-1)
                        .expand(ph_mask.shape[0], ph_mask.shape[1], mel_mask.shape[-1])
                        .float()
                    )
                    attn_mask = attn_mask < 1
                    # d_algn.masked_fill_(attn_mask, 0.0)
                    # Out-of-place to avoid autograd version bumps issues
                    d_algn = d_algn.masked_fill(attn_mask, 0.0)

                # Encode phonemes
                h_ph = model.text_encoder(phonemes, ph_inp_lens, ph_mask)

                h_algn = h_ph @ d_algn

                # Get clips
                # Note: Validation uses local min length, not gathered length like training
                # mel_input_length_all = accelerator.gather(mel_input_length)  # for balanced load
                mel_len_gt = min([int(mel_inp_len.min().item() / 2 - 1), cfg.max_len // 2])

                # --- Pre-allocate tensors ---
                # Calculate fixed waveform segment length
                wav_len = (mel_len_gt * 2) * cfg.preprocess_params.spect_params.hop_length

                # Pre-allocate tensors with the calculated fixed length
                # Note: Style tensor `mel_st` is not used in validation
                ph_algn = torch.empty(
                    bsize,
                    h_algn.shape[1],
                    mel_len_gt,
                    device=device,
                    dtype=h_algn.dtype,
                )
                mel_gt = torch.empty(
                    bsize,
                    mels.shape[1],
                    mel_len_gt * 2,
                    device=device,
                    dtype=mels.dtype,
                )
                wav_gt = torch.empty(bsize, wav_len, device=device, dtype=torch.float)

                # Iterate through the batch samples
                for bidx in range(bsize):
                    # Mel-spectrogram length (dividing by 2 due to a downsampling factor?)
                    mel_len = int(mel_inp_len[bidx].item() / 2)

                    # Randomly select a start point for the mel spectrogram within valid range
                    beg_gt = np.random.randint(0, mel_len - mel_len_gt)

                    # Extract text-audio aligned encoded features and assign to tensor
                    ph_algn[bidx] = h_algn[bidx, :, beg_gt : beg_gt + mel_len_gt]
                    # Extract ground-truth mel spectrogram and assign to tensor
                    mel_gt[bidx] = mels[bidx, :, (beg_gt * 2) : ((beg_gt + mel_len_gt) * 2)]
                    # Extract corresponding ground-truth audio and assign to tensor
                    beg_idx_wav = (beg_gt * 2) * cfg.preprocess_params.spect_params.hop_length
                    end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                    wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                # # There is no need to detach tensors as in training loop
                # wav_gt = wav_gt.detach()
                # mel_gt = mel_gt.detach()

                # --- End of Pre-allocated tensors ---

                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                style = model.acoustic_style_encoder(mel_gt.unsqueeze(1))
                norm_real = log_norm(mel_gt.unsqueeze(1)).squeeze(1)
                y_rec = model.decoder(ph_algn, f0_real, norm_real, style)

                # Compute mel-spectrogram loss
                loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())

                loss_test += acc.gather(loss_mel).mean().item()
                iters_test += 1

        # Compute average loss over all validation batches
        curr_loss = loss_test / iters_test
        # Update best_loss
        best_loss = min(curr_loss, best_loss)

        logger.info(
            "Epoch [%3d/%d]: Validation loss: %.3f (best: %.3f)",
            epoch + 1,
            epochs,
            curr_loss,
            best_loss,
        )

        if acc.is_main_process:
            wb_logger.log(
                {"eval/mel_loss": curr_loss},
                step=iters,
            )
            wb_logger.summary["max_vram"] = max_vram  # Log max VRAM usage per epoch

            # Generate validation samples
            with torch.no_grad():
                # Iterate over the defined number of validation samples
                for idx in range(min(cfg.data_params.n_val_audios, bsize)):
                    mel_len = int(mel_inp_len[idx].item())
                    # Reconstruct audio from ground-truth mel spectrogram and
                    # phoneme-audio alignment
                    wav = pts.reconstruct(
                        mels[idx, :, :mel_len].unsqueeze(0),  # Ground-truth mel spectrogram
                        # Ground-truth phoneme-audio alignment
                        h_algn[idx, :, : mel_len // 2].unsqueeze(0),
                    )

                    # Write and save val audio
                    if cfg.data_params.save_val_audio and epoch % cfg.save_freq == 0:
                        outfile = f"epoch_1st_{epoch:0>5}_val-rec-{idx}.wav"
                        pts.save_wav(wav, osp.join(test_audio_dir, outfile))

                    # Save ground truth audio in given epochs
                    if epoch in (0, tma_epoch):
                        wav_gt = waves[idx].squeeze()
                        if cfg.data_params.save_val_audio:
                            outfile = f"epoch_1st_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))

            if epoch % cfg.save_freq == 0:
                model.save(
                    optimizer,
                    epoch,
                    iters,
                    curr_loss,
                    "epoch_1st",
                    log_dir,
                    cfg.max_saved_models,
                )

            # Save pre-TMA model
            if cfg.save_milestones and epoch == tma_epoch - 1:
                model.save(
                    optimizer,
                    epoch,
                    iters,
                    curr_loss,
                    "stage1_pre-tma",
                    log_dir,
                )

        # Sync after I/O so other processes wait for the main process
        acc.wait_for_everyone()

    if acc.is_main_process:
        wb_logger.summary["max_vram"] = max_vram

        # Save final 1st stage model
        final_filepath = model.save(
            optimizer,
            epoch,
            iters,
            curr_loss,
            "epoch_1st",
            log_dir,
            cfg.max_saved_models,
        )
        if epoch > tma_epoch - 1:
            try:
                first_stage_symlink = osp.join(
                    log_dir, cfg.get("first_stage_path", "first_stage.pth")
                )
                os.symlink(osp.basename(final_filepath), first_stage_symlink)
                logger.info("Final first-stage model saved to %s", final_filepath)
            except FileExistsError:
                logger.warning(
                    "Symlink or file %s already exists\
                    => %s was not symlinked!",
                    first_stage_symlink,
                    final_filepath,
                )

        # Ending work with NVIDIA NVLM
        nvidia_smi.nvmlShutdown()
        logger.info(
            "Max VRAM usage: %d/%d GB (%.2f%%)",
            max_vram,
            total_vram,
            max_vram / total_vram * 100,
        )
        logger.info("NVLM shutdown")

        # End logging
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Bezpečné ukončení DDP (i po výjimce)
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
