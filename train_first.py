import argparse
import logging
import os
import os.path as osp
import random
import time
import warnings

import numpy as np
import nvidia_smi
import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs

# from accelerate.logging import get_logger
from monotonic_align import mask_from_lens
from munch import Munch

# from torch.utils.tensorboard import SummaryWriter

from logger import get_logger, setup_logging
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, create_slm_loss
from meldataset import build_dataloader
from models import (
    build_model,
    load_ASR_models,
    load_checkpoint,
    load_F0_models,
    save_checkpoint,
    model2device,
    model2mode,
)
from Modules.pts import PTS
from optimizers import build_optimizer
from text_utils import TextCleaner
from utils import (
    get_data_path_list,
    # get_image,
    length_to_mask,
    log_norm,
    maximum_path,
    recursive_munch,
)
from Utils.PLBERT.util import load_plbert

warnings.simplefilter("ignore")

# Disable TF32 computations for cuDNN
torch.backends.cudnn.allow_tf32 = False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="StyleTTS2 stage 1 training")
    parser.add_argument("config_path", type=str, help="path to config")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="number of workers")
    parser.add_argument("-L", "--log_level", type=int, default=logging.INFO, help="log level")
    args = parser.parse_args()

    # Load config
    with open(args.config_path, encoding="utf-8") as fr:
        config = yaml.safe_load(fr)

    # writer = None
    wb_logger = None  # WandB logger

    # Set up logging
    log_dir = config["log_dir"]
    # exp_label = config.get("label", "")  # Experiment label
    formatter_file = logging.Formatter(
        fmt="%(levelname)s:%(asctime)s: %(message)s",
        datefmt="%y%m%d-%H:%M:%S",
    )
    setup_logging(
        level=args.log_level,
        file=osp.join(log_dir, "train.log"),
        formatter_file=formatter_file,
        level_file=args.log_level,
    )
    logger = get_logger(__name__)  # Get a logger

    # Distributed computing
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])

    if acc.is_main_process:
        # writer = SummaryWriter(osp.join(log_dir, "tensorboard"))
        # Initialize the wandb logger and name wandb project and run
        wb_logger = wandb.init(
            # Set the wandb project where this run will be logged.
            project="StyleTTS2+spkenc",
            # Set run name
            # name=f"{osp.basename(log_dir)}_{exp_label}",
            name=f"{osp.basename(log_dir)}",
            # Track hyperparameters and run metadata.
            config=config,
            dir=log_dir,
        )

    # Set up device
    device = acc.device

    # Init NVLM
    nvidia_smi.nvmlInit()
    n_gpus = nvidia_smi.nvmlDeviceGetCount()
    max_vram = 0  # Track maximum VRAM usage
    # Get total VRAM of the first GPU
    total_vram = (
        nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(0)).total >> 30
    )
    if acc.is_main_process:
        logger.info("NVLM initialized")

    # Set up training parameters
    batch_size = config.get("batch_size", 4)
    grad_accum_steps = config.get("grad_accum_steps", 1)  # JMa: gradient accumulation
    max_len = config.get("max_len", 200)
    log_interval = config.get("log_interval", 10)
    saving_epoch = config.get("save_freq", 2)
    max_saved_models = config.get("max_saved_models", 2)
    save_milestones = config.get("save_milestones", False)
    grad_clip = config.get("grad_clip", None)  # JMa: gradient clipping support

    # Set up epochs
    epochs = config["epochs"].get("stage1", 200)
    tma_epoch = config["epochs"].get("tma", 50)

    # Set up data parameters
    data_params = config.get("data_params", None)
    sr = config["preprocess_params"].get("sr", 24000)
    hop_length = config["preprocess_params"]["spect_params"].get("hop_length", 300)
    train_path = data_params["train_data"]
    val_path = data_params["val_data"]
    root_path = data_params["root_path"]
    # ood_data = data_params["OOD_data"]  # OOD texts are not used during 1st stage training
    save_val_audio = data_params.get("save_val_audio", False)
    n_val_audios = config["data_params"].get("n_val_audios", 3)
    save_test_audio = False
    test_audio_dir = os.path.join(
        config["log_dir"],
        config["data_params"].get("test_audio_dir", "test_audios"),
    )

    model_params = recursive_munch(config["model_params"])
    multispeaker = model_params.multispeaker
    loss_params = Munch(config["loss_params"])

    # Set up text cleaner and pre-processing function
    text_cleaner = TextCleaner(data_params["symbol_dict_path"], pad=data_params["pad"])
    if acc.is_main_process:
        logger.debug("Number of symbols: %d", len(text_cleaner))
    assert len(text_cleaner) == 81, f"Number of symbols must be 81 but it is {len(text_cleaner)}"
    assert (
        model_params.n_token == 81
    ), f"Number of tokens must be 81 but it is {model_params.n_token}"

    # Load utility models
    with acc.main_process_first():
        # load pretrained ASR model
        asr_config = config.get("ASR_config", False)
        asr_path = config.get("ASR_path", False)
        text_aligner = load_ASR_models(asr_path, asr_config)

        # load pretrained F0 model
        f0_path = config.get("F0_path", False)
        pitch_extractor = load_F0_models(f0_path)

        # load BERT model
        bert_path = config.get("PLBERT_dir", False)
        plbert = load_plbert(bert_path)

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    bert_size = model.bert.config.max_position_embeddings  # ALBERT config

    for k in model:
        model[k] = acc.prepare(model[k])

    # Load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    # Set up dataset parameters (from config)
    dataset_config = {
        "sr": sr,
        "min_length": data_params["min_length"],
        # limit max length of the input sequence to the max length of the BERT model
        "max_length": bert_size,
        "silence_beg": config["preprocess_params"].get("silence_beg", 4800),
        "silence_end": config["preprocess_params"].get("silence_end", 4800),
        "n_mels": config["model_params"].get("n_mels", 80),
        "spect_params": config["preprocess_params"].get(
            "spect_params",
            {
                "n_fft": 2048,
                "win_length": 1024,
                "hop_length": 300,
            },
        ),
        "use_ref_sample": False,
    }

    # Prepare dataloaders
    logger.info("Building training dataloader...")
    train_dataloader = build_dataloader(
        train_list,
        root_path,
        text_cleaner,
        validation=False,
        ood_data=None,  # OOD data not used for 1st stage training
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
        dataset_config=dataset_config,
        use_speaker_sampler=True if multispeaker else False,
    )
    logger.info("Building validation dataloader...")
    val_dataloader = build_dataloader(
        val_list,
        root_path,
        text_cleaner,
        ood_data=None,  # OOD data not used for validation
        batch_size=batch_size,
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

    scheduler_params = {
        "max_lr": float(config["optimizer_params"].get("lr", 1e-4)),
        "pct_start": float(config["optimizer_params"].get("pct_start", 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    # Move models to device (cuda)
    model = model2device(model, device)

    # initialize optimizers after preparing models for compatibility with FSDP
    parameters_dict = {key: model[key].parameters() for key in model}
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    lr = float(config["optimizer_params"].get("lr", 1e-4))
    optimizer = build_optimizer(parameters_dict, scheduler_params_dict, lr)

    for k, _ in optimizer.optimizers.items():
        optimizer.optimizers[k] = acc.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = acc.prepare(optimizer.schedulers[k])

    with acc.main_process_first():
        if config.get("pretrained_model", "") != "":
            model, optimizer, start_epoch, iters = load_checkpoint(
                model,
                optimizer,
                config["pretrained_model"],
                load_only_params=config.get("load_only_params", True),
            )
            # advance start epoch or we'd re-train and rewrite the last epoch file
            # start_epoch += 1
            logger.info("Loading pre-trained model: %s", config["pretrained_model"])
            logger.info("Starting epoch:            %d", start_epoch)
            logger.info("Starting iterations:       %d", iters)
            logger.info("")
        else:
            start_epoch = 0
            iters = 0

    # in case not distributed computing
    try:
        n_down = model.text_aligner.module.n_down
    except AttributeError:
        logger.warning("Distributed computing NOT used")
        n_down = model.text_aligner.n_down

    # wrapped losses for compatibility with mixed precision
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = create_slm_loss(model_params.slm, model.wd, sr).to(device)

    # Create test audio dir under log/eval dir
    if (save_val_audio or save_test_audio) and not os.path.exists(test_audio_dir):
        os.makedirs(test_audio_dir, exist_ok=True)

    # Create phoneme-to-speech object for synthesizing validation sentences
    # - use global noise for speed
    pts = PTS(config, model, use_glob_noise=True)

    # Number of steps per epoch for the current process
    steps_per_epoch = len(train_dataloader)

    best_loss = float("inf")  # best test loss

    if acc.is_main_process:
        logger.info(" > Start training cycles:")
        logger.info(" | > Starting epoch:   %d", start_epoch)
        logger.info(" | > Total epochs:     %d", epochs)
        logger.info(" | > Steps per epoch:  %d", steps_per_epoch)
        logger.info(" | > Input iterations: %d\n", iters)

    # === Start of training loop ==============================================
    model = model2mode(model, "eval")

    # Iterate through the defined number of epochs
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()
        train_dataloader.batch_sampler.epoch = epoch  # Set epoch for the sampler

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
        model = model2mode(model, "train", train_components)

        # JMa: Zero gradients of all optimizers at each epoch start
        optimizer.zero_grad()

        # Train loop for each epoch
        for batch_idx, batch in enumerate(train_dataloader):
            waves = batch[0]  # Keep ground truth audio
            # Move other batch tensors to device
            batch = [b.to(device) for b in batch[1:]]
            # Keep individual batch tensors
            (
                spk_embs,  # Speaker embeddings [B, spk_emd_dim]
                phonemes,  # Padded input phoneme IDs [B, T_text]
                ph_inp_lens,  # Input phoneme lengths [B]
                _,  # OOD texts not used in 1st stage training
                _,  # OOD phoneme lengths not used in 1st stage training
                mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                mel_inp_len,  # Mel spectrogram lengths [B]
                _,  # Reference mel spectrograms not used in 1st stage
                _,  # Reference speaker embeddings not used in 1st stage
            ) = batch

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

            d_algn.masked_fill_(attn_mask, 0.0)  # Apply attention mask to the attention matrix

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
            mel_len_gt = min([int(mel_inp_len_all.min().item() / 2 - 1), max_len // 2])
            # Early check for segment length:
            # - mel_len_gt * 2 is the length of the original mel spectrogram
            # - multiplication by 2 is due to the downsampling factor between mel and text aligner
            if mel_len_gt * 2 < 80:
                logger.warning(
                    "Segment is too short (%d frames, %d samples)=> skipping batch %d.",
                    mel_len_gt * 2,
                    (mel_len_gt * 2) * hop_length,
                    batch_idx,
                )
                continue

            mel_len_st = int(mel_inp_len.min().item() / 2 - 1)

            bsize = mel_inp_len.shape[0]  # Use current batch size
            wav_len = (mel_len_gt * 2) * hop_length  # Calculate fixed waveform segment length

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
                beg_idx_wav = (beg_gt * 2) * hop_length
                end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                # --- Segment for mel_st ---
                # Style reference (better to be different from the GT)
                beg_st = np.random.randint(0, mel_len - mel_len_st)
                # Extract style reference mel spectrogram for style conditioning and assign to tensor
                mel_st[bidx] = mels[bidx, :, (beg_st * 2) : ((beg_st + mel_len_st) * 2)]

            # Detach tensors to avoid unnecessary gradient tracking
            # `h_algn_seg` is not detached as it is used for gradient computation
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
            mel4style = mel_st.unsqueeze(1) if multispeaker else mel_gt.unsqueeze(1)
            # Only (acoustic) style encoder is trained within 1st stage training
            style = model.acoustic_style_encoder(
                mel4style, spk_embs if multispeaker and epoch >= tma_epoch else None
            )

            # Reconstruct the audio from the text-audio aligned encoded features, predicted style,
            # and ground truth pitch and norm
            y_rec = model.decoder(ph_algn, f0_real, norm_real, style)

            # --- Discriminator loss ---
            if epoch >= tma_epoch:
                # Compute decoder's discriminator loss
                loss_disc = dl(wav_gt.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                loss_disc = loss_disc / grad_accum_steps  # JMa: normalize loss
                # JMa: Compute gradients only for discriminators
                acc.backward(
                    loss_disc, inputs=list(model.mpd.parameters()) + list(model.msd.parameters())
                )
                # JMa: Gradient accumulation
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # JMa: gradient clipping
                    if grad_clip:
                        _ = [acc.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
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
                g_loss = (
                    loss_params.lambda_mel * loss_mel
                    + loss_params.lambda_mono * loss_mono
                    + loss_params.lambda_s2s * loss_s2s
                    + loss_params.lambda_gen * loss_gen_all
                    + loss_params.lambda_slm * loss_slm
                )

            else:
                loss_s2s = 0
                loss_mono = 0
                loss_gen_all = 0
                loss_slm = 0
                g_loss = loss_mel

            g_loss /= grad_accum_steps  # JMa: normalize loss
            # JMa: Compute gradients only for generator
            inputs = (
                list(model.decoder.parameters())
                + list(model.acoustic_style_encoder.parameters())
                + list(model.text_encoder.parameters())
            )
            if epoch >= tma_epoch:
                inputs += list(model.text_aligner.parameters())
            acc.backward(g_loss, inputs=inputs)

            # Accumulate mean mel-spectrogram loss (over all GPUs) across batches for logging
            running_loss += acc.gather(loss_mel).mean().item()

            # JMa: Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # JMa: gradient clipping
                if grad_clip:
                    _ = [acc.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]

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

            iters += 1

            # Log training progress
            if (batch_idx + 1) % log_interval == 0 and acc.is_main_process:
                mel_loss = running_loss / log_interval
                logger.info(
                    "Epoch [%3d/%d], Step [%4d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f, Fusion Weight: %.5f",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    steps_per_epoch,
                    mel_loss,
                    loss_gen_all,
                    loss_disc,
                    loss_mono,
                    loss_s2s,
                    loss_slm,
                    acc.unwrap_model(model.acoustic_style_encoder).fusion_weight.item(),
                )

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
                        "train/mel_loss": mel_loss,
                        "train/gen_loss": loss_gen_all,
                        "train/disc_loss": loss_disc,
                        "train/mono_loss": loss_mono,
                        "train/s2s_loss": loss_s2s,
                        "train/slm_loss": loss_slm,
                        "train/fusion_weight": acc.unwrap_model(
                            model.acoustic_style_encoder
                        ).fusion_weight.item(),
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

        # === Start of validation part ==============================================

        # Validation
        loss_test = 0
        # Set all models to eval mode
        model = model2mode(model, "eval")

        with torch.no_grad():
            iters_test = 0
            for _, batch in enumerate(val_dataloader):
                # optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                (
                    spk_embs,  # Speaker embeddings [B, spk_emb_dim]
                    phonemes,  # Padded input phoneme IDs [B, T_text]
                    ph_inp_lens,  # Input phoneme lengths [B]
                    _,  # OOD texts not used in 1st stage training
                    _,  # OOD phoneme lengths not used in 1st stage training
                    mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                    mel_inp_len,  # Mel spectrogram lengths [B]
                    _,  # Reference mel spectrograms not used in 1st stage
                    _,  # Reference speaker embeddings not used in 1st stage
                ) = batch
                # Current batch size
                bsize = mel_inp_len.shape[0]

                with torch.no_grad():
                    mel_mask = length_to_mask(mel_inp_len // (2**n_down)).to("cuda")
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
                    d_algn.masked_fill_(attn_mask, 0.0)

                # Encode phonemes
                h_ph = model.text_encoder(phonemes, ph_inp_lens, ph_mask)

                h_algn = h_ph @ d_algn

                # Get clips
                # Note: Validation uses local min length, not gathered length like training
                # mel_input_length_all = accelerator.gather(mel_input_length)  # for balanced load
                mel_len_gt = min([int(mel_inp_len.min().item() / 2 - 1), max_len // 2])

                # --- Pre-allocate tensors ---
                wav_len = (mel_len_gt * 2) * hop_length  # Calculate fixed waveform segment length

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
                    beg_idx_wav = (beg_gt * 2) * hop_length
                    end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                    wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                # # There is no need to detach tensors as in training loop
                # wav_gt = wav_gt.detach()
                # mel_gt = mel_gt.detach()

                # --- End of Pre-allocated tensors ---

                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                norm_real = log_norm(mel_gt.unsqueeze(1)).squeeze(1)

                # Style encoding:
                style = model.acoustic_style_encoder(
                    mel_gt.unsqueeze(1), spk_embs if multispeaker and epoch >= tma_epoch else None
                )

                # Reconstruct the audio from the text-audio aligned encoded features, predicted style,
                # and ground truth pitch and norm
                y_rec = model.decoder(ph_algn, f0_real, norm_real, style)

                # Compute mel-spectrogram loss
                loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())

                loss_test += acc.gather(loss_mel).mean().item()
                iters_test += 1

        if acc.is_main_process:
            logger.info(
                "Epoch [%3d/%d]: Validation loss: %.3f",
                epoch + 1,
                epochs,
                loss_test / iters_test,
            )
            # attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            wb_logger.log(
                {"eval/mel_loss": loss_test / iters_test},
                step=iters,
            )

            with torch.no_grad():
                # Iterate over the defined number of validation samples
                for idx in range(min(n_val_audios, bsize)):
                    mel_len = int(mel_inp_len[idx].item())
                    # Reconstruct audio from ground-truth mel spectrogram and
                    # phoneme-audio alignment
                    wav = pts.reconstruct(
                        mels[idx, :, :mel_len].unsqueeze(0),  # Ground-truth mel spectrogram
                        # Ground-truth phoneme-audio alignment
                        h_algn[idx, :, : mel_len // 2].unsqueeze(0),
                        spk_embs[idx].unsqueeze(0) if multispeaker and epoch >= tma_epoch else None,
                    )

                    # Write and save val audio
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_1st_{epoch:0>5}_val-rec-{idx}.wav"
                        pts.save_wav(wav, osp.join(test_audio_dir, outfile))

                    # Save ground truth audio in given epochs
                    if epoch in (0, tma_epoch):
                        wav_gt = waves[idx].squeeze()
                        if save_val_audio:
                            outfile = f"epoch_1st_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))

            if epoch % saving_epoch == 0:
                curr_loss = loss_test / iters_test
                best_loss = min(curr_loss, best_loss)
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    iters,
                    curr_loss,
                    "epoch_1st",
                    log_dir,
                    max_saved_models,
                )
            # Save pre-TMA model
            if save_milestones and epoch == tma_epoch - 1:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    iters,
                    loss_test / iters_test,
                    "stage1_pre-tma",
                    log_dir,
                )

    if acc.is_main_process:
        # Save final 1st stage model
        final_filepath = save_checkpoint(
            model,
            optimizer,
            epoch,
            iters,
            loss_test / iters_test,
            "epoch_1st",
            log_dir,
            max_saved_models,
        )
        if epoch > tma_epoch - 1:
            try:
                first_stage_symlink = osp.join(
                    log_dir, config.get("first_stage_path", "first_stage.pth")
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
    main()
