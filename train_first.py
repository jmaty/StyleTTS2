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
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs

# from accelerate.logging import get_logger
from monotonic_align import mask_from_lens
from munch import Munch
from torch.utils.tensorboard import SummaryWriter

from logger import get_logger, setup_logging
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, create_slm_loss
from meldataset import build_dataloader
from models import build_model, load_ASR_models, load_checkpoint, load_F0_models, save_checkpoint
from Modules.pts import PTS
from optimizers import build_optimizer
from text_utils import TextCleaner
from utils import (
    get_data_path_list,
    get_image,
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

    writer = None

    # Set up logging
    log_dir = config["log_dir"]
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

    # Distrinuted computing
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])
    if accelerator.is_main_process:
        writer = SummaryWriter(osp.join(log_dir, "tensorboard"))

    # Set up device
    device = accelerator.device

    # Init NVLM
    nvidia_smi.nvmlInit()
    n_gpus = nvidia_smi.nvmlDeviceGetCount()
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        logger.debug("Number of symbols: %d", len(text_cleaner))
    assert len(text_cleaner) == 81, f"Number of symbols must be 81 but it is {len(text_cleaner)}"
    assert (
        model_params.n_token == 81
    ), f"Number of tokens must be 81 but it is {model_params.n_token}"

    # Load utility models
    with accelerator.main_process_first():
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
        model[k] = accelerator.prepare(model[k])

    # Load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    # Set up dataset parameters (from config)
    dataset_config = {
        "sr": sr,
        "min_length": data_params["min_length"],
        "max_length": bert_size,  # limit max length of the input sequence to the max length of the BERT model
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
        "use_ref_mel": False,
    }

    # Prepare dataloaders
    logger.info("Building training dataloader...")
    train_dataloader = build_dataloader(
        train_list,
        root_path,
        text_cleaner,
        ood_data=None,  # OOD data not used for 1st stage training
        batch_size=batch_size,
        num_workers=args.num_workers,
        device=device,
        dataset_config=dataset_config,
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
    )
    train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)

    scheduler_params = {
        "max_lr": float(config["optimizer_params"].get("lr", 1e-4)),
        "pct_start": float(config["optimizer_params"].get("pct_start", 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    # Move models to device (cuda)
    _ = [model[key].to(device) for key in model]

    # initialize optimizers after preparing models for compatibility with FSDP
    parameters_dict = {key: model[key].parameters() for key in model}
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    lr = float(config["optimizer_params"].get("lr", 1e-4))
    optimizer = build_optimizer(parameters_dict, scheduler_params_dict, lr)

    for k, _ in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    with accelerator.main_process_first():
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

    # # Total number of steps given the batch size
    # tot_num_steps = len(train_list) // batch_size
    # Number of steps per epoch for the current process
    steps_this_epoch = len(train_dataloader)

    best_loss = float("inf")  # best test loss

    if accelerator.is_main_process:
        logger.info(" > Start training cycles:")
        logger.info(" | > Starting epoch:   %d", start_epoch)
        logger.info(" | > Total epochs:     %d", epochs)
        logger.info(" | > Steps per epoch:  %d", steps_this_epoch)
        logger.info(" | > Input iterations: %d\n", iters)

    # === Start of training loop ==============================================

    # Iterate through the defined number of epochs
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        # Set all models to train mode
        _ = [model[key].train() for key in model]

        # JMa: Zero gradients of all optimizers at each epoch start
        optimizer.zero_grad()

        # Train loop for each epoch
        for i, batch in enumerate(train_dataloader):
            waves = batch[0]  # Keep ground truth audio
            # Move other batch tensors to device
            batch = [b.to(device) for b in batch[1:]]
            # Keep individual batch tensors
            (
                spk_embs,  # Speaker embeddings [B, 512]
                texts,  # Padded input phoneme IDs [B, T_text]
                input_lengths,  # Input phoneme lengths [B]
                _,
                _,
                mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                mel_input_length,  # Mel spectrogram lengths [B]
                _,
            ) = batch

            # Generate masks for text and mel spectrograms
            with torch.no_grad():
                # `2**n_down` scaling ensures the mask aligns with the downsampled feature dimension
                mel_mask = length_to_mask(mel_input_length // (2**n_down)).to(
                    mel_input_length.device
                )
                text_mask = length_to_mask(input_lengths).to(texts.device)

            # Align text and audio (mel)
            _, s2s_pred, s2s_attn = model.text_aligner(mels, mel_mask, texts)
            # Refine attention matrix
            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            # Create attention mask
            with torch.no_grad():
                attn_mask = (
                    (~mel_mask)
                    .unsqueeze(-1)
                    .expand(mel_mask.shape[0], mel_mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask.float()
                    * (~text_mask)
                    .unsqueeze(-1)
                    .expand(text_mask.shape[0], text_mask.shape[1], mel_mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1  # Convert to boolean tensor

            s2s_attn.masked_fill_(attn_mask, 0.0)  # Apply attention mask to the attention matrix

            with torch.no_grad():
                # Create monotonic attention
                mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_st)

            # Encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = t_en @ s2s_attn
            else:
                asr = t_en @ s2s_attn_mono

            # Get clips
            # TODO: not to divide by 2?
            # TODO: get max (+ padding) instead of min?

            # # --- Original code ---

            # mel_input_length_all = accelerator.gather(mel_input_length)  # for balanced load
            # mel_len_gt = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])
            # mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            # en, mel_gt, wav_gt, mel_st = [], [], [], []
            # # Iterate through the batch samples
            # for idx, (mel_input_length_item, wave_item) in enumerate(zip(mel_input_length, waves)):
            #     # Mel-spectrogram length (dividing by 2 due to a downsampling factor?)
            #     mel_length = int(mel_input_length_item.item() / 2)
            #     # Randomly select a start point for the mel spectrogram within valid range
            #     random_start = np.random.randint(0, mel_length - mel_len_gt)
            #     # Extract text-audio aligned encoded features
            #     en.append(asr[idx, :, random_start : random_start + mel_len_gt])
            #     # Extract ground-truth mel spectrogram
            #     mel_gt.append(mels[idx, :, (random_start * 2) : ((random_start + mel_len_gt) * 2)])
            #     # Extract corresponding ground-truth audio
            #     y = wave_item[
            #         (random_start * 2) * hop_length : ((random_start + mel_len_gt) * 2) * hop_length
            #     ]
            #     wav_gt.append(y.to(device))

            #     # Style reference (better to be different from the GT)
            #     random_start = np.random.randint(0, mel_length - mel_len_st)
            #     # Extract style reference mel spectrogram for style conditioning
            #     mel_st.append(mels[idx, :, (random_start * 2) : ((random_start + mel_len_st) * 2)])

            # # Stack the extracted features into batched-sized tensors
            # en = torch.stack(en)
            # mel_gt = torch.stack(mel_gt).detach()
            # mel_st = torch.stack(mel_st).detach()
            # wav_gt = torch.stack(wav_gt).float().detach()

            # # --- End of Original code ---

            # --- Pre-allocate tensors ---

            mel_input_length_all = accelerator.gather(mel_input_length)  # for balanced load
            mel_len_gt = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)

            bsize = mel_input_length.shape[0]  # Use current batch size
            wav_len = (mel_len_gt * 2) * hop_length  # Calculate fixed waveform segment length

            # Pre-allocate tensors with the calculated fixed length
            en = torch.empty(bsize, asr.shape[1], mel_len_gt, device=device, dtype=asr.dtype)
            mel_gt = torch.empty(
                bsize, mels.shape[1], mel_len_gt * 2, device=device, dtype=mels.dtype
            )
            mel_st = torch.empty(
                bsize, mels.shape[1], mel_len_st * 2, device=device, dtype=mels.dtype
            )
            wav_gt = torch.empty(bsize, wav_len, device=device, dtype=torch.float)

            # Iterate through the batch samples
            for bidx in range(bsize):
                # Mel-spectrogram length (dividing by 2 due to a downsampling factor?)
                mel_length = int(mel_input_length[bidx].item() / 2)

                # --- Segment for en, mel_gt, wav_gt ---
                # Randomly select a start point for the mel spectrogram within valid range
                beg_gt = np.random.randint(0, mel_length - mel_len_gt)

                # Extract text-audio aligned encoded features and assign to tensor
                en[bidx] = asr[bidx, :, beg_gt : beg_gt + mel_len_gt]
                # Extract ground-truth mel spectrogram and assign to tensor
                mel_gt[bidx] = mels[bidx, :, (beg_gt * 2) : ((beg_gt + mel_len_gt) * 2)]
                # Extract corresponding ground-truth audio and assign to tensor
                beg_idx_wav = (beg_gt * 2) * hop_length
                end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                # --- Segment for mel_st ---
                # Style reference (better to be different from the GT)
                beg_st = np.random.randint(0, mel_length - mel_len_st)
                # Extract style reference mel spectrogram for style conditioning and assign to tensor
                mel_st[bidx] = mels[bidx, :, (beg_st * 2) : ((beg_st + mel_len_st) * 2)]

            # Detach tensors to avoid unnecessary gradient tracking
            # `en` is not detached as it is used for gradient computation
            mel_gt = mel_gt.detach()
            mel_st = mel_st.detach()
            wav_gt = wav_gt.detach()

            # --- End of Pre-allocated tensors ---

            # Segment too short to be used by the style encoder => skipping
            if mel_gt.shape[-1] < 80:
                logger.warning(
                    "GT mel spectrogram is too short => skipping batch %d in epoch %d",
                    i,
                    epoch,
                )
                continue

            with torch.no_grad():
                # Get the pitch and norm of the ground truth samples
                real_norm = log_norm(mel_gt.unsqueeze(1)).squeeze(1).detach()
                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))

            # Style encoding:
            # - if not multispeaker, use the ground truth mel spectrogram
            # - if multispeaker, use other (style reference) mel spectrogram
            s = model.style_encoder(mel_st.unsqueeze(1) if multispeaker else mel_gt.unsqueeze(1))
            s = torch.cat([spk_embs, s], dim=1)

            # Recontruct the audio from the text-audio aligned encoded features, predicted style,
            # and ground truth pitch and norm
            y_rec = model.decoder(en, f0_real, real_norm, s)

            # --- Discriminator loss ---
            if epoch >= tma_epoch:
                # Compute decoder's discriminator loss
                d_loss = dl(wav_gt.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                d_loss = d_loss / grad_accum_steps  # JMa: normalize loss
                # JMa: Compute gradients only for discriminators
                accelerator.backward(
                    d_loss, inputs=list(model.mpd.parameters()) + list(model.msd.parameters())
                )
                # JMa: Gradient accumulation
                if (i + 1) % grad_accum_steps == 0:
                    # JMa: gradient clipping
                    if grad_clip:
                        _ = [
                            accelerator.clip_grad_norm_(model[k].parameters(), grad_clip)
                            for k in model
                        ]
                    optimizer.step("msd")
                    optimizer.step("mpd")
                    optimizer.zero_grad("msd")
                    optimizer.zero_grad("mpd")
            else:
                d_loss = 0

            # --- Generator loss ---
            loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())

            if epoch >= tma_epoch:  # Start TMA training
                # Seq2seq loss measures the difference between the predicted and
                # ground truth text tokens
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                    loss_s2s += F.cross_entropy(
                        _s2s_pred[:_text_length], _text_input[:_text_length]
                    )
                loss_s2s /= texts.size(0)

                # Monotonic attention loss measures the difference between the
                # predicted and ground truth attention weights
                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

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
                + list(model.style_encoder.parameters())
                + list(model.text_encoder.parameters())
            )
            if epoch >= tma_epoch:
                inputs += list(model.text_aligner.parameters())
            accelerator.backward(g_loss, inputs=inputs)

            # Accumulate mean mel-spectrogram loss (over all GPUs) across batches for logging
            running_loss += accelerator.gather(loss_mel).mean().item()

            # JMa: Gradient accumulation
            if (i + 1) % grad_accum_steps == 0:
                # JMa: gradient clipping
                if grad_clip:
                    _ = [
                        accelerator.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model
                    ]

                optimizer.step("text_encoder")
                optimizer.step("style_encoder")
                optimizer.step("decoder")
                # optimizer.zero_grad('text_encoder')
                # optimizer.zero_grad('style_encoder')
                # optimizer.zero_grad('decoder')

                if epoch >= tma_epoch:
                    optimizer.step("text_aligner")
                    # JMa: pitch extractor should not be updated, see:
                    # https://github.com/yl4579/StyleTTS2/issues/10#issuecomment-1783701686
                    # optimizer.step('pitch_extractor')
                    # optimizer.zero_grad('text_aligner')

                # Zero all gradients
                optimizer.zero_grad()

            iters += 1

            # Log training progress
            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                mel_loss = running_loss / log_interval
                logger.info(
                    "Epoch [%3d/%d], Step [%4d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f",
                    epoch + 1,
                    epochs,
                    i + 1,
                    steps_this_epoch,  # tot_num_steps,
                    mel_loss,
                    loss_gen_all,
                    d_loss,
                    loss_mono,
                    loss_s2s,
                    loss_slm,
                )
                writer.add_scalar("train/mel_loss", mel_loss, iters)
                writer.add_scalar("train/gen_loss", loss_gen_all, iters)
                writer.add_scalar("train/d_loss", d_loss, iters)
                writer.add_scalar("train/mono_loss", loss_mono, iters)
                writer.add_scalar("train/s2s_loss", loss_s2s, iters)
                writer.add_scalar("train/slm_loss", loss_slm, iters)

                for device_idx in range(n_gpus):
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_idx)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    logger.info(
                        "Device %d VRAM usage: %d/%d GB (%.2f%%)",
                        device_idx,
                        info.used >> 30,
                        info.total >> 30,
                        info.used / info.total * 100,
                    )
                logger.info("Time elapsed: %.2f seconds", time.time() - start_time)

                running_loss = 0  # Reset running loss for next log interval

        # === Start of validation part ==============================================

        # Validation
        loss_test = 0
        # Set all models to eval mode
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for _, batch in enumerate(val_dataloader):
                # optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                (
                    spk_embs,  # Speaker embeddings [B, 512]
                    texts,  # Padded input phoneme IDs [B, T_text]
                    input_lengths,  # Input phoneme lengths [B]
                    _,
                    _,
                    mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                    mel_input_length,  # Mel spectrogram lengths [B]
                    _,
                ) = batch

                with torch.no_grad():
                    mel_mask = length_to_mask(mel_input_length // (2**n_down)).to("cuda")
                    _, s2s_pred, s2s_attn = model.text_aligner(mels, mel_mask, texts)

                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    attn_mask = (
                        (~mel_mask)
                        .unsqueeze(-1)
                        .expand(mel_mask.shape[0], mel_mask.shape[1], text_mask.shape[-1])
                        .float()
                        .transpose(-1, -2)
                    )
                    attn_mask = (
                        attn_mask.float()
                        * (~text_mask)
                        .unsqueeze(-1)
                        .expand(text_mask.shape[0], text_mask.shape[1], mel_mask.shape[-1])
                        .float()
                    )
                    attn_mask = attn_mask < 1
                    s2s_attn.masked_fill_(attn_mask, 0.0)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)

                asr = t_en @ s2s_attn

                # Get clips
                # Note: Validation uses local min length, not gathered length like training
                # mel_input_length_all = accelerator.gather(mel_input_length)  # for balanced load
                mel_len_gt = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])

                # # --- Original code ---

                # en, mel_gt, wav_gt = [], [], []
                # for idx, (mel_input_length_item, wave_item) in enumerate(
                #     zip(mel_input_length, waves)
                # ):
                #     mel_length = int(mel_input_length_item.item() / 2)

                #     random_start = np.random.randint(0, mel_length - mel_len_gt)
                #     en.append(asr[idx, :, random_start : random_start + mel_len_gt])
                #     mel_gt.append(
                #         mels[idx, :, (random_start * 2) : ((random_start + mel_len_gt) * 2)]
                #     )
                #     y = wave_item[
                #         (random_start * 2)
                #         * hop_length : ((random_start + mel_len_gt) * 2)
                #         * hop_length
                #     ]
                #     # Change to device
                #     wav_gt.append(y.to(device))

                # wav_gt = torch.stack(wav_gt).float().detach()
                # en = torch.stack(en)
                # mel_gt = torch.stack(mel_gt).detach()

                # # --- Original code ---

                # --- Pre-allocate tensors ---

                bsize = mel_input_length.shape[0]  # Use current batch size
                wav_len = (mel_len_gt * 2) * hop_length  # Calculate fixed waveform segment length

                # Pre-allocate tensors with the calculated fixed length
                # Note: Style tensor `mel_st` is not used in validation
                en = torch.empty(bsize, asr.shape[1], mel_len_gt, device=device, dtype=asr.dtype)
                mel_gt = torch.empty(
                    bsize, mels.shape[1], mel_len_gt * 2, device=device, dtype=mels.dtype
                )
                wav_gt = torch.empty(bsize, wav_len, device=device, dtype=torch.float)

                # Iterate through the batch samples
                for bidx in range(bsize):
                    # Mel-spectrogram length (dividing by 2 due to a downsampling factor?)
                    mel_length = int(mel_input_length[bidx].item() / 2)

                    # Randomly select a start point for the mel spectrogram within valid range
                    beg_gt = np.random.randint(0, mel_length - mel_len_gt)

                    # Extract text-audio aligned encoded features and assign to tensor
                    en[bidx] = asr[bidx, :, beg_gt : beg_gt + mel_len_gt]
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

                s = model.style_encoder(mel_gt.unsqueeze(1))
                s = torch.cat([spk_embs, s], dim=1)

                real_norm = log_norm(mel_gt.unsqueeze(1)).squeeze(1)
                y_rec = model.decoder(en, f0_real, real_norm, s)

                loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())

                loss_test += accelerator.gather(loss_mel).mean().item()
                iters_test += 1

        if accelerator.is_main_process:
            logger.info(
                "Epoch [%3d/%d]: validation loss: %.3f",
                epoch + 1,
                epochs,
                loss_test / iters_test,
            )
            writer.add_scalar("eval/mel_loss", loss_test / iters_test, epoch)
            attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            writer.add_figure("eval/attn", attn_image, epoch)

            with torch.no_grad():
                # Iterate over the defined number of validation samples
                for idx in range(min(n_val_audios, bsize)):
                    mel_length = int(mel_input_length[idx].item())
                    # Reconstruct audio from ground-truth mel spectrogram and
                    # phoneme-audio alignment
                    wav = pts.reconstruct(
                        mels[idx, :, :mel_length].unsqueeze(0),  # Ground-truth mel spectrogram
                        # Ground-truth phoneme-audio alignment
                        asr[idx, :, : mel_length // 2].unsqueeze(0),
                        spk_embs[idx].unsqueeze(0),
                    )

                    # Write and save val audio
                    writer.add_audio(f"eval/y{idx}", wav, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_1st_{epoch:0>5}_val-rec-{idx}.wav"
                        pts.save_wav(wav, os.path.join(test_audio_dir, outfile))

                    # Save ground truth
                    if epoch == 0:
                        wav_gt = waves[idx].squeeze()
                        if save_val_audio:
                            outfile = f"epoch_1st_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))
                        writer.add_audio(f"gt/y{idx}", wav_gt, epoch, sample_rate=sr)

            # # --- Vectorized Preparation for Validation Audio Generation ---
            # with torch.no_grad():
            #     # Determine the number of samples to generate/save
            #     # Use the batch size from the last validation batch processed
            #     num_samples_in_batch = mel_input_length.shape[0]
            #     num_samples_to_process = min(n_val_audios, num_samples_in_batch)

            #     # Select the relevant data slices for the samples to process
            #     mels_val_subset = mels[:num_samples_to_process]
            #     asr_val_subset = asr[:num_samples_to_process]
            #     mel_input_length_subset = mel_input_length[:num_samples_to_process]
            #     # Assuming 'waves' is a list/tuple of numpy arrays from the dataloader
            #     waves_val_subset = waves[:num_samples_to_process]

            #     # Get actual lengths as a list of Python ints
            #     mel_lengths_list = [int(l.item()) for l in mel_input_length_subset]

            #     # Prepare lists of sliced tensors using list comprehension
            #     # Each element will have batch size 1 for pts.reconstruct
            #     mel_gt_list = [
            #         mels_val_subset[i, :, :length].unsqueeze(0)
            #         for i, length in enumerate(mel_lengths_list)
            #     ]
            #     en_list = [
            #         asr_val_subset[i, :, : length // 2].unsqueeze(0)
            #         for i, length in enumerate(mel_lengths_list)
            #     ]

            #     # --- Loop for Reconstruction and Saving (Hard to fully vectorize) ---
            #     # Iterate through the prepared samples
            #     for idx in range(num_samples_to_process):
            #         mel_gt = mel_gt_list[idx]
            #         en_gt = en_list[idx]

            #         # Reconstruct audio from ground-truth mel spectrogram and
            #         # phoneme-audio alignment using pts object
            #         # pts.reconstruct likely expects single-item batches
            #         # TODO: Enable reconstruction from multiple tensors
            #         wav_rec = pts.reconstruct(mel_gt, en_gt)

            #         # Write and save reconstructed validation audio
            #         writer.add_audio(f"eval/y{idx}", wav_rec, epoch, sample_rate=sr)
            #         if save_val_audio and epoch % saving_epoch == 0:
            #             outfile = f"epoch_1st_{epoch:0>5}_val-rec-{idx}.wav"
            #             # pts.save_wav likely saves a single waveform
            #             pts.save_wav(wav_rec, os.path.join(test_audio_dir, outfile))

            #         # Save original ground truth audio (only at epoch 0)
            #         if epoch == 0:
            #             # Get the numpy version of original waveform for this index
            #             wav_gt = waves_val_subset[idx].cpu().numpy().squeeze()
            #             if save_val_audio:
            #                 outfile = f"epoch_1st_{epoch:0>5}_gt-{idx}.wav"
            #                 pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))
            #             writer.add_audio(f"gt/y{idx}", wav_gt, epoch, sample_rate=sr)
            # # --- End of Vectorized Preparation and Saving Loop ---

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

    if accelerator.is_main_process:
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
        logger.info("NVLM shutdown")


if __name__ == "__main__":
    main()
