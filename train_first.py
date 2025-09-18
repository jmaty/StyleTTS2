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
from models import StyleTTS2, load_ASR_models, load_F0_models, load_spkenc_model
from Modules.pts import PTS, set_random_seed
from optimizers import build_optimizer
from text_utils import TextCleaner
from utils import (
    Resampler,
    get_data_path_list,
    length_to_mask,
    log_norm,
    maximum_path,
    nccl_warmup,
    warmup_scheduler,
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
    # parser.add_argument(
    #     "-L",
    #     "--log_level",
    #     type=str,
    #     default="INFO",
    #     help="log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    # )
    add_logging_args(parser)  # --log-level, --log-file
    args = parser.parse_args()

    # Load config
    with open(args.config_path, encoding="utf-8") as fr:
        config = munchify(yaml.safe_load(fr))

    wb_logger = None  # WandB logger

    # Set up logging
    set_random_seed(config.seed)
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # must be before Accelerator
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])

    # NCCL warm-up
    nccl_warmup(device=getattr(acc, "device", None), local_rank=local_rank)

    # Uniform logging (main process only)
    log_file = args.log_file or osp.join(log_dir, "train.log")
    setup_logging(args.log_level, log_file, accelerator=acc)
    logger = get_logger(__name__)

    # Configure logging only on main process to avoid duplicate output
    if acc.is_main_process:
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

    # Shortcuts to access parameters
    model_params = config.model_params
    loss_params = config.loss_params
    spkenc_params = config.model_params.spkenc_params

    # Optional SCL: treat as disabled when lambda_scl is absent or <= 0
    use_scl = bool(loss_params.get("lambda_scl", 0.0))

    # Optional fine-tuning of speaker encoder
    # - freeze: if False, allow training
    # - unfreeze_epoch: optionally delay training until given epoch
    # - lr: optional dedicated LR for speaker_encoder
    spkenc_unfreeze_epoch = spkenc_params.get("unfreeze_epoch", tma_epoch)
    # Note: do not force freeze; allow optional finetuning via config

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
        # Load pretrained ASR model
        asr_config = config.get("ASR_config", False)
        asr_path = config.get("ASR_path", False)
        text_aligner = load_ASR_models(asr_path, asr_config)

        # Load pretrained F0 model
        f0_path = config.get("F0_path", False)
        pitch_extractor = load_F0_models(f0_path)

        # Load BERT model
        bert_path = config.get("PLBERT_dir", False)
        plbert = load_plbert(bert_path)

        # Load speaker encoder model
        speaker_encoder = load_spkenc_model(spkenc_params.model, spkenc_params.freeze)

    # Initialize StyleTTS2 model
    logger.info("Building StyleTTS2 model...")
    model = StyleTTS2(model_params, text_aligner, pitch_extractor, plbert, speaker_encoder)
    # The following parameters must be set before accelerator.prepare()
    acoustic_style_dim = model.acoustic_style_encoder.style_dim
    bert_size = model.bert.config.max_position_embeddings  # ALBERT config

    # Prepare model for distributed training
    for k in model:
        model[k] = acc.prepare(model[k])

    # Load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    # Set up dataset parameters (from config)
    dataset_config = {
        "sr": sr,
        "min_length": data_params.min_length,
        # limit max length of the input sequence to the max length of the BERT model
        "max_length": bert_size,  # ALBERT config
        "silence_beg": config.preprocess_params.silence_beg,
        "silence_end": config.preprocess_params.silence_end,
        "n_mels": config.model_params.n_mels,
        "spect_params": config.preprocess_params.spect_params,
        "max_ref_mel_length": config.preprocess_params.max_ref_mel_length,
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
        use_speaker_sampler=bool(model.multispeaker),
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
        "steps_per_epoch": int(np.ceil(len(train_dataloader) / max(1, grad_accum_steps))),
    }

    # Move models to device (cuda)
    model.to(device)

    # Initialize optimizers after preparing models for compatibility with FSDP
    lr = float(config["optimizer_params"].get("lr", 1e-4))
    raw_param_groups = {k: list(model[k].parameters()) for k in model}
    # Leave only parameters with requires_grad=True (default),
    # but optionally include speaker_encoder params even if currently frozen,
    # so we can unfreeze later without rebuilding optimizers.
    parameters_filtered = {
        k: [p for p in v if p.requires_grad] for k, v in raw_param_groups.items()
    }
    # Always include speaker_encoder params in optimizer if multispeaker and
    # finetuning is desired now or later (unfreeze_epoch specified)
    if (
        model.multispeaker
        and "speaker_encoder" in raw_param_groups
        and (not getattr(spkenc_params, "freeze", True) or hasattr(spkenc_params, "unfreeze_epoch"))
    ):
        parameters_filtered["speaker_encoder"] = raw_param_groups["speaker_encoder"]

    not_trainable_modules = [k for k, v in parameters_filtered.items() if len(v) == 0]
    parameters_dict = {k: v for k, v in parameters_filtered.items() if v}

    if acc.is_main_process:
        logger.info("Optimizer groups: %s", list(parameters_dict.keys()))
        if not_trainable_modules:
            logger.info("Not trainable modules: %s", not_trainable_modules)
    scheduler_params_dict = {k: scheduler_params.copy() for k in parameters_dict}
    optimizer = build_optimizer(parameters_dict, scheduler_params_dict, lr)

    # Optional dedicated LR for speaker encoder
    if "speaker_encoder" in optimizer.optimizers:
        try:
            for pg in optimizer.optimizers["speaker_encoder"].param_groups:
                pg["lr"] = spkenc_params.lr
        except Exception:
            pass

    # Prepare optimizers and schedulers for distributed training
    for k, _ in optimizer.optimizers.items():
        optimizer.optimizers[k] = acc.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = acc.prepare(optimizer.schedulers[k])

    # Load model weights
    with acc.main_process_first():
        if config.get("pretrained_model", "") != "":
            optimizer, start_epoch, iters = model.load(
                config["pretrained_model"],
                optimizer,
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
        logger.info("")

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
    wl = create_slm_loss(model.slm, model.wd, sr).to(device)

    # Create test audio dir under log/eval dir
    if (save_val_audio or save_test_audio) and not os.path.exists(test_audio_dir):
        os.makedirs(test_audio_dir, exist_ok=True)

    # Create resampler for speaker encoder
    spkenc_resampler = (
        Resampler(sr, spkenc_params.get("sr", 16000), device=device) if model.multispeaker else None
    )

    # Create phoneme-to-speech object for synthesizing validation sentences
    # - use global noise for speed
    pts = PTS(config, model, use_glob_noise=True)

    # Number of iteration-steps per epoch (per process)
    steps_per_epoch = len(train_dataloader)
    # Warmup iterations (valid for mode="mix")
    # Number of update-steps per epoch (accounts for grad accumulation)
    updates_per_epoch = int(np.ceil(steps_per_epoch / max(1, grad_accum_steps)))
    # Warmup thresholds in update-krocích (stabilní vůči grad_accum_steps)
    warmup_beg_updates = int(updates_per_epoch * model_params.style_mix.warmup_beg_epoch)
    warmup_end_updates = int(updates_per_epoch * model_params.style_mix.warmup_end_epoch)
    # warmup_beg_iters = int(steps_per_epoch * model_params.style_mix.warmup_beg_epoch)
    # warmup_end_iters = int(steps_per_epoch * model_params.style_mix.warmup_end_epoch)

    best_loss = float("inf")  # best test loss

    if acc.is_main_process:
        logger.info(" > Start training cycles:")
        logger.info(" | > Random seed:         %s", config.seed)
        logger.info(" | > Experiment label:    %s", config.label)
        logger.info(" | > Starting epoch:      %d", start_epoch)
        logger.info(" | > Total epochs:        %d", epochs)
        logger.info(" | > Steps per epoch:     %d", steps_per_epoch)
        logger.info(" | > Updates per epoch:   %d", updates_per_epoch)
        logger.info(" | > Input iterations:    %d", iters)
        logger.info(" | > Style mix mode:      %s", model_params.style_mix.mode)
        logger.info(" | > Warmup mode:         %s", model_params.style_mix.warmup_mode)
        logger.info(
            " | > Warmup (epochs/it.): %d-%d / %d-%d",
            model_params.style_mix.warmup_beg_epoch,
            model_params.style_mix.warmup_end_epoch,
            # warmup_beg_iters,
            # warmup_end_iters,
            warmup_beg_updates,
            warmup_end_updates,
        )
        logger.info(" | > Train data:          %s", data_params.train_data)
        logger.info(" | > Valid data:          %s", data_params.val_data)
        logger.info(" | > Pretrained model:    %s", config.pretrained_model)
        logger.info(" | > Text aligner:        %s", config.ASR_path)
        logger.info(" | > F0 model:            %s", config.F0_path)
        logger.info(" | > PL-BERT:             %s", config.PLBERT_dir)
        logger.info(" | > Batch size:          %d", batch_size)
        logger.info(" | > Grad. accum. steps:  %d", grad_accum_steps)
        logger.info(" | > Effect. batch size:  %d", batch_size * grad_accum_steps)
        logger.info(" | > Max len:             %d", max_len)
        logger.info(" | > SLM loss:            %s", model_params.slm.model)
        logger.info(" | > Inp. spk. emb. dim:  %d", model_params.spkenc_params.dim_in)
        logger.info(" | > Acoust style dim:    %d", acoustic_style_dim)
        logger.info(" | > Pros. style dim:     %d", model_params.style_dim)
        logger.info(" | > Use SCL:             %s", use_scl)
        logger.info(
            " | > SpkEnc freeze:       %s (unfreeze@epoch=%d, lr=%s)",
            spkenc_params.freeze,
            spkenc_unfreeze_epoch,
            spkenc_params.lr,
        )
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

        # Decide if speaker encoder should be trained this epoch
        spkenc_train_enabled = (
            model.multispeaker
            and not spkenc_params.freeze
            and use_scl
            and epoch >= spkenc_unfreeze_epoch
        )
        # If we plan to train speaker encoder now, ensure its params require grads
        if spkenc_train_enabled and "speaker_encoder" in model:
            for p in model.speaker_encoder.parameters():
                p.requires_grad = True
            logger.debug("| > Speaker encoder training ENABLED at epoch %d", epoch + 1)

        # Models in train mode from the beginning
        train_components = [
            "decoder",
            "text_encoder",
        ]
        # Append acoustic style encoder if it is trainable
        if "acoustic_style_encoder" not in not_trainable_modules:
            train_components.append("acoustic_style_encoder")

        # Models in train mode based on the epoch
        if epoch >= tma_epoch:
            train_components.extend(["msd", "mpd", "text_aligner"])
            if spkenc_train_enabled:
                train_components.append("speaker_encoder")

        # Set models to train mode
        model.set_mode("train", train_components)

        # JMa: Zero gradients of all optimizers at each epoch start
        optimizer.zero_grad()

        # Train loop for each epoch
        for batch_idx, batch in enumerate(train_dataloader):
            waves = batch[0]  # Keep ground truth audio
            # Move other batch tensors to device
            batch = [b.to(device) for b in batch[1:]]
            # Keep individual batch tensors
            (
                phonemes,  # Padded input phoneme IDs [B, T_text]
                ph_inp_lens,  # Input phoneme lengths [B]
                _,  # OOD texts not used in 1st stage training
                _,  # OOD phoneme lengths not used in 1st stage training
                mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                mel_inp_len,  # Mel spectrogram lengths [B]
                _,  # Reference waveforms not used in 1st stage
                _,  # Reference mel spectrograms not used in 1st stage
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
            wav_st_len = (mel_len_st * 2) * hop_length  # Calculate fixed style segment length

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
            wav_st = torch.empty(bsize, wav_st_len, device=device, dtype=torch.float)

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
                # Extract corresponding ground-truth audio and assign to tensor
                beg_idx_wav = (beg_st * 2) * hop_length
                end_idx_wav = beg_idx_wav + wav_st_len  # Use pre-calculated length
                wav_st[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

            # Detach tensors to avoid unnecessary gradient tracking
            # `h_algn_seg` is not detached as it is used for gradient computation
            mel_gt = mel_gt.detach()
            mel_st = mel_st.detach()
            wav_gt = wav_gt.detach()
            wav_st = wav_st.detach()

            # --- End of Pre-allocated tensors ---

            with torch.no_grad():
                # Get the pitch and norm of the ground truth samples
                norm_real = log_norm(mel_gt.unsqueeze(1)).squeeze(1).detach()
                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))

            # Style encoding:
            # - if not multispeaker, use the ground truth mel spectrogram
            # - if multispeaker, use other (style reference) mel spectrogram
            mel4style = mel_st.unsqueeze(1) if model.multispeaker else mel_gt.unsqueeze(1)
            # Compute warmup coefficient
            warmup_coef = (
                warmup_scheduler(updates + 1, warmup_beg_updates, warmup_end_updates)
                if model.multispeaker and model_params.style_mix.mode == "mix"
                else None
            )
            # warmup_coef = (
            #     warmup_scheduler(iters + 1, warmup_beg_iters, warmup_end_iters)
            #     if model.multispeaker and model.mix_mode == "mix"
            #     else None
            # )

            # Speaker encoding:
            spk_embs_st = None
            # Resample ground-truth segments for speaker encoder
            seg4style = spkenc_resampler(wav_st) if model.multispeaker else spkenc_resampler(wav_gt)

            # # Target speaker embedding for style conditioning:
            # # - compute in eval() + no_grad() to avoid BN running stats in-place updates
            # # - detach to prevent grads flowing into speaker encoder via style path
            # spk_embs_tgt = None
            # if model.multispeaker:
            #     _spkenc_was_training = model.speaker_encoder.training
            #     model.speaker_encoder.eval()
            #     with torch.no_grad():
            #         spk_embs_st = model.speaker_encoder(seg4style)
            #     if _spkenc_was_training:
            #         model.speaker_encoder.train()
            #     spk_embs_tgt = spk_embs_st.detach()

            # Target speaker embedding for style conditioning:
            spk_embs_tgt = None
            if model.multispeaker:
                if spkenc_params.freeze:
                    # Frozen encoder: eval + no_grad, bez togglování módů
                    with torch.no_grad():
                        spk_embs_tgt = model.speaker_encoder(seg4style)
                else:
                    # Nezmrazený: ponech stávající chování (target bez gradů)
                    # Not-frozen encoder
                    spk_embs_st = model.speaker_encoder(seg4style)
                    spk_embs_tgt = spk_embs_st.detach()

            # Only (acoustic) style encoder is trained within 1st stage training
            style = model.acoustic_style_encoder(
                mel4style,
                # Use detached speaker embedding to avoid backprop into speaker encoder
                spk_emb=spk_embs_tgt if model.multispeaker else None,
                warmup_coef=warmup_coef,
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
                    loss_disc,
                    inputs=list(model.mpd.parameters()) + list(model.msd.parameters()),
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

                # # Calculate speaker consistency loss
                # with torch.no_grad():
                #     ## Sync speaker encoder weights
                #     # speaker_encoder_infer.load_state_dict(model.speaker_encoder.state_dict())
                #     # Calculate speaker embeddings for the reconstructed audio
                #     seg_rec_for_spkenc = spkenc_resampler(y_rec.squeeze())
                #     # spk_embs_rec = speaker_encoder_infer(seg_rec_for_spkenc)
                #     spk_embs_rec = speaker_encoder(seg_rec_for_spkenc)
                # # Compute loss: use embeddings form style waves as target speaker embeddings
                # loss_scl = 1 - F.cosine_similarity(spk_embs_st, spk_embs_rec).mean()

                # Speaker Consistency Loss (SCL) — compute only if enabled
                if use_scl:
                    seg_rec_for_spkenc = spkenc_resampler(y_rec.squeeze())
                    # reconstructed = embeddings from the reconstructed audio
                    spk_embs_rec = model.speaker_encoder(seg_rec_for_spkenc)
                    loss_scl = 1 - F.cosine_similarity(spk_embs_tgt, spk_embs_rec).mean()
                else:
                    loss_scl = 0.0

                # Final generator loss is a weighted sum of the above losses
                g_loss = (
                    loss_params.lambda_mel * loss_mel
                    + loss_params.lambda_mono * loss_mono
                    + loss_params.lambda_s2s * loss_s2s
                    + loss_params.lambda_gen * loss_gen_all
                    + loss_params.lambda_slm * loss_slm
                    + loss_params.lambda_scl * loss_scl
                )

            else:
                loss_s2s = 0
                loss_mono = 0
                loss_gen_all = 0
                loss_slm = 0
                loss_scl = 0
                g_loss = loss_mel

            g_loss = g_loss / grad_accum_steps  # JMa: normalize loss
            # JMa: Compute gradients only for generator
            inputs = (
                list(model.decoder.parameters())
                # + list(model.acoustic_style_encoder.parameters())
                + list(model.text_encoder.parameters())
            )
            if "acoustic_style_encoder" not in not_trainable_modules:
                inputs += list(model.acoustic_style_encoder.parameters())
            if epoch >= tma_epoch:
                inputs += list(model.text_aligner.parameters())
                if spkenc_train_enabled:
                    # Do not do this if speaker encoder is frozen
                    # => SCL updates decoder
                    inputs += list(model.speaker_encoder.parameters())

            acc.backward(g_loss, inputs=inputs)

            # Accumulate mean mel-spectrogram loss (over all GPUs) across batches for logging
            running_loss += acc.gather(loss_mel).mean().item()

            # JMa: Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # JMa: gradient clipping
                if grad_clip:
                    _ = [acc.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]

                optimizer.step("text_encoder")
                if "acoustic_style_encoder" not in not_trainable_modules:
                    optimizer.step("acoustic_style_encoder")
                optimizer.step("decoder")

                if epoch >= tma_epoch:
                    optimizer.step("text_aligner")
                    if spkenc_train_enabled:
                        optimizer.step("speaker_encoder")
                    # JMa: pitch extractor should not be updated, see:
                    # https://github.com/yl4579/StyleTTS2/issues/10#issuecomment-1783701686
                    # optimizer.step('pitch_extractor')

                # Zero all gradients
                optimizer.zero_grad()
                updates += 1  # Increment update counter

            iters += 1  # Increment iteration counter

            # Log training progress
            if (batch_idx + 1) % log_interval == 0:
                loss_mel = running_loss / log_interval
                curr_updates = min(updates - updates_at_epoch_start, updates_per_epoch)
                logger.info(
                    # "Epoch [%3d/%d], Step [%4d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f, SCL Loss: %.5f",
                    "Epoch [%3d/%d], Batch [%4d/%d], Upd [%4d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f, SCL Loss: %.5f",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    steps_per_epoch,
                    curr_updates,
                    updates_per_epoch,
                    loss_mel,
                    loss_gen_all,
                    loss_disc,
                    loss_mono,
                    loss_s2s,
                    loss_slm,
                    loss_scl,
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
                            "train/mel_loss": loss_mel,
                            "train/gen_loss": loss_gen_all,
                            "train/disc_loss": loss_disc,
                            "train/mono_loss": loss_mono,
                            "train/s2s_loss": loss_s2s,
                            "train/slm_loss": loss_slm,
                            "train/scl_loss": loss_scl,
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
        loss_test, loss_sim = 0, 0
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
                    _,  # Reference waveforms not used in 1st stage
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

                # Resample ground-truth segments for speaker encoder
                seg_gt_for_spkenc = spkenc_resampler(wav_gt)
                # Calculate ground truth speaker embeddings
                spk_embs_gt = model.speaker_encoder(seg_gt_for_spkenc)

                # Style encoding:
                style = model.acoustic_style_encoder(
                    # mel_gt.unsqueeze(1), spk_embs if multispeaker and epoch >= tma_epoch else None
                    mel_gt.unsqueeze(1),
                    spk_emb=spk_embs_gt if model.multispeaker else None,
                    warmup_coef=warmup_coef,
                )

                # Reconstruct the audio from the text-audio aligned encoded features,
                # predicted style, and ground truth pitch and norm
                y_rec = model.decoder(ph_algn, f0_real, norm_real, style)

                # Compute mel-spectrogram loss
                loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())

                # # Calculate speaker consistency loss
                # if epoch >= tma_epoch:
                #     spk_embs_tgt = spk_embs_gt  # target speaker embeddings
                #     # Calculate speaker embeddings for the reconstructed audio
                #     # seg_rec_for_spkenc = resample(y_rec.squeeze(), spkenc_resampler)
                #     seg_rec_for_spkenc = spkenc_resampler(y_rec.squeeze())
                #     # spk_embs_rec = speaker_encoder_infer(seg_rec_for_spkenc.detach())
                #     spk_embs_rec = model.speaker_encoder(seg_rec_for_spkenc.detach())
                #     # Compute speaker consistency loss (i.e. cosine similarity)
                #     loss_scl = 1 - F.cosine_similarity(spk_embs_tgt, spk_embs_rec)
                #     # Gather similarity loss across all processes
                #     loss_sim += acc.gather(loss_scl).mean().item()

                # Speaker consistency loss (SCL) — metric only if enabled
                if epoch >= tma_epoch and use_scl:
                    # SCL in validation: metric only (no grads), speaker_encoder is frozen
                    spk_embs_tgt = spk_embs_gt
                    seg_rec_for_spkenc = spkenc_resampler(y_rec.squeeze())
                    spk_embs_rec = model.speaker_encoder(seg_rec_for_spkenc)
                    loss_scl = 1 - F.cosine_similarity(spk_embs_tgt, spk_embs_rec)
                    loss_sim += acc.gather(loss_scl).mean().item()  # gather across all processes

                loss_test += acc.gather(loss_mel).mean().item()
                iters_test += 1

        # Compute average loss over all validation batches
        curr_loss = loss_test / iters_test
        # Update best_loss
        best_loss = min(curr_loss, best_loss)

        gate_param = acc.unwrap_model(model.acoustic_style_encoder).gate_param
        # For learnable gate, show values after sigmoid activation
        # For non-learnable gate, show raw values
        gate_values = (
            torch.sigmoid(gate_param)
            if acc.unwrap_model(model.acoustic_style_encoder).learnable_gate
            else gate_param
        )
        logger.info(
            "Epoch [%3d/%d]: Validation loss: %.3f (best: %.3f), Speaker Consistency Loss: %.3f, Warmup: %.6f, Gate weights: %.6f±%.6f (%.6f-%.6f)",
            epoch + 1,
            epochs,
            curr_loss,
            best_loss,
            loss_sim / iters_test,
            warmup_coef if model_params.style_mix.mode == "mix" else 0,
            gate_values.mean().item() if model_params.style_mix.mode == "mix" else 0,
            gate_values.std().item() if model_params.style_mix.mode == "mix" else 0,
            gate_values.min().item() if model_params.style_mix.mode == "mix" else 0,
            gate_values.max().item() if model_params.style_mix.mode == "mix" else 0,
        )

        if acc.is_main_process:
            wb_logger.log(
                {"eval/mel_loss": curr_loss, "eval/scl_loss": loss_sim / iters_test},
                step=iters,
            )

            # Generate validation samples
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
                        spk_emb=spk_embs_gt[idx].unsqueeze(0) if spk_embs_gt is not None else None,
                        warmup_coef=warmup_coef,
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
                model.save(
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
        # Save final 1st stage model
        final_filepath = model.save(
            optimizer,
            epoch,
            iters,
            curr_loss,
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
    try:
        main()
    finally:
        # Bezpečné ukončení DDP (i po výjimce)
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
