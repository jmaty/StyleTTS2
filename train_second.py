import argparse
import copy
import os
import os.path as osp
import time
import traceback
import warnings

import numpy as np
import nvidia_smi
import torch
import torch.nn.functional as F
import wandb
import yaml
from IPython.core.debugger import set_trace
from monotonic_align import mask_from_lens
from munch import munchify
from torch import nn

from logger import add_logging_args, get_logger, setup_logging
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, create_slm_loss
from meldataset import build_dataloader
from models import StyleTTS2, load_ASR_models, load_F0_models, load_spkenc_model
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Modules.pts import PTS, set_random_seed
from Modules.slmadv import SLMAdversarialLoss
from optimizers import build_optimizer
from text_utils import TextCleaner
from utils import Resampler, get_data_path_list, length_to_mask, log_norm, maximum_path
from Utils.PLBERT.util import load_plbert

warnings.simplefilter("ignore")

# Disable TF32 computations for cuDNN
torch.backends.cudnn.allow_tf32 = False


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="StyleTTS2 stage 2 training")
    parser.add_argument("config_path", type=str, help="path to config")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="number of workers")
    add_logging_args(parser)  # --log_level, --log_file
    args = parser.parse_args()

    # Load config
    with open(args.config_path, encoding="utf-8") as fr:
        config = munchify(yaml.safe_load(fr))
    cfg_name, cfg_ext = osp.splitext(osp.basename(args.config_path))

    # Set up logging
    set_random_seed(config.seed)
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    # Unified logging (console + file). Without Accelerate => logs this process.
    log_file = args.log_file or osp.join(log_dir, "train.log")
    setup_logging(args.log_level, log_file)
    logger = get_logger(__name__)

    wb_logger = wandb.init(
        # Set the wandb project where this run will be logged.
        project="StyleTTS2+spkenc",
        # Set run name
        name=f"{osp.basename(log_dir)}",
        # Track hyperparameters and run metadata.
        config=config,
        dir=log_dir,
    )

    # Init NVLM
    nvidia_smi.nvmlInit()
    n_gpus = nvidia_smi.nvmlDeviceGetCount()
    max_vram = 0  # Track maximum VRAM usage
    # Get total VRAM of the first GPU
    total_vram = (
        nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(0)).total >> 30
    )
    logger.info("NVLM initialized")

    # Set up training parameters
    batch_size = config.get("batch_size", 10)
    max_len = config.get("max_len", 200)
    log_interval = config.get("log_interval", 10)
    saving_epoch = config.get("save_freq", 2)
    max_saved_models = config.get("max_saved_models", 2)
    save_milestones = config.get("save_milestones", False)
    grad_clip = config.get("grad_clip", None)  # JMa: gradient clipping support
    device = config.get("cuda", "cuda")  # Set to cuda

    # Set up epochs
    epochs = config["epochs"].get("stage2", 100)
    diff_epoch = config["epochs"].get("diff", 20)
    joint_epoch = config["epochs"].get("joint", 50)

    # Set up data parameters
    data_params = config.get("data_params", None)
    sr = config["preprocess_params"].get("sr", 24000)
    hop_length = config["preprocess_params"]["spect_params"].get("hop_length", 300)
    silence_beg = config["preprocess_params"].get("silence_beg", 4800)
    silence_end = config["preprocess_params"].get("silence_end", 4800)
    train_path = data_params["train_data"]
    val_path = data_params["val_data"]
    root_path = data_params["root_path"]
    ood_data = data_params["OOD_data"]
    save_val_audio = data_params.get("save_val_audio", False)
    n_val_audios = config["data_params"].get("n_val_audios", 3)
    save_test_audio = data_params.get("save_test_audio", False)
    test_audio_dir = os.path.join(
        config["log_dir"],
        config["data_params"].get("test_audio_dir", "test_audios"),
    )
    # Set up test sentences
    test_sentences = data_params.get("test_sentences", [])
    logger.debug("Test sentences: %s", test_sentences)

    # Shortcuts to access parameters
    model_params = config.model_params
    loss_params = config.loss_params
    optimizer_params = config.optimizer_params
    spkenc_params = config.model_params.spkenc_params

    # Optional SCL: disabled when lambda_scl is absent or <= 0
    use_scl = bool(getattr(loss_params, "lambda_scl", 0.0))

    # Optional fine-tuning for speaker encoder
    # - freeze: if False, allow training
    # - unfreeze_epoch: set to joint start
    spkenc_unfreeze_epoch = joint_epoch

    # Set up text cleaner
    text_cleaner = TextCleaner(data_params.symbol_dict_path, pad=data_params.pad)
    logger.debug("Number of symbols: %d", len(text_cleaner))
    assert len(text_cleaner) == 81, f"Number of symbols must be 81 but it is {len(text_cleaner)}"

    # Load pretrained utility models
    # Load ASR model
    asr_config = config.get("ASR_config", False)
    asr_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(asr_path, asr_config)
    # Load pretrained F0 model
    f0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(f0_path)
    # Load PL-BERT model
    bert_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(bert_path)
    # Load speaker encoder model
    speaker_encoder = load_spkenc_model(spkenc_params.model, spkenc_params.freeze)

    # Build model
    model = StyleTTS2(model_params, text_aligner, pitch_extractor, plbert, speaker_encoder)

    # Load data & dataloaders
    train_list, val_list = get_data_path_list(train_path, val_path)

    dataset_config = {
        "sr": sr,
        "min_length": data_params.min_length,
        "max_length": model.bert.config.max_position_embeddings,  # ALBERT config
        "silence_beg": silence_beg,
        "silence_end": silence_end,
        "n_mels": config.model_params.n_mels,
        "spect_params": config.preprocess_params.spect_params,
        "max_ref_mel_length": config.preprocess_params.max_ref_mel_length,
        "use_ref_sample": True,
    }

    # Prepare dataloaders
    logger.info("Building training dataloader...")
    train_dataloader = build_dataloader(
        train_list,
        root_path,
        text_cleaner=text_cleaner,
        ood_data=ood_data,
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
        text_cleaner=text_cleaner,
        ood_data=None,  # OOD data not used for validation
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        device=device,
        dataset_config=dataset_config,
        use_speaker_sampler=False,
    )
    wb_logger.summary["n_train_samples"] = len(train_dataloader.dataset)
    wb_logger.summary["n_valid_samples"] = len(val_dataloader.dataset)
    wb_logger.summary["n_ood_texts"] = train_dataloader.dataset.number_ood_texts()

    # Move models to device (cuda)
    model.to(device)

    # DP
    for key in model:
        # speaker_encoder will be left on a single GPU due to scatter/view + in-place issues
        if key not in ("mpd", "msd", "wd", "speaker_encoder"):
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    # Total number of steps given the batch size
    steps_per_epoch = len(train_dataloader)

    load_pretrained = config.get("pretrained_model", "") != "" and config.get(
        "second_stage_load_pretrained",
        False,
    )

    if not load_pretrained:
        if config.get("first_stage_path", "") != "":
            first_stage_path = osp.join(log_dir, config.get("first_stage_path", "first_stage.pth"))
            logger.info("Loading the first stage model at %s ...", first_stage_path)
            _, start_epoch, _ = model.load(
                first_stage_path,
                None,
                load_only_params=True,
                # keep starting epoch for tensorboard log
                ignore_modules=[
                    "bert",
                    "bert_encoder",
                    "prosodic_predictor",
                    "msd",
                    "mpd",
                    "wd",
                    "diffusion",
                ],
            )

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            if model_params.style_mix.mode != "external":
                # Inherit the style encoder component from acoustic style encoder to the prosodic style encoder
                # But when we are in external mode, we keep the randomly initialized prosodic style encoder
                model.prosodic_style_encoder = copy.deepcopy(
                    model.acoustic_style_encoder.style_encoder
                )
        else:
            raise ValueError("You need to specify the path to the first stage model.")

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = create_slm_loss(model_params.slm, model.wd, sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        # empirical parameters
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    # Build parameter groups for optimizer
    # Optional per-module optimizer overrides from config
    per_module_overrides = {
        "bert": {
            "lr": config.optimizer_params.bert_lr,
            "betas": (0.9, 0.99),
            "weight_decay": 0.01,
            "initial_lr": config.optimizer_params.bert_lr,
            "min_lr": 0,
            "max_lr": config.optimizer_params.bert_lr * 2,
        },
        "decoder": {
            "lr": config.optimizer_params.ft_lr,
            "betas": (0.0, 0.99),
            "weight_decay": 1e-4,
            "initial_lr": config.optimizer_params.ft_lr,
            "min_lr": 0,
            "max_lr": config.optimizer_params.ft_lr * 2,
        },
        "prosodic_style_encoder": {
            "lr": (
                config.optimizer_params.ft_lr
                if model_params.style_mix.mode != "external"
                else config.optimizer_params.lr
            ),
            "betas": (0.0, 0.99),
            "weight_decay": 1e-4,
            "initial_lr": (
                config.optimizer_params.ft_lr
                if model_params.style_mix.mode != "external"
                else config.optimizer_params.lr
            ),
            "min_lr": 0,
            "max_lr": (
                config.optimizer_params.ft_lr * 2
                if model_params.style_mix.mode != "external"
                else config.optimizer_params.lr * 2
            ),
        },
        "speaker_encoder": {
            "lr": config.optimizer_params.ft_lr,
            "betas": (0.0, 0.99),
            "weight_decay": 1e-4,
            "initial_lr": config.optimizer_params.ft_lr,
            "min_lr": 0,
            "max_lr": config.optimizer_params.ft_lr * 2,
        },
        # Won't work if acoustic_style_encoder is not trainable
        "acoustic_style_encoder": {
            "lr": config.optimizer_params.ft_lr,
            "betas": (0.0, 0.99),
            "weight_decay": 1e-4,
            "initial_lr": config.optimizer_params.ft_lr,
            "min_lr": 0,
            "max_lr": config.optimizer_params.ft_lr * 2,
        },
    }

    # Modules to use double max_lr (typically for 2nd stage training)
    modules = [
        "bert",
        "decoder",
        "prosodic_style_encoder",
        "acoustic_style_encoder",
        "speaker_encoder",
    ]
    parameters_dict, scheduler_params_dict, not_trainable_modules = model.params_for_optimizer(
        epochs,
        steps_per_epoch,
        optimizer_params,
        double_max_lr=modules,
        optimizer_overrides=per_module_overrides,
    )
    logger.info("Optimizer groups: %s", list(parameters_dict.keys()))
    logger.info("Not trainable modules: %s", not_trainable_modules)
    logger.debug("Scheduler parameters: %s", scheduler_params_dict)

    # Create optimizer
    optimizer = build_optimizer(
        parameters_dict,
        scheduler_params_dict,
        optimizer_params.lr,
    )
    logger.debug("Optimizer: %s", optimizer.optimizers)

    # Load models if there is a pre-trained model
    if load_pretrained:
        optimizer, start_epoch, iters = model.load(
            config.pretrained_model,
            optimizer,
            load_only_params=config.get("load_only_params", True),
        )
        # # advance start epoch or we'd re-train and rewrite the last epoch file
        # start_epoch += 1
        logger.info("Loading pre-trained model: %s", config.pretrained_model)
        logger.info("Starting epoch:            %d", start_epoch)
        logger.info("Starting iterations:       %d", iters)
        logger.info("")

    n_down = model.text_aligner.n_down

    best_loss = float("inf")  # best test loss
    # iters = 0  # !!! Should it be resetting?

    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to(device)

    # === Change sigma data calculation ===
    # Working with running values to enable following calculation from already saved model
    running_std = []
    # sigma data mean from already processed epochs stored in config
    inp_sigma_data = float(model_params.diffusion.dist.sigma_data)
    # Count of processed epochs stored
    inp_sigma_count = start_epoch - diff_epoch if start_epoch > diff_epoch else 0

    slmadv_params = config.slmadv_params
    slmadv = (
        SLMAdversarialLoss(
            model,
            wl,
            sampler,
            slmadv_params.min_len,
            slmadv_params.max_len,
            batch_percentage=slmadv_params.batch_percentage,
            skip_update=slmadv_params.iter,
            sig=slmadv_params.sig,
        )
        if slmadv_params.batch_percentage is not None
        else None
    )

    # Create test audio dir under log/eval dir
    if (save_val_audio or save_test_audio) and not os.path.exists(test_audio_dir):
        os.makedirs(test_audio_dir, exist_ok=True)

    # Create resampler for speaker encoder
    spkenc_resampler = (
        Resampler(sr, spkenc_params.sr, device=device) if model.multispeaker else None
    )

    # Create phoneme-to-speech object for synthesizing test sentences
    # - use global noise for speed
    pts = PTS(config, model, use_glob_noise=True)

    logger.info(" > Start training cycles:")
    logger.info(" | > Random seed:        %s", config.seed)
    logger.info(" | > Experiment label:   %s", config.label)
    logger.info(" | > Starting epoch:     %d", start_epoch)
    logger.info(" | > Total epochs:       %d", epochs)
    logger.info(" | > Steps per epoch:    %d", steps_per_epoch)
    logger.info(" | > Input iterations:   %d", iters)
    logger.info(" | > Style mix mode:     %s", model_params.style_mix.mode)
    logger.info(" | > Train data:         %s", data_params.train_data)
    logger.info(" | > Valid data:         %s", data_params.val_data)
    logger.info(" | > Pretrained model:   %s", config.pretrained_model)
    logger.info(" | > Text aligner:       %s", config.ASR_path)
    logger.info(" | > F0 model:           %s", config.F0_path)
    logger.info(" | > PL-BERT:            %s", config.PLBERT_dir)
    logger.info(" | > Batch size:         %d", batch_size)
    logger.info(" | > Max len:            %d", max_len)
    logger.info(" | > Sigma data:         %f", inp_sigma_data)
    logger.info(" | > SLM loss:           %s", model_params.slm.model)
    logger.info(" | > SLM adv training:   %s", slmadv_params.batch_percentage is not None)
    logger.info(" | > SLM min len:        %d", slmadv_params.min_len)
    logger.info(" | > SLM max len:        %d", slmadv_params.max_len)
    logger.info(" | > Spk. embedding dim: %d", model_params.spkenc_params.dim_in)
    logger.info(" | > Acoust style dim:   %d", model_params.spkenc_params.spk_emb_dim)
    logger.info(" | > Pros. style dim:    %d", model_params.style_dim)
    logger.info(" | > Use SCL:            %s", use_scl)
    logger.info(
        " | > SpkEnc freeze:       %s (unfreeze@epoch=%d, lr=%s)",
        spkenc_params.freeze,
        spkenc_unfreeze_epoch,
        config.optimizer_params.ft_lr,
    )
    logger.info("")

    # === Start of training loop ==============================================

    # Train model
    for epoch in range(start_epoch, epochs):
        logger.debug("> ----- Epoch %d/%d -----", epoch + 1, epochs)
        running_loss = 0
        start_time = time.time()
        train_dataloader.batch_sampler.epoch = epoch  # Set epoch for the sampler

        # Set all models to eval mode
        model.set_mode("eval")

        # Models in train mode from the beginning
        train_components = [
            "prosodic_predictor",
            "bert_encoder",
            "bert",
            "prosodic_style_encoder",
        ]

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

        logger.debug(
            "| > Speaker encoder training %s at epoch %d",
            "ENABLED" if spkenc_train_enabled else "DISABLED",
            epoch + 1,
        )

        # Models in train mode based on the epoch
        if epoch >= diff_epoch:
            train_components.extend(["msd", "mpd", "diffusion"])
        if epoch >= joint_epoch:
            train_components.extend(["decoder", "wd"])
            if "acoustic_style_encoder" not in not_trainable_modules:
                train_components.append("acoustic_style_encoder")
            if spkenc_train_enabled:
                train_components.append("speaker_encoder")

        # Set models to train mode
        model.set_mode("train", train_components)

        # Train loop for each epoch
        for batch_idx, batch in enumerate(train_dataloader):
            waves = batch[0]  # Keep ground truth audio
            # Move other batch tensors to device
            batch = [b.to(device) for b in batch[1:]]
            # Keep individual batch tensors
            (
                phonemes,  # Padded input phoneme IDs [B, T_text]
                ph_inp_lens,  # Input phoneme lengths [B]
                ref_phonemes,  # OOD texts
                ref_lens,  # OOD phoneme lengths
                mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                mel_inp_len,  # Mel spectrogram lengths [B]
                ref_waves,  # Reference waveforms
                ref_mels,  # Reference mel spectrograms
            ) = batch
            # Current batch size
            bsize = mel_inp_len.shape[0]

            with torch.no_grad():
                mel_mask = length_to_mask(mel_inp_len // (2**n_down)).to(device)
                ph_mask = length_to_mask(ph_inp_lens).to(phonemes.device)

                try:
                    _, _, d_algn = model.text_aligner(mels, mel_mask, phonemes)
                    d_algn = d_algn.transpose(-1, -2)
                    d_algn = d_algn[..., 1:]
                    d_algn = d_algn.transpose(-1, -2)
                except Exception as e:
                    logger.warning("Error: %s", e)
                    continue  # skip batch

                mask_st = mask_from_lens(d_algn, ph_inp_lens, mel_inp_len // (2**n_down))
                d_algn_mono = maximum_path(d_algn, mask_st)

                # Encode
                h_ph = model.text_encoder(phonemes, ph_inp_lens, ph_mask)
                h_algn = h_ph @ d_algn_mono
                d_gt = d_algn_mono.sum(axis=-1).detach()

                # Compute reference styles
                ref_style = None
                if model.multispeaker and epoch >= diff_epoch:
                    # Vectorized computation for reference styles
                    ref_mels_batch = ref_mels.unsqueeze(1)  # Shape: [B, 1, n_mels, max_ref_len]
                    # Shape: [B, spk_emb_dim]
                    with torch.no_grad():
                        ref_spk_embs = model.speaker_encoder(ref_waves)
                    ref_acoust_style = model.acoustic_style_encoder(
                        ref_mels_batch,
                        spk_emb=ref_spk_embs,
                        warmup_coef=1.0,  # no warmup
                    )
                    ref_pros_style = model.prosodic_style_encoder(ref_mels_batch)
                    ref_style = torch.cat([ref_acoust_style, ref_pros_style], dim=1)

            # --- Compute the style of the entire utterance ---
            # This operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            # ---
            # Initialize global prosodic and acoustic styles
            pros_style = torch.empty(bsize, model_params.style_dim, device=device)
            acoust_style = torch.empty(bsize, model.acoustic_style_encoder.style_dim, device=device)
            for bidx in range(bsize):
                # Extract mel spectrogram for the current sample and unsqueeze
                # to add batch and channel dims
                mels4style = mels[bidx, :, : mel_inp_len[bidx].item()][None, None]

                # Resample ground-truth segments for speaker encoder
                seg4spkenc = spkenc_resampler(waves[bidx].to(device).unsqueeze(0))

                # No grad proposed by Codex
                with torch.no_grad():
                    # Get speaker embeddings for the segment
                    spk_embs = model.speaker_encoder(seg4spkenc)
                    # Compute acoustic and prosodic styles
                    acoust_style[bidx, :] = model.acoustic_style_encoder(
                        mels4style,
                        spk_emb=spk_embs,
                        warmup_coef=1.0,  # no warmup
                    )
                # prosodic style used for prosodic prediction => use grad
                pros_style[bidx, :] = model.prosodic_style_encoder(mels4style)

            # Set ground truth style for denoiser
            target_style = torch.cat([acoust_style, pros_style], dim=-1).detach()

            try:
                # Compute contextualized embeddings from phonetic input
                h_bert = model.bert(phonemes, attention_mask=(~ph_mask).int())
            except RuntimeError as e:
                logger.warning("Error while computing PL-BERT embeddings: %s", e)
                logger.warning("Skipping batch: %d", batch_idx)
                continue  # skip batch

            # Encoded duration information [B, max_len, 768]
            h_bert_en = model.bert_encoder(h_bert).transpose(-1, -2)

            loss_sty, loss_diff = 0.0, 0.0  # Initialize losses that are computed conditionally

            # Denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    # Batch-wise std estimation
                    model.diffusion.module.diffusion.sigma_data = (
                        target_style.std(axis=-1).mean().item()
                    )
                    running_std.append(model.diffusion.module.diffusion.sigma_data)

                if model.multispeaker:
                    pred_style = sampler(
                        noise=torch.randn_like(target_style).unsqueeze(1).to(device),
                        embedding=h_bert,
                        embedding_scale=1,
                        features=ref_style,  # reference from the same speaker as the embedding
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion(
                        target_style.unsqueeze(1),
                        embedding=h_bert,
                        features=ref_style,
                    ).mean()
                else:  # single speaker
                    pred_style = sampler(
                        noise=torch.randn_like(target_style).unsqueeze(1).to(device),
                        embedding=h_bert,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion.module.diffusion(
                        # loss_diff = model.diffusion.diffusion(
                        target_style.unsqueeze(1),
                        embedding=h_bert,
                    ).mean()
                # style reconstruction loss
                loss_sty = F.l1_loss(pred_style, target_style.detach())

            # Predict prosodic features
            d, p_algn = model.prosodic_predictor(
                h_bert_en,
                pros_style,
                ph_inp_lens,
                d_algn_mono,
                ph_mask,
            )

            # --- Pre-allocated Segment Extraction ---

            # Set up maximum lengths based on `max_len` from config
            # TODO: Use max and pad shorter segments?
            mel_len_gt = min(int(mel_inp_len.min().item() / 2 - 1), max_len // 2)
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
            pros_algn = torch.empty(
                bsize,
                p_algn.shape[1],
                mel_len_gt,
                device=device,
                dtype=p_algn.dtype,
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
                pros_algn[bidx] = p_algn[bidx, :, beg_gt : beg_gt + mel_len_gt]
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
            # `en` and `p_en` are not detached as they are used for gradient computation
            mel_gt = mel_gt.detach()
            mel_st = mel_st.detach()
            wav_gt = wav_gt.detach()

            # --- End of Pre-allocated Segment Extraction ---

            # Recompute styles based on the extracted segments
            # Use mel_gt for single speaker, mel_st for multispeaker reference
            style_input_mel = mel_st if model.multispeaker else mel_gt
            # Add channel dim for encoders
            style_input_mel_batch = style_input_mel.unsqueeze(1)

            # Speaker encoding (target for SCL + conditioning for acoustic style)
            seg_st_for_spkenc = spkenc_resampler(wav_st)
            if not spkenc_train_enabled:
                # Frozen: metric only, no gradients to speaker_encoder
                with torch.no_grad():
                    spk_embs_tgt = model.speaker_encoder(seg_st_for_spkenc)
                spk_embs_for_style = spk_embs_tgt
            else:
                spk_embs_st = model.speaker_encoder(seg_st_for_spkenc)
                spk_embs_tgt = spk_embs_st.detach()  # target without gradients
                # allows fine-tuning speaker encoder through style path
                spk_embs_for_style = spk_embs_st

            # Compute styles for the extracted segments
            pros_style = model.prosodic_style_encoder(style_input_mel_batch)
            acoust_style = model.acoustic_style_encoder(
                style_input_mel_batch,
                # Pass a clone to avoid potential in-place modifications affecting SCL target
                # spk_emb=spk_embs_st.clone(),
                # For frozen we use the target directly;
                # for FT we pass the computed embeddings to allow grad flow to spkenc
                spk_emb=spk_embs_for_style,
                warmup_coef=1.0,  # no warmup
            )

            with torch.no_grad():
                # Extract F0 and normalization from the ground truth segment [B, 1, n_mels, mel_len * 2]
                # f0_real, _, f0 = model.pitch_extractor(gt.unsqueeze(1))
                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                # f0 = f0.reshape(f0.shape[0], f0.shape[1] * 2, f0.shape[2], 1).squeeze()
                n_real = log_norm(mel_gt.unsqueeze(1)).squeeze(1)  # [B, n_mels, mel_len * 2]

                # Ground truth waveform segment is already in 'wav_gt'
                y_rec_gt = wav_gt.unsqueeze(1)
                y_rec_gt_pred = model.decoder(ph_algn, f0_real, n_real, acoust_style)

                # Use recording if decoder is tuned (joint training), otherwise use reconstruction
                wav_gt = y_rec_gt if epoch >= joint_epoch else y_rec_gt_pred

            # Predict F0 and Norm using predicted components
            # f0_fake, n_fake = model.prosodic_predictor.F0Ntrain(pros_algn, pros_style)
            f0_fake, n_fake = model.prosodic_predictor(pros_algn, pros_style, compute_f0=True)
            # Reconstruct waveform using predicted F0/Norm
            y_rec = model.decoder(ph_algn, f0_fake, n_fake, acoust_style)

            # Calculate losses using the extracted/generated segments
            loss_f0_rec = (F.smooth_l1_loss(f0_real, f0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

            # --- Discriminator loss ---
            if epoch >= diff_epoch:
                optimizer.zero_grad()
                # Use wav_target (either real segment or reconstructed GT) for discriminator
                loss_disc = dl(wav_gt.detach(), y_rec.detach()).mean()
                loss_disc.backward()
                # JMa: gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.msd.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.mpd.parameters(), grad_clip)
                optimizer.step("msd")
                optimizer.step("mpd")
            else:
                loss_disc = 0

            # --- Generator loss ---
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav_gt)
            loss_gen_all = gl(wav_gt, y_rec).mean() if epoch >= diff_epoch else 0
            loss_lm = wl(wav_gt.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce, loss_dur = 0, 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), ph_inp_lens):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p_algn in range(_s2s_trg.shape[0]):
                    _s2s_trg[p_algn, : _text_input[p_algn]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(
                    _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
                )
                loss_ce += F.binary_cross_entropy_with_logits(
                    _s2s_pred.flatten(), _s2s_trg.flatten()
                )

            loss_ce /= phonemes.size(0)
            loss_dur /= phonemes.size(0)

            # Speaker consistency loss (SCL): compute only if enabled
            if use_scl:
                # gradient flows into y_rec/decoder
                seg_rec_for_spkenc = spkenc_resampler(y_rec.squeeze())
                spk_embs_rec = model.speaker_encoder(seg_rec_for_spkenc)
                loss_scl = 1 - F.cosine_similarity(spk_embs_tgt, spk_embs_rec).mean()
            else:
                loss_scl = 0.0

            loss_gen = (
                loss_params.lambda_mel * loss_mel
                + loss_params.lambda_F0 * loss_f0_rec
                + loss_params.lambda_ce * loss_ce
                + loss_params.lambda_norm * loss_norm_rec
                + loss_params.lambda_dur * loss_dur
                + loss_params.lambda_gen * loss_gen_all
                + loss_params.lambda_slm * loss_lm
                + loss_params.lambda_sty * loss_sty
                + loss_params.lambda_diff * loss_diff
                + loss_params.lambda_scl * loss_scl
            )

            running_loss += loss_mel.item()
            loss_gen.backward()
            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_norm_(model.bert_encoder.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.bert.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.prosodic_predictor.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.prosodic_style_encoder.parameters(), grad_clip)
            if torch.isnan(loss_gen):
                set_trace()

            optimizer.step("bert_encoder")
            optimizer.step("bert")
            optimizer.step("prosodic_predictor")
            optimizer.step("prosodic_style_encoder")

            if epoch >= diff_epoch:
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.diffusion.parameters(), grad_clip)
                optimizer.step("diffusion")

            if epoch >= joint_epoch:
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.prosodic_style_encoder.parameters(), grad_clip)
                    if "acoustic_style_encoder" not in not_trainable_modules:
                        nn.utils.clip_grad_norm_(
                            model.acoustic_style_encoder.parameters(),
                            grad_clip,
                        )
                    if spkenc_train_enabled:
                        nn.utils.clip_grad_norm_(model.speaker_encoder.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
                if "acoustic_style_encoder" not in not_trainable_modules:
                    optimizer.step("acoustic_style_encoder")
                if spkenc_train_enabled:
                    optimizer.step("speaker_encoder")
                optimizer.step("decoder")

                # --- SLM adversarial training ---
                assert (
                    slmadv is None
                ), "SLM adversarial training not supported for external speaker embeddings."

                if slmadv is not None:  # None means no SLM discriminator training
                    # Do SLM discriminator training

                    # randomly pick whether to use in-distribution text
                    use_ind = np.random.rand() < 0.5

                    if use_ind:
                        ref_lens = ph_inp_lens
                        ref_phonemes = phonemes

                    slm_out = slmadv(
                        batch_idx,
                        y_rec_gt,
                        y_rec_gt_pred,
                        waves,
                        mel_inp_len,
                        ref_phonemes,
                        ref_lens,
                        use_ind,
                        target_style.detach(),
                        ref_style if model.multispeaker else None,
                    )

                    if slm_out is None:
                        logger.warning(
                            "SLM discriminator training not performed => skipping batch %d",
                            batch_idx,
                        )
                        # Clean up memory
                        del slm_out, y_rec_gt, y_rec_gt_pred, target_style
                        torch.cuda.empty_cache()
                        continue

                    loss_disc_slm, loss_gen_lm, _ = slm_out

                    # SLM generator loss
                    optimizer.zero_grad()
                    loss_gen_lm.backward()
                    # JMa: gradient clipping
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.bert_encoder.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.bert.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.prosodic_predictor.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.diffusion.parameters(), grad_clip)

                    # compute the gradient norm
                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [
                            p
                            for p in model[key].parameters()
                            if p.grad is not None and p.requires_grad
                        ]
                        for p_algn in parameters:
                            param_norm = p_algn.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    if total_norm["prosodic_predictor"] > slmadv_params.thresh:
                        for key in model.keys():
                            for p_algn in model[key].parameters():
                                if p_algn.grad is not None:
                                    p_algn.grad *= 1 / total_norm["prosodic_predictor"]

                    for p_algn in model.prosodic_predictor.duration_proj.parameters():
                        if p_algn.grad is not None:
                            p_algn.grad *= slmadv_params.scale

                    for p_algn in model.prosodic_predictor.lstm.parameters():
                        if p_algn.grad is not None:
                            p_algn.grad *= slmadv_params.scale

                    for p_algn in model.diffusion.parameters():
                        if p_algn.grad is not None:
                            p_algn.grad *= slmadv_params.scale

                    optimizer.step("bert_encoder")
                    optimizer.step("bert")
                    optimizer.step("prosodic_predictor")
                    optimizer.step("diffusion")

                    # SLM discriminator loss
                    if loss_disc_slm != 0:
                        optimizer.zero_grad()
                        # d_loss_slm.backward(retain_graph=True)
                        loss_disc_slm.backward()
                        # JMa: gradient clipping
                        if grad_clip:
                            nn.utils.clip_grad_norm_(model.wd.parameters(), grad_clip)
                        optimizer.step("wd")
                else:
                    # SLM discriminator training is not used
                    loss_disc_slm, loss_gen_lm = 0, 0  # zero loss if not using SLM
            else:  # epoch < joint_epoch
                loss_disc_slm, loss_gen_lm = 0, 0  # zero loss if not joint training

            # Increment global step counter
            iters += 1

            if (batch_idx + 1) % log_interval == 0:
                loss_mel = running_loss / log_interval
                logger.info(
                    "Epoch [%d/%d], "
                    "Step [%d/%d], "
                    "Mel Loss: %.5f, "
                    "Disc Loss: %.5f, "
                    "Dur Loss: %.5f, "
                    "CE Loss: %.5f, "
                    "Norm Loss: %.5f, "
                    "F0 Loss: %.5f, "
                    "SCL Loss: %.5f, "
                    "LM Loss: %.5f, "
                    "Gen Loss: %.5f, "
                    "Sty Loss: %.5f, "
                    "Diff Loss: %.5f, "
                    "DiscLM Loss: %.5f, "
                    "GenLM Loss: %.5f",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    steps_per_epoch,
                    loss_mel,
                    loss_disc,
                    loss_dur,
                    loss_ce,
                    loss_norm_rec,
                    loss_f0_rec,
                    loss_scl,
                    loss_lm,
                    loss_gen_all,
                    loss_sty,
                    loss_diff,
                    loss_disc_slm,
                    loss_gen_lm,
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
                        "train/mel_loss": loss_mel,
                        "train/gen_loss": loss_gen_all,
                        "train/d_loss": loss_disc,
                        "train/ce_loss": loss_ce,
                        "train/dur_loss": loss_dur,
                        "train/slm_loss": loss_lm,
                        "train/norm_loss": loss_norm_rec,
                        "train/F0_loss": loss_f0_rec,
                        "train/sty_loss": loss_sty,
                        "train/scl_loss": loss_scl,
                        "train/diff_loss": loss_diff,
                        "train/d_loss_slm": loss_disc_slm,
                        "train/gen_loss_slm": loss_gen_lm,
                        "train/curr_vram": curr_vram,
                        "train/max_vram": max_vram,
                        "train/epoch": epoch,
                    },
                    step=iters,
                )

                running_loss = 0
                logger.info(
                    "Max VRAM usage: %d/%d GB (%.2f%%)",
                    max_vram,
                    total_vram,
                    max_vram / total_vram * 100,
                )
                logger.info("Time elapsed: %.2f seconds", time.time() - start_time)

        # === Start of validation part ==============================================

        # Validation
        loss_test, loss_align, loss_f, loss_sim = 0, 0, 0, 0
        # Set all models to eval mode
        model.set_mode("eval")

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                # optimizer.zero_grad()

                try:
                    # Keep ground truth audio
                    waves = batch[0]
                    # Move other batch tensors to device
                    batch = [b.to(device) for b in batch[1:]]
                    # Keep individual batch tensors
                    (
                        phonemes,  # Padded input phoneme IDs [B, T_text]
                        ph_inp_lens,  # Input phoneme lengths [B]
                        ref_phonemes,  # OOD texts
                        ref_lens,  # OOD phoneme lengths
                        mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                        mel_inp_len,  # Mel spectrogram lengths [B]
                        ref_waves,  # Reference waveforms
                        ref_mels,  # Reference mel spectrograms
                    ) = batch
                    # Current batch size
                    bsize = mel_inp_len.shape[0]

                    mel_mask = length_to_mask(mel_inp_len // (2**n_down)).to(device)
                    ph_mask = length_to_mask(ph_inp_lens).to(phonemes.device)

                    _, _, d_algn = model.text_aligner(mels, mel_mask, phonemes)
                    d_algn = d_algn.transpose(-1, -2)
                    d_algn = d_algn[..., 1:]
                    d_algn = d_algn.transpose(-1, -2)

                    mask_st = mask_from_lens(d_algn, ph_inp_lens, mel_inp_len // (2**n_down))
                    d_algn_mono = maximum_path(d_algn, mask_st)

                    # Encode phonemes
                    h_ph = model.text_encoder(phonemes, ph_inp_lens, ph_mask)
                    h_algn = h_ph @ d_algn_mono

                    d_gt = d_algn_mono.sum(axis=-1).detach()

                    # Compute prosodic style for the entire utterance
                    # This operation cannot be done in batch because of the avgpool layer
                    # (may need to work on masked avgpool)
                    pros_style = torch.empty(bsize, model_params.style_dim, device=device)
                    for bidx in range(bsize):
                        mels_ok = mels[bidx, :, : mel_inp_len[bidx]]
                        pros_style[bidx, :] = model.prosodic_style_encoder(
                            mels_ok.unsqueeze(0).unsqueeze(1)
                        )

                    h_bert = model.bert(phonemes, attention_mask=(~ph_mask).int())  # [B, T, 768]
                    h_bert_en = model.bert_encoder(h_bert).transpose(-1, -2)  # [B, 256, T]

                    # Predict duration and pitch [B, 256, T]
                    d, p_algn = model.prosodic_predictor(
                        h_bert_en,
                        pros_style,
                        ph_inp_lens,
                        d_algn_mono,
                        ph_mask,
                    )

                    # --- Pre-allocated Segment Extraction ---
                    # Get clips
                    mel_len_gt = int(mel_inp_len.min().item() / 2 - 1)

                    # Calculate fixed waveform segment length
                    wav_len = (mel_len_gt * 2) * hop_length

                    # Pre-allocate tensors with the calculated fixed length
                    # Note: Style tensor `mel_st` is not used in validation
                    ph_algn = torch.empty(
                        bsize,
                        h_algn.shape[1],
                        mel_len_gt,
                        device=device,
                        dtype=h_algn.dtype,
                    )
                    pros_algn = torch.empty(
                        bsize,
                        p_algn.shape[1],
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
                        # Extract predicted pitch features
                        pros_algn[bidx] = p_algn[bidx, :, beg_gt : beg_gt + mel_len_gt]
                        # Extract ground-truth mel spectrogram and assign to tensor
                        mel_gt[bidx] = mels[bidx, :, (beg_gt * 2) : ((beg_gt + mel_len_gt) * 2)]
                        # Extract corresponding ground-truth audio and assign to tensor
                        beg_idx_wav = (beg_gt * 2) * hop_length
                        end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                        wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                    # # There is no need to detach tensors as in training loop
                    # wav_gt = wav_gt.detach()
                    # mel_gt = mel_gt.detach()
                    # --- End of Pre-allocated Segment Extraction ---

                    # Recompute style using style_encoder for decoder input
                    pros_style = model.prosodic_style_encoder(mel_gt.unsqueeze(1))

                    # Predict F0 and Norm using predicted components
                    # f0_fake, n_fake = model.prosodic_predictor.F0Ntrain(pros_algn, pros_style)
                    f0_fake, n_fake = model.prosodic_predictor(
                        pros_algn,
                        pros_style,
                        compute_f0=True,
                    )

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), ph_inp_lens):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, : _text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(
                            _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
                        )
                    loss_dur /= phonemes.size(0)

                    # Resample ground-truth segments for speaker encoder
                    seg_gt_for_spkenc = spkenc_resampler(wav_gt)
                    # Calculate ground truth speaker embeddings
                    spk_embs_gt = model.speaker_encoder(seg_gt_for_spkenc)

                    # Recompute style using style_encoder for decoder input
                    acoust_style = model.acoustic_style_encoder(
                        mel_gt.unsqueeze(1),
                        spk_emb=spk_embs_gt,
                        warmup_coef=1.0,  # no warmup
                    )

                    y_rec = model.decoder(ph_algn, f0_fake, n_fake, acoust_style)
                    loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())
                    f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                    loss_f0 = F.l1_loss(f0_real, f0_fake) / 10

                    # Speaker Consistency Loss (metric): compute only if enabled
                    spk_embs_tgt = spk_embs_gt  # target speaker embeddings
                    if use_scl:
                        seg_rec_for_spkenc = spkenc_resampler(y_rec.squeeze())
                        spk_embs_rec = model.speaker_encoder(seg_rec_for_spkenc)
                        # cosine similarity (no grads in eval)
                        loss_scl = 1 - F.cosine_similarity(spk_embs_tgt, spk_embs_rec).mean()
                    else:
                        loss_scl = 0.0

                    # Aggregate losses (loss_dur is the mean over valid elements)
                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_f0).mean()
                    loss_sim += loss_scl

                    iters_test += 1

                except Exception as e:
                    logger.error("[!] Error in validation batch %d:\n%s", batch_idx, e)
                    traceback.print_exc()
                    logger.error("n_fake shape: %s", n_fake.shape)
                    logger.error("f0_fake shape: %s", f0_fake.shape)
                    logger.error("f0_real shape: %s", f0_real.shape)
                    logger.error("Skipping batch %d", batch_idx)
                    continue  # Skipping the batch

        # Calculate average losses
        avg_loss_test = loss_test.item() / iters_test
        avg_loss_align = loss_align.item() / iters_test
        avg_loss_f = loss_f.item() / iters_test
        avg_loss_sim = loss_sim / iters_test
        # Update best validation loss
        best_loss = min(avg_loss_test, best_loss)

        # For learnable gate, show values after sigmoid activation
        # For non-learnable gate, show raw values
        gate_values = (
            torch.sigmoid(model.acoustic_style_encoder.gate_param)
            if model.acoustic_style_encoder.learnable_gate
            else model.acoustic_style_encoder.gate_param
        )
        logger.info(
            "Epoch [%3d/%d]: Validation loss: %.3f (best: %.3f), Dur loss: %.3f, F0 loss: %.3f, SCL: %.3f, Gate weights: %.6f±%.6f (%.6f-%.6f)",
            epoch + 1,
            epochs,
            avg_loss_test,
            best_loss,
            avg_loss_align,
            avg_loss_f,
            avg_loss_sim,
            gate_values.mean().item() if model_params.style_mix.mode == "mix" else 0,
            gate_values.std().item() if model_params.style_mix.mode == "mix" else 0,
            gate_values.min().item() if model_params.style_mix.mode == "mix" else 0,
            gate_values.max().item() if model_params.style_mix.mode == "mix" else 0,
        )
        wb_logger.log(
            {
                "eval/mel_loss": avg_loss_test,
                "eval/dur_loss": avg_loss_align,
                "eval/F0_loss": avg_loss_f,
                "eval/scl_loss": avg_loss_sim,
            },
            step=iters,
        )

        # Generate validation samples
        n_val_samples = min(n_val_audios, bsize)
        if epoch < joint_epoch:
            # Generating reconstruction examples with GT duration
            with torch.no_grad():
                # Iterate over the defined number of validation samples
                for idx in range(n_val_samples):
                    mel_len = int(mel_inp_len[idx].item())

                    # Reconstruct audio from ground-truth mel spectrogram,
                    # and extracted and predicted phoneme-audio alignment encoding
                    # TODO: Enable reconstruction from multiple tensors
                    wav_pred = pts.reconstruct(
                        mels[idx, :, :mel_len].unsqueeze(0),  # Ground-truth mel spectrogram
                        # Ground-truth phonemes-audio alignment
                        h_algn[idx, :, : mel_len // 2].unsqueeze(0),
                        spk_emb=spk_embs_gt[idx].unsqueeze(0),
                        # Predicted phonemes-audio alignment encoding
                        p_en=p_algn[idx, :, : mel_len // 2].unsqueeze(0),
                    )

                    # Write and save val audio
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav_pred, os.path.join(test_audio_dir, outfile))

                    # Save ground truth
                    if epoch in (0, diff_epoch, joint_epoch):
                        # wav_gt = np.squeeze(waves[idx].cpu().numpy())
                        wav_gt = waves[idx].squeeze()
                        if save_val_audio and epoch % saving_epoch == 0:
                            outfile = f"epoch_2nd_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))
                        # writer.add_audio(f"gt/y{idx}", wav_gt, epoch, sample_rate=sr)

        else:
            # Generating sampled speech from text directly
            with torch.no_grad():
                ref_style = None

                # --- Vectorized reference style computation ---
                if model.multispeaker and epoch >= diff_epoch:
                    # Take only the first `n_val_samples` samples
                    # Add channel dimension
                    # Shape: [n_val_samples, 1, n_mels, max_len]
                    ref_mels_val = ref_mels[:n_val_samples].unsqueeze(1)
                    ref_spk_embs_val = ref_spk_embs[:n_val_samples]

                    # Call encoders with the entire batch
                    # Shape: [ref_mels_val, style_dim]
                    ref_acoust_style = model.acoustic_style_encoder(
                        ref_mels_val,
                        spk_emb=ref_spk_embs_val,
                        warmup_coef=1.0,  # no warmup
                    )
                    # Shape: [ref_mels_val, style_dim]
                    ref_pros_style = model.prosodic_style_encoder(ref_mels_val)
                    # Combined style [B, 256+512, T]
                    ref_style = torch.cat([ref_acoust_style, ref_pros_style], dim=1)
                # --- End of Vectorized style computation ---

                # Iterate over the defined number of validation samples
                for idx in range(n_val_samples):
                    # Generate audio from phoneme features of the `idx`-th validation file
                    curr_ref_style = ref_style[idx].unsqueeze(0) if ref_style is not None else None
                    # TODO: Enable reconstruction from multiple tensors
                    wav_pred, _ = pts.infer_from_ph_features(
                        ph_inp_lens[idx, ...].unsqueeze(0),
                        ph_mask[idx, : ph_inp_lens[idx]].unsqueeze(0),
                        h_ph[idx, :, : ph_inp_lens[idx]].unsqueeze(0),
                        h_bert[idx].unsqueeze(0),
                        h_bert_en[idx, :, : ph_inp_lens[idx]].unsqueeze(0),
                        ref_s=curr_ref_style,
                    )

                    # Write and save val audio
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav_pred, os.path.join(test_audio_dir, outfile))

                    # Save ground truth
                    if epoch in (0, diff_epoch, joint_epoch):
                        wav_gt = waves[idx].squeeze()
                        if save_val_audio and epoch % saving_epoch == 0:
                            outfile = f"epoch_2nd_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))

        # --- End of validation part ------------------------------------------

        # --- Start of saving part --------------------------------------------

        # Save progress
        if epoch % saving_epoch == 0:
            model.save(
                optimizer,
                epoch,
                iters,
                avg_loss_test,
                "epoch_2nd",
                log_dir,
                max_saved_models,
            )

            # if estimate sigma, save the estimated sigma to the config file
            if epoch >= diff_epoch and model_params.diffusion.dist.estimate_sigma_data:
                sigma_sum = inp_sigma_count * inp_sigma_data + np.sum(running_std)
                sigma_count = inp_sigma_count + len(running_std)
                config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(
                    sigma_sum / sigma_count
                )
                logger.info("Estimated sigma: %f", config.model_params.diffusion.dist.sigma_data)

                # Save config file updated with estimated sigma
                cfg_path = osp.join(log_dir, f"{cfg_name}.processed{cfg_ext}")
                with open(cfg_path, "w", encoding="utf-8") as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)

            # Synthesize test audios to evaluate the model's performance
            # after joint training has started.
            if save_test_audio and epoch >= joint_epoch:
                # Set up number of speakers to test if multispeaker is enabled
                n_speakers = min(3, len(ref_style)) if model.multispeaker else 1
                logger.debug(
                    "Synthesizing %d test sentences for %d speakers",
                    len(test_sentences),
                    n_speakers,
                )
                # Iterate over the defined number of validation test speakers
                for sidx in range(n_speakers):
                    # Generate test sentences for each speaker
                    test_wavs = pts(
                        test_sentences,
                        ref_s=ref_style[sidx].unsqueeze(0) if model.multispeaker else None,
                    )
                    # Save test sentences
                    for widx, w in enumerate(test_wavs):
                        outfile = f"epoch_2nd_{epoch:0>5}_test-{sidx}{widx}.wav"
                        pts.save_wav(w, os.path.join(test_audio_dir, outfile))

        # Save milestone models
        if save_milestones:
            if epoch == diff_epoch - 1:
                model.save(
                    optimizer,
                    epoch,
                    iters,
                    loss_test / iters_test,
                    "stage2_pre-diff",
                    log_dir,
                    use_epoch_in_name=False,
                )
            if epoch == joint_epoch - 1:
                model.save(
                    optimizer,
                    epoch,
                    iters,
                    loss_test / iters_test,
                    "stage2_pre-joint",
                    log_dir,
                    use_epoch_in_name=False,
                )

        # --- End of saving part ----------------------------------------------

    # === End of training loop ================================================

    # === Final model saving ==================================================

    # Save the final checkpoint
    final_filepath = model.save(
        optimizer,
        epoch,
        iters,
        loss_test / iters_test,
        "epoch_2nd",
        log_dir,
        max_saved_models,
    )
    try:
        if epoch > joint_epoch - 1:
            # Create a symlink to the final model
            final_model_symlink = osp.join(log_dir, "second_stage.pth")
            os.symlink(osp.basename(final_filepath), final_model_symlink)
            logger.info("Final second-stage model saved to %s", final_filepath)

            # Reduce the final model size by removing the optimizer state
            del model["mpd"]
            del model["msd"]
            del model["wd"]
            del model["text_aligner"]
            del model["pitch_extractor"]
            if not model.multispeaker:
                del model["acoustic_style_encoder"]
                del model["prosodic_style_encoder"]
            # Save the reduced model
            state_dict = {key: model[key].state_dict() for key in model}
            filepath = osp.join(log_dir, "model4tts.pth")
            torch.save(state_dict, filepath)
            logger.info("Reduced model saved to %s", filepath)

    except FileExistsError:
        logger.warning(
            "Symlink or file %s already exists => %s was not symlinked!",
            final_model_symlink,
            final_filepath,
        )
    except Exception as e:
        logger.error("Error when reducing model: %s", e)

    # if estimate sigma, save the estimated sigma to the config file
    if epoch >= diff_epoch and model_params.diffusion.dist.estimate_sigma_data:
        sigma_sum = inp_sigma_count * inp_sigma_data + np.sum(running_std)
        sigma_count = inp_sigma_count + len(running_std)
        config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(sigma_sum / sigma_count)

        cfg_path = osp.join(log_dir, f"{cfg_name}.processed{cfg_ext}")
        with open(cfg_path, "w", encoding="utf-8") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        logger.info(
            "Estimated sigma: %f", config["model_params"]["diffusion"]["dist"]["sigma_data"]
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
