import argparse
import copy
import os
import os.path as osp
import random
import time
import traceback
import warnings

import numpy as np
import nvidia_smi
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from IPython.core.debugger import set_trace
from monotonic_align import mask_from_lens
from munch import munchify

from logger import add_logging_args, get_logger, setup_logging
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, create_slm_loss
from meldataset import build_dataloader
from models import StyleTTS2, load_ASR_models, load_F0_models
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Modules.pts import PTS
from Modules.slmadv import SLMAdversarialLoss
from optimizers import build_optimizer
from text_utils import TextCleaner
from utils import (
    get_data_path_list,
    h100_fix,
    length_to_mask,
    log_norm,
    maximum_path,
    # nccl_warmup,
    set_random_seed,
)
from Utils.PLBERT.util import load_plbert

warnings.simplefilter("ignore")  # ignore warnings
h100_fix()  # torch.backends.cudnn.allow_tf32 = False  # Disable TF32 computations for cuDNN

acc = Accelerator()  # Initialize the Accelerator


# Simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="StyleTTS2 finetuning")
    parser.add_argument("config_path", type=str, help="path to config")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="number of workers")
    add_logging_args(parser)  # --log-level, --log-file
    args = parser.parse_args()

    # Load config
    with open(args.config_path, encoding="utf-8") as fr:
        cfg = munchify(yaml.safe_load(fr))
    cfg_name, cfg_ext = osp.splitext(osp.basename(args.config_path))

    set_random_seed(cfg.seed)  # set random seed
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Uniform logging (main process only)
    log_file = args.log_file or osp.join(log_dir, "train.log")
    setup_logging(args.log_level, log_file)
    logger = get_logger(__name__)

    # Basic checks
    if cfg.slmadv_params.batch_percentage and not cfg.data_params.OOD_data:
        raise ValueError("OOD data must be provided for SLM adversarial training.")

    # Initialize W&B first (it may add logging handlers); we'll override logging next
    wb_logger = None  # WandB logger
    if acc.is_main_process:
        wb_logger = wandb.init(
            project="StyleTTS2_cs",
            name=f"{osp.basename(log_dir)}",
            config=cfg,
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
    logger.info("NVLM initialized")

    # Set up epochs
    epochs = cfg.epochs.total  # total epochs
    diff_epoch = cfg.epochs.diff  # diffusion epoch
    joint_epoch = cfg.epochs.joint  # joint training epoch

    # Set up test data output
    test_audio_dir = os.path.join(cfg.log_dir, cfg.data_params.test_audio_dir)
    logger.debug("Test sentences: %s", cfg.data_params.test_sentences)

    # Set up text cleaner
    text_cleaner = TextCleaner(cfg.data_params.symbol_dict_path, pad=cfg.data_params.pad)
    logger.debug("Number of symbols: %d", len(text_cleaner))
    assert len(text_cleaner) == 81, f"Number of symbols must be 81 but it is {len(text_cleaner)}"

    # Load pretrained utility models
    with acc.main_process_first():
        text_aligner = load_ASR_models(cfg.ASR_path, cfg.ASR_config)  # pretrained ASR model
        pitch_extractor = load_F0_models(cfg.F0_path)  # pretrained F0 model
        plbert = load_plbert(cfg.PLBERT_dir)  # pretrained phoneme-level BERT

    # Build model
    model = StyleTTS2(cfg.model_params, text_aligner, pitch_extractor, plbert)

    # Load data & dataloaders
    train_list, val_list = get_data_path_list(cfg.data_params.train_data, cfg.data_params.val_data)

    # Initialize training with device
    device = acc.device

    dataset_config = {
        "sr": cfg.preprocess_params.sr,
        "min_length": cfg.data_params.min_length,
        "max_length": model.bert.config.max_position_embeddings,  # ALBERT config
        "silence_beg": cfg.preprocess_params.silence_beg,
        "silence_end": cfg.preprocess_params.silence_end,
        "n_mels": cfg.model_params.n_mels,
        "spect_params": cfg.preprocess_params.spect_params,
        "max_ref_mel_length": cfg.preprocess_params.max_ref_mel_length,
        "use_ref_sample": True,  # TODO: ???
    }

    with acc.main_process_first():
        # Prepare dataloaders
        logger.info("Building training dataloader...")
        train_dataloader = build_dataloader(
            train_list,
            cfg.data_params.root_path,
            text_cleaner,
            validation=False,
            ood_data=cfg.data_params.OOD_data,
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
    if acc.is_main_process:
        wb_logger.summary["n_train_samples"] = len(train_dataloader.dataset)
        wb_logger.summary["n_valid_samples"] = len(val_dataloader.dataset)
        wb_logger.summary["n_ood_texts"] = train_dataloader.dataset.number_ood_texts()

    # Move models to device (cuda)
    model.to(device)

    # DP
    for key in model:
        if key not in ("mpd", "msd", "wd"):
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    load_pretrained = cfg.get("pretrained_model", "") != "" and cfg.get(
        "second_stage_load_pretrained", False
    )

    if not load_pretrained:
        if cfg.get("first_stage_path", "") != "":
            first_stage_path = osp.join(log_dir, cfg.get("first_stage_path", "first_stage.pth"))
            logger.info("Loading the first stage model at %s ...", first_stage_path)
            _, start_epoch, iters = model.load(
                first_stage_path,
                None,
                load_only_params=True,
                # keep starting epoch for tensorboard log
                ignore_modules=[
                    "bert",
                    "bert_encoder",
                    "prosodic_predictor",
                    "prosodic_encoder",
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
            model.predictor_encoder = copy.deepcopy(model.acoustic_style_encoder)
        else:
            raise ValueError("You need to specify the path to the first stage model.")

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = create_slm_loss(cfg.model_params.slm, model.wd, cfg.preprocess_params.sr).to(device)

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

    # Create optimizer
    optimizer = build_optimizer(
        {k: list(model[k].parameters()) for k in model},  # modules to optimize
        cfg.optimizer_params,
    )

    # Adjust optimizers for specific modules
    refined_optim_params = {
        "bert": {
            "lr": cfg.optimizer_params.bert_lr,
            "betas": cfg.optimizer_params.bert_betas,
            "weight_decay": cfg.optimizer_params.bert_weight_decay,
        },
        "ft": {
            "lr": cfg.optimizer_params.ft_lr,
            "betas": cfg.optimizer_params.ft_betas,
            "weight_decay": cfg.optimizer_params.ft_weight_decay,
        },
    }

    optimizer["bert"] = refined_optim_params["bert"]
    optimizer["decoder"] = refined_optim_params["ft"]
    # TODO: Add for finetuning? Pre-trained model already has prosodic style encoder
    # optimizer["prosodic_style_encoder"] = refined_optim_params["ft"]
    optimizer["acoustic_style_encoder"] = refined_optim_params["ft"]

    logger.debug("Optimizer updated:\n%s", optimizer.optimizers)

    # Load models if there is a model
    if load_pretrained:
        optimizer, start_epoch, iters = model.load(
            cfg.pretrained_model,
            optimizer,
            load_only_params=cfg.get("load_only_params", True),
        )
        logger.info("Loading pre-trained model: %s", cfg["pretrained_model"])
        logger.info("Starting epoch:      %d", start_epoch)
        logger.info("Starting iterations: %d", iters)
        logger.info("")

    n_down = model.text_aligner.n_down

    best_loss = float("inf")  # best test loss
    # iters = 0  # !!! Should it be resetting?

    torch.cuda.empty_cache()  # clear cache

    stft_loss = MultiResolutionSTFTLoss().to(device)

    # === Change sigma data calculation ===
    # Working with running values to enable following calculation from already saved model
    running_std = []
    # sigma data mean from already processed epochs stored in config
    inp_sigma_data = float(cfg.model_params.diffusion.dist.sigma_data)
    # Count of processed epochs stored
    inp_sigma_count = start_epoch - diff_epoch if start_epoch > diff_epoch else 0

    # SLM Adversarial Loss
    slmadv_params = cfg.slmadv_params
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
            hop_len=cfg.preprocess_params.spect_params.hop_length,
        )
        if slmadv_params.batch_percentage is not None
        else None
    )

    # Prepare model for training
    model, optimizer, train_dataloader = acc.prepare(model, optimizer, train_dataloader)

    # Create test audio dir under log/eval dir
    if (cfg.data_params.save_val_audio or cfg.data_params.save_test_audio) and not os.path.exists(
        test_audio_dir
    ):
        os.makedirs(test_audio_dir, exist_ok=True)

    # Create phoneme-to-speech object for synthesizing test sentences
    # - use global noise for speed
    pts = PTS(cfg, model, use_glob_noise=True)

    # Total number of steps given the batch size
    steps_per_epoch = len(train_list) // cfg.batch_size

    logger.info(" > Start training cycles:")
    logger.info(" | > Random seed:        %s", cfg.seed)
    logger.info(" | > Experiment label:   %s", cfg.label)
    logger.info(" | > Starting epoch:     %d", start_epoch)
    logger.info(" | > Total epochs:       %d", epochs)
    logger.info(" | > Steps per epoch:    %d", steps_per_epoch)
    logger.info(" | > Input iterations:   %d", iters)
    logger.info(" | > Train data:         %s", cfg.data_params.train_data)
    logger.info(" | > Valid data:         %s", cfg.data_params.val_data)
    logger.info(" | > Pretrained model:   %s", cfg.pretrained_model)
    logger.info(" | > Text aligner:       %s", cfg.ASR_path)
    logger.info(" | > F0 model:           %s", cfg.F0_path)
    logger.info(" | > PL-BERT:            %s", cfg.PLBERT_dir)
    logger.info(" | > Batch size:         %d", cfg.batch_size)
    logger.info(" | > Max len:            %d", cfg.max_len)
    logger.info(" | > Sigma data:         %f", inp_sigma_data)
    logger.info(" | > SLM loss:           %s", cfg.model_params.slm.model)
    logger.info(" | > SLM adv training:   %s", slmadv_params.batch_percentage is not None)
    logger.info(" | > SLM min len:        %d", slmadv_params.min_len)
    logger.info(" | > SLM max len:        %d", slmadv_params.max_len)
    logger.info(" | > Acoust style dim:   %d", cfg.model_params.style_dim)
    logger.info(" | > Pros. style dim:    %d", cfg.model_params.style_dim)
    logger.info("")

    # === Start of training loop ==============================================

    for epoch in range(start_epoch, epochs):
        logger.debug("> ----- Epoch %d/%d -----", epoch + 1, epochs)
        running_loss = 0
        start_time = time.time()
        train_dataloader.batch_sampler.epoch = epoch  # Set epoch for the sampler

        # Set all models to eval mode
        model.set_mode("eval")

        # Models in train mode from the beginning
        train_components = [
            "text_aligner",
            "text_encoder",
            "prosodic_predictor",
            "bert_encoder",
            "bert",
            "msd",
            "mpd",
            # The following components were originally not set to train mode
            # => was it intentional? (They are still updated!)
            # It might be better for finetuning stability to leave them in eval mode
            "acoustic_style_encoder",
            "prosodic_style_encoder",
            "diffusion",
            "decoder",
            "wd",
        ]
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
                ref_mels,  # Reference mel spectrograms
            ) = batch
            # Current batch size
            bsize = mel_inp_len.shape[0]

            with torch.no_grad():
                mel_mask = length_to_mask(mel_inp_len // (2**n_down)).to(device)
                ph_mask = length_to_mask(ph_inp_lens).to(phonemes.device)

                # # --- Original code ---
                # # Compute reference styles
                # ref = None
                # if multispeaker and epoch >= diff_epoch:
                #     ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                #     ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                #     ref = torch.cat([ref_ss, ref_sp], dim=1)
                # # --- End of Original code ---

                # Compute reference styles
                ref_style = None
                if model.multispeaker and epoch >= diff_epoch:
                    # Vectorized computation for reference styles
                    ref_mels_batch = ref_mels.unsqueeze(1)  # Shape: [B, 1, n_mels, max_ref_len]
                    ref_acoust_style = model.acoustic_style_encoder(ref_mels_batch)
                    ref_pros_style = model.prosodic_style_encoder(ref_mels_batch)
                    ref_style = torch.cat([ref_acoust_style, ref_pros_style], dim=1)

            try:
                _, s2s_pred, d_algn = model.text_aligner(mels, mel_mask, phonemes)
                d_algn = d_algn.transpose(-1, -2)
                d_algn = d_algn[..., 1:]
                d_algn = d_algn.transpose(-1, -2)
            except Exception as e:
                logger.warning("Error during text alignment: %s", e)
                continue  # skip batch

            mask_st = mask_from_lens(d_algn, ph_inp_lens, mel_inp_len // (2**n_down))
            d_algn_mono = maximum_path(d_algn, mask_st)

            # encode
            h_ph = model.text_encoder(phonemes, ph_inp_lens, ph_mask)

            # Randomly choose between using the monotonic version or not.
            # This helps in regularizing the model by providing different alignment paths
            # during training.
            h_algn = h_ph @ d_algn if bool(random.getrandbits(1)) else h_ph @ d_algn_mono
            d_gt = d_algn_mono.sum(axis=-1).detach()

            # --- Compute the style of the entire utterance ---
            # This operation cannot be done in batch because of the avgpool layer
            # (may need to work on masked avgpool)
            # ---
            # Initialize global prosodic and acoustic styles
            pros_style = torch.empty(bsize, cfg.model_params.style_dim, device=device)
            acoust_style = torch.empty(bsize, cfg.model_params.style_dim, device=device)
            for bidx in range(bsize):
                # Extract mel spectrogram for the current sample and unsqueeze
                # to add batch and channel dims
                mels4style = mels[bidx, :, : mel_inp_len[bidx].item()][None, None]
                # Compute acoustic and prosodic styles
                acoust_style[bidx, :] = model.acoustic_style_encoder(mels4style)
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
                num_diff_steps = np.random.randint(3, 5)

                if cfg.model_params.diffusion.dist.estimate_sigma_data:
                    # Batch-wise std estimation
                    sigma_data_value = target_style.std(axis=-1).mean().item()
                    model.diffusion.module.diffusion.sigma_data = sigma_data_value
                    running_std.append(sigma_data_value)  # Update running std values

                if model.multispeaker:
                    pred_style = sampler(
                        noise=torch.randn_like(target_style).unsqueeze(1).to(device),
                        embedding=h_bert,
                        embedding_scale=1,
                        features=ref_style,  # reference from the same speaker as the embedding
                        embedding_mask_proba=0.1,
                        num_steps=num_diff_steps,
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion(
                        target_style.unsqueeze(1),
                        embedding=h_bert,
                        features=ref_style,
                    ).mean()
                else:
                    pred_style = sampler(
                        noise=torch.randn_like(target_style).unsqueeze(1).to(device),
                        embedding=h_bert,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_diff_steps,
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion.module.diffusion(
                        target_style.unsqueeze(1),
                        embedding=h_bert,
                    ).mean()
                # Style reconstruction loss
                loss_sty = F.l1_loss(pred_style, target_style.detach())

            dur, p_algn = model.prosodic_predictor(
                h_bert_en,
                pros_style,
                ph_inp_lens,
                d_algn_mono,
                ph_mask,
            )

            # --- Pre-allocated Segment Extraction ---

            # Set up maximum lengths based on `max_len` from config
            mel_len_gt = min(int(mel_inp_len.min().item() / 2 - 1), cfg.max_len // 2)
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

            # Calculate fixed waveform segment length
            wav_len = (mel_len_gt * 2) * cfg.preprocess_params.spect_params.hop_length

            # Pre-allocate tensors with the calculated fixed length
            ph_algn = torch.empty(
                bsize, h_algn.shape[1], mel_len_gt, device=device, dtype=h_algn.dtype
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
                beg_idx_wav = (beg_gt * 2) * cfg.preprocess_params.spect_params.hop_length
                end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                # --- Segment for mel_st ---
                # Style reference (better to be different from the GT)
                beg_st = np.random.randint(0, mel_len - mel_len_st)
                # Extract style reference mel spectrogram for style conditioning
                mel_st[bidx] = mels[bidx, :, (beg_st * 2) : ((beg_st + mel_len_st) * 2)]

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
            # Compute styles for the extracted segments
            pros_style = model.prosodic_style_encoder(style_input_mel_batch)
            acoust_style = model.acoustic_style_encoder(style_input_mel_batch)

            with torch.no_grad():
                # Extract F0 from the ground truth segment [B, 1, n_mels, mel_len * 2]
                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                n_real = log_norm(mel_gt.unsqueeze(1)).squeeze(1)

                # Ground truth waveform segment is already in 'wav_gt'
                y_rec_gt = wav_gt.unsqueeze(1)
                y_rec_gt_pred = model.decoder(ph_algn, f0_real, n_real, acoust_style)

                # Use ground truth from recording since decoder is already well trained
                wav_gt = y_rec_gt

            # Predict F0 and Norm using predicted components
            f0_fake, n_fake = model.prosodic_predictor(pros_algn, pros_style, compute_f0=True)
            # Reconstruct waveform using predicted F0/Norm
            y_rec = model.decoder(ph_algn, f0_fake, n_fake, acoust_style)

            # Calculate losses using the extracted/generated segments
            loss_f0_rec = (F.smooth_l1_loss(f0_real, f0_fake)) / 10  # why /10?
            loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

            # --- Discriminator loss ---
            optimizer.zero_grad()
            loss_disc = dl(wav_gt.detach(), y_rec.detach()).mean()
            acc.backward(loss_disc)
            # JMa: gradient clipping
            if cfg.grad_clip:
                acc.clip_grad_norm_(model.msd.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.mpd.parameters(), cfg.grad_clip)
                # _ = [accelerator.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
            optimizer.step("msd")
            optimizer.step("mpd")

            # --- Generator loss ---
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav_gt)
            loss_gen_all = gl(wav_gt, y_rec).mean()
            loss_lm = wl(wav_gt.detach().squeeze(), y_rec.squeeze()).mean()

            # Duration and alignment losses for phoneme-to-mel mapping
            # For each sample in batch:
            #   - Create target alignment matrix (1 for frames belonging to each phoneme)
            #   - loss_dur: L1 between predicted and GT durations (excluding boundary phonemes)
            #   - loss_ce: BCE between predicted alignment logits and target binary matrix
            loss_ce, loss_dur = 0, 0
            for _s2s_pred, _text_input, _text_length in zip(dur, (d_gt), ph_inp_lens):
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

            # Compute sequence-to-sequence loss for phoneme reconstruction from mel
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, phonemes, ph_inp_lens):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= phonemes.size(0)

            loss_mono = F.l1_loss(d_algn, d_algn_mono) * 10  # why *10?

            loss_gen = (
                cfg.loss_params.lambda_mel * loss_mel
                + cfg.loss_params.lambda_F0 * loss_f0_rec
                + cfg.loss_params.lambda_ce * loss_ce
                + cfg.loss_params.lambda_norm * loss_norm_rec
                + cfg.loss_params.lambda_dur * loss_dur
                + cfg.loss_params.lambda_gen * loss_gen_all
                + cfg.loss_params.lambda_slm * loss_lm
                + cfg.loss_params.lambda_sty * loss_sty
                + cfg.loss_params.lambda_diff * loss_diff
                + cfg.loss_params.lambda_mono * loss_mono
                + cfg.loss_params.lambda_s2s * loss_s2s
            )

            # Accumulate mean mel-spectrogram loss (over all GPUs) across batches for logging
            running_loss += loss_mel.item()
            acc.backward(loss_gen)
            if torch.isnan(loss_gen):
                logger.warning(
                    "NaN detected in generator loss at batch %d, step %d", batch_idx, iters
                )
                set_trace()

            # Gradient clipping
            if cfg.grad_clip:
                acc.clip_grad_norm_(model.bert_encoder.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.bert.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.prosodic_predictor.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.prosodic_style_encoder.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.acoustic_style_encoder.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.decoder.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.text_encoder.parameters(), cfg.grad_clip)
                acc.clip_grad_norm_(model.text_aligner.parameters(), cfg.grad_clip)

            optimizer.step("bert_encoder")
            optimizer.step("bert")
            optimizer.step("prosodic_predictor")
            optimizer.step("prosodic_style_encoder")
            optimizer.step("acoustic_style_encoder")
            optimizer.step("decoder")
            optimizer.step("text_encoder")
            optimizer.step("text_aligner")

            if epoch >= diff_epoch:
                if cfg.grad_clip:
                    acc.clip_grad_norm_(model.diffusion.parameters(), cfg.grad_clip)
                optimizer.step("diffusion")

            loss_disc_slm, loss_gen_lm = 0, 0

            if epoch >= joint_epoch:

                if slmadv is not None:  # None means no SLM discriminator training

                    # Randomly pick whether to use in-distribution text
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

                    if slm_out is not None:
                        loss_disc_slm, loss_gen_lm, _ = slm_out

                        # # Check for NaN in SLM losses
                        # if isinstance(loss_disc_slm, torch.Tensor) and torch.isnan(loss_disc_slm):
                        #     logger.warning(
                        #         "NaN detected in loss_disc_slm at batch %d, iteration %d => skipping SLM training",
                        #         batch_idx, iters
                        #     )
                        #     loss_disc_slm = 0
                        #     loss_gen_lm = 0
                        #     del slm_out, y_rec_gt, y_rec_gt_pred, target_style
                        #     torch.cuda.empty_cache()
                        #     continue

                        # if torch.isnan(loss_gen_lm):
                        #     logger.warning(
                        #         "NaN detected in loss_gen_lm at batch %d, iteration %d => skipping SLM training",
                        #         batch_idx, iters
                        #     )
                        #     loss_disc_slm = 0
                        #     loss_gen_lm = 0
                        #     del slm_out, y_rec_gt, y_rec_gt_pred, target_style
                        #     torch.cuda.empty_cache()
                        #     continue

                        # SLM generator loss
                        optimizer.zero_grad()
                        acc.backward(loss_gen_lm)

                        # JMa: gradient clipping
                        if cfg.grad_clip:
                            acc.clip_grad_norm_(model.bert_encoder.parameters(), cfg.grad_clip)
                            acc.clip_grad_norm_(model.bert.parameters(), cfg.grad_clip)
                            acc.clip_grad_norm_(
                                model.prosodic_predictor.parameters(), cfg.grad_clip
                            )
                            acc.clip_grad_norm_(model.diffusion.parameters(), cfg.grad_clip)

                        # Compute the gradient norm
                        total_norm = {}
                        for name, module in model.items():
                            sq_sum = 0.0
                            for p in module.parameters():
                                if p.grad is not None and p.requires_grad:
                                    param_norm = p.grad.detach().data.norm(2)
                                    sq_sum += param_norm.item() ** 2
                            total_norm[name] = sq_sum**0.5

                        # Gradient scaling
                        if total_norm.get("prosodic_predictor", 0) > slmadv_params.thresh:
                            scale = 1 / total_norm["prosodic_predictor"]
                            for module in model.values():
                                for p in module.parameters():
                                    if p.grad is not None:
                                        p.grad *= scale

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
                            acc.backward(loss_disc_slm)
                            # Gradient clipping
                            if cfg.grad_clip:
                                acc.clip_grad_norm_(model.wd.parameters(), cfg.grad_clip)
                            optimizer.step("wd")

                    else:
                        logger.warning(
                            "SLM discriminator training not performed => skipping batch %d",
                            batch_idx,
                        )
                        # Clean up memory
                        del slm_out, y_rec_gt, y_rec_gt_pred, target_style
                        torch.cuda.empty_cache()
                        continue  # skip batch

            # Increment global step counter
            iters += 1

            # Log training progress
            if (batch_idx + 1) % cfg.log_interval == 0:
                loss_mel = running_loss / cfg.log_interval
                logger.info(
                    "Epoch [%d/%d], "
                    "Step [%d/%d], "
                    "Mel Loss: %.5f, "
                    "Disc Loss: %.5f, "
                    "Dur Loss: %.5f, "
                    "CE Loss: %.5f, "
                    "Norm Loss: %.5f, "
                    "F0 Loss: %.5f, "
                    "LM Loss: %.5f, "
                    "Gen Loss: %.5f, "
                    "Sty Loss: %.5f, "
                    "Diff Loss: %.5f, "
                    "DiscLM Loss: %.5f, "
                    "GenLM Loss: %.5f, "
                    "S2S Loss: %.5f, "
                    "Mono Loss: %.5f",
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
                    loss_lm,
                    loss_gen_all,
                    loss_sty,
                    loss_diff,
                    loss_disc_slm,
                    loss_gen_lm,
                    loss_s2s,
                    loss_mono,
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
                        "train/diff_loss": loss_diff,
                        "train/disc_slm_loss": loss_disc_slm,
                        "train/gen_loss_slm": loss_gen_lm,
                        "train/s2s_loss": loss_s2s,
                        "train/mono_loss": loss_mono,
                        "train/curr_vram": curr_vram,
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

        loss_test, loss_align, loss_f = 0, 0, 0  # Reset loss for validation
        model.set_mode("eval")  # Set all models to eval mode

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]  # Keep ground truth audio
                    batch = [b.to(device) for b in batch[1:]]  # Move other batch tensors to device
                    # Keep individual batch tensors
                    (
                        phonemes,  # Padded input phoneme IDs [B, T_text]
                        ph_inp_lens,  # Input phoneme lengths [B]
                        ref_phonemes,  # OOD texts
                        ref_lens,  # OOD phoneme lengths
                        mels,  # Padded mel spectrograms [B, n_mels, T_mel]
                        mel_inp_len,  # Mel spectrogram lengths [B]
                        ref_mels,  # Reference mel spectrograms
                    ) = batch
                    # Current batch size
                    bsize = mel_inp_len.shape[0]

                    mel_mask = length_to_mask(mel_inp_len // (2**n_down)).to("cuda")
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

                    # # --- Original code ---
                    # ss, gs = [], []
                    # for idx, m in enumerate(mel_input_length):
                    #     mel_length = int(m.item())
                    #     mel = mels[idx, :, :m]
                    #     ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                    #     gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

                    # s = torch.stack(ss).squeeze(dim=1)
                    # # TODO: not used anymore!?
                    # # gs = torch.stack(gs).squeeze(dim=1)
                    # # s_trg = torch.cat([s, gs], dim=-1).detach()
                    # # --- End of Original code ---

                    # Compute prosodic style for the entire utterance
                    # This operation cannot be done in batch because of the avgpool layer
                    # (may need to work on masked avgpool)
                    pros_style = torch.empty(bsize, cfg.model_params.style_dim, device=device)
                    for bidx in range(bsize):
                        mels_ok = mels[bidx, :, : mel_inp_len[bidx]]
                        pros_style[bidx, :] = model.prosodic_style_encoder(
                            mels_ok.unsqueeze(0).unsqueeze(1)
                        )

                    h_bert = model.bert(phonemes, attention_mask=(~ph_mask).int())
                    h_bert_en = model.bert_encoder(h_bert).transpose(-1, -2)

                    # Predict duration and pitch [B, 256, T]
                    dur, p_algn = model.prosodic_predictor(
                        h_bert_en,
                        pros_style,
                        ph_inp_lens,
                        d_algn_mono,
                        ph_mask,
                    )

                    # --- Pre-allocated Segment Extraction ---
                    # Get clips
                    mel_len_gt = int(mel_inp_len.min().item() / 2 - 1)

                    bsize = mel_inp_len.shape[0]  # Use current batch size
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
                        beg_idx_wav = (beg_gt * 2) * cfg.preprocess_params.spect_params.hop_length
                        end_idx_wav = beg_idx_wav + wav_len  # Use pre-calculated length
                        wav_gt[bidx] = waves[bidx][beg_idx_wav:end_idx_wav]

                    # # There is no need to detach tensors as in training loop
                    # wav_gt = wav_gt.detach()
                    # mel_gt = mel_gt.detach()

                    # --- End of Pre-allocated Segment Extraction ---

                    # Recompute style using style_encoder for decoder input
                    pros_style = model.prosodic_style_encoder(mel_gt.unsqueeze(1))

                    # Predict F0 and Norm using predicted components
                    f0_fake, n_fake = model.prosodic_predictor(
                        pros_algn,
                        pros_style,
                        compute_f0=True,
                    )

                    # Compute duration prediction loss for validation
                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(dur, (d_gt), ph_inp_lens):
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

                    # Recompute style using style_encoder for decoder input
                    acoust_style = model.acoustic_style_encoder(mel_gt.unsqueeze(1))

                    y_rec = model.decoder(ph_algn, f0_fake, n_fake, acoust_style)
                    loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())
                    f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                    loss_f0 = F.l1_loss(f0_real, f0_fake) / 10  # why /10?

                    # Aggregate losses (loss_dur is the mean over valid elements)
                    loss_test += loss_mel.mean()
                    loss_align += loss_dur.mean()
                    loss_f += loss_f0.mean()

                    iters_test += 1

                except Exception as e:
                    logger.error("[!] Error in validation batch %d:\n%s", batch_idx, e)
                    traceback.print_exc()
                    logger.error("n_fake shape: %s", n_fake.shape)
                    logger.error("f0_fake shape: %s", f0_fake.shape)
                    logger.error("f0_real shape: %s", f0_real.shape)
                    logger.error("Skipping batch %d", batch_idx)
                    continue  # Skipping the batch

        # Average validation losses
        avg_loss_test = loss_test.item() / iters_test
        avg_loss_align = loss_align.item() / iters_test
        avg_loss_f = loss_f.item() / iters_test
        best_loss = min(avg_loss_test, best_loss)  # Update best validation loss

        logger.info(
            "Validation loss: %.3f (best: %.3f), Dur loss: %.3f, F0 loss: %.3f",
            avg_loss_test,
            best_loss,
            avg_loss_align,
            avg_loss_f,
        )
        wb_logger.log(
            {
                "eval/mel_loss": avg_loss_test,
                "eval/dur_loss": avg_loss_align,
                "eval/F0_loss": avg_loss_f,
            },
            step=iters,
        )
        wb_logger.summary["max_vram"] = max_vram  # Log max VRAM usage per epoch

        # Generate validation samples
        n_val_samples = min(cfg.data_params.n_val_audios, bsize)
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
                        # Predicted phonemes-audio alignment encoding
                        p_en=p_algn[idx, :, : mel_len // 2].unsqueeze(0),
                    )

                    # Write and save val audio
                    if cfg.data_params.save_val_audio and epoch % cfg.save_freq == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav_pred, os.path.join(test_audio_dir, outfile))

                    # Save ground truth
                    if epoch in (0, diff_epoch, joint_epoch):
                        wav_gt = waves[idx].squeeze()
                        if cfg.data_params.save_val_audio and epoch % cfg.save_freq == 0:
                            outfile = f"epoch_2nd_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))

        else:
            # Generating sampled speech from text directly
            with torch.no_grad():
                ref_style = None
                # --- Vectorized style computation ---
                if model.multispeaker and epoch >= diff_epoch:
                    # Take only the first `n_val_samples` samples
                    # Add channel dimension
                    # Shape: [n_val_samples, 1, n_mels, max_len]
                    ref_mels_val = ref_mels[:n_val_samples].unsqueeze(1)

                    # Call encoders with the entire batch
                    # Shape: [ref_mels_val, style_dim]
                    ref_acoust_style = model.acoustic_style_encoder(ref_mels_val)
                    # Shape: [ref_mels_val, style_dim]
                    ref_pros_style = model.prosodic_style_encoder(ref_mels_val)
                    # Combined style [B, 256, T]
                    ref_style = torch.cat([ref_acoust_style, ref_pros_style], dim=1)
                # --- End of Vectorized style computation ---

                # Iterate over the defined number of validation samples
                for idx in range(n_val_samples):
                    # Generate audio from phoneme features of the `idx`-th validation file
                    curr_ref_style = ref_style[idx].unsqueeze(0) if ref_style else None

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
                    if cfg.data_params.save_val_audio and epoch % cfg.save_freq == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav_pred, os.path.join(test_audio_dir, outfile))
                    # Save ground truth
                    if epoch in (0, diff_epoch, joint_epoch):
                        wav_gt = waves[idx].squeeze()
                        if cfg.data_params.save_val_audio and epoch % cfg.save_freq == 0:
                            outfile = f"epoch_2nd_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))

        # --- End of validation part ------------------------------------------

        # --- Start of saving part --------------------------------------------

        # Save progress
        if epoch % cfg.save_freq == 0:
            model.save(
                optimizer,
                epoch,
                iters,
                avg_loss_test,
                "epoch_2nd",
                log_dir,
                cfg.max_saved_models,
            )

            # If estimate sigma, save the estimated sigma to the config file
            if epoch >= diff_epoch and cfg.model_params.diffusion.dist.estimate_sigma_data:
                sigma_sum = inp_sigma_count * inp_sigma_data + np.sum(running_std)
                sigma_count = inp_sigma_count + len(running_std)
                cfg["model_params"]["diffusion"]["dist"]["sigma_data"] = float(
                    sigma_sum / sigma_count
                )
                logger.info("Estimated sigma: %f", cfg.model_params.diffusion.dist.sigma_data)

                # Save config file updated with estimated sigma
                cfg_path = osp.join(log_dir, f"{cfg_name}.processed{cfg_ext}")
                with open(cfg_path, "w", encoding="utf-8") as outfile:
                    yaml.dump(cfg, outfile, default_flow_style=False)

            # Synthesize test audios to evaluate the model's performance
            # after diffusion training has started.
            if cfg.data_params.save_test_audio and epoch >= joint_epoch:
                # Set up number of speakers to test if multispeaker is enabled
                n_speakers = min(3, len(ref_style)) if model.multispeaker else 1
                logger.debug(
                    "Synthesizing %d test sentences for %d speakers",
                    len(cfg.data_params.test_sentences),
                    n_speakers,
                )
                # Iterate over the defined number of validation test speakers
                for sidx in range(n_speakers):
                    # Generate test sentences for each speaker
                    test_wavs = pts(
                        cfg.data_params.test_sentences,
                        ref_s=ref_style[sidx].unsqueeze(0) if model.multispeaker else None,
                    )
                    # Save test sentences
                    for widx, w in enumerate(test_wavs):
                        outfile = f"epoch_2nd_{epoch:0>5}_test-{sidx}{widx}.wav"
                        pts.save_wav(w, os.path.join(test_audio_dir, outfile))

        # Save milestone models
        if cfg.save_milestones:
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
        cfg.max_saved_models,
    )
    try:
        if epoch > joint_epoch - 1:
            # Create a symlink to the final model
            final_model_symlink = osp.join(log_dir, "second_stage.pth")
            os.symlink(osp.basename(final_filepath), final_model_symlink)
            print(f"Final second-stage model saved to {final_filepath}")

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
    if epoch >= diff_epoch and cfg.model_params.diffusion.dist.estimate_sigma_data:
        sigma_sum = inp_sigma_count * inp_sigma_data + np.sum(running_std)
        sigma_count = inp_sigma_count + len(running_std)
        cfg["model_params"]["diffusion"]["dist"]["sigma_data"] = float(sigma_sum / sigma_count)

        cfg_path = osp.join(log_dir, f"{cfg_name}.processed{cfg_ext}")
        with open(cfg_path, "w", encoding="utf-8") as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

        logger.info("Estimated sigma: %f", cfg["model_params"]["diffusion"]["dist"]["sigma_data"])

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
