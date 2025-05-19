import argparse
import copy
import logging
import os
import os.path as osp
import time
import traceback
import warnings

import numpy as np
import nvidia_smi
import torch
import torch.nn.functional as F
import yaml
from IPython.core.debugger import set_trace
from monotonic_align import mask_from_lens
from munch import Munch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, create_slm_loss
from meldataset import build_dataloader
from models import build_model, load_ASR_models, load_checkpoint, load_F0_models, save_checkpoint
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Modules.pts import PTS
from Modules.slmadv import SLMAdversarialLoss
from optimizers import build_optimizer
from text_utils import TextCleaner
from utils import get_data_path_list, length_to_mask, log_norm, maximum_path, recursive_munch
from logger import setup_logging, get_logger
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
    parser.add_argument("-L", "--log_level", type=int, default=logging.INFO, help="log level")
    args = parser.parse_args()

    # Load config
    with open(args.config_path, encoding="utf-8") as fr:
        config = yaml.safe_load(fr)
    cfg_name, cfg_ext = osp.splitext(osp.basename(args.config_path))

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
    writer = SummaryWriter(osp.join(log_dir, "tensorboard"))

    # Init NVLM
    nvidia_smi.nvmlInit()
    n_gpus = nvidia_smi.nvmlDeviceGetCount()
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

    # Set up loss and optimizer parameters
    loss_params = Munch(config["loss_params"])
    optimizer_params = Munch(config["optimizer_params"])

    # Set up text cleaner
    text_cleaner = TextCleaner(data_params["symbol_dict_path"], pad=data_params["pad"])
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

    # Build model
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # Set up single/multi-speaker training
    multispeaker = model_params.multispeaker

    # Load data & dataloaders
    train_list, val_list = get_data_path_list(train_path, val_path)

    logger.info("BERT size: %d", model.bert.config.max_position_embeddings)

    dataset_config = {
        "sr": sr,
        "min_length": data_params["min_length"],
        "max_length": model.bert.config.max_position_embeddings,  # ALBERT config
        "silence_beg": silence_beg,
        "silence_end": silence_end,
        "n_mels": config["model_params"].get("n_mels", 80),
        "spect_params": config["preprocess_params"].get(
            "spect_params",
            {
                "n_fft": 2048,
                "win_length": 1024,
                "hop_length": 300,
            },
        ),
        "use_ref_mel": True,
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
    )

    # Move models to device (cuda)
    _ = [model[key].to(device) for key in model]

    # DP
    for key in model:
        if key not in ("mpd", "msd", "wd"):
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    load_pretrained = config.get("pretrained_model", "") != "" and config.get(
        "second_stage_load_pretrained", False
    )

    if not load_pretrained:
        if config.get("first_stage_path", "") != "":
            first_stage_path = osp.join(log_dir, config.get("first_stage_path", "first_stage.pth"))
            logger.info("Loading the first stage model at %s ...", first_stage_path)
            model, _, start_epoch, _ = load_checkpoint(
                model,
                None,
                first_stage_path,
                load_only_params=True,
                # keep starting epoch for tensorboard log
                ignore_modules=[
                    "bert",
                    "bert_encoder",
                    "predictor",
                    "predictor_encoder",
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
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
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

    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict["bert"]["max_lr"] = optimizer_params.bert_lr * 2
    scheduler_params_dict["decoder"]["max_lr"] = optimizer_params.ft_lr * 2
    scheduler_params_dict["style_encoder"]["max_lr"] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict,
        lr=optimizer_params.lr,
    )

    # adjust BERT learning rate
    for g in optimizer.optimizers["bert"].param_groups:
        g["betas"] = (0.9, 0.99)
        g["lr"] = optimizer_params.bert_lr
        g["initial_lr"] = optimizer_params.bert_lr
        g["min_lr"] = 0
        g["weight_decay"] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g["betas"] = (0.0, 0.99)
            g["lr"] = optimizer_params.ft_lr
            g["initial_lr"] = optimizer_params.ft_lr
            g["min_lr"] = 0
            g["weight_decay"] = 1e-4

    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(
            model,
            optimizer,
            config["pretrained_model"],
            load_only_params=config.get("load_only_params", True),
        )
        # # advance start epoch or we'd re-train and rewrite the last epoch file
        # start_epoch += 1
        logger.info("Loading pre-trained model: %s", config["pretrained_model"])
        logger.info("Starting epoch:            %d", start_epoch)
        logger.info("Starting iterations:       %d", iters)
        logger.info("")

    n_down = model.text_aligner.n_down

    best_loss = float("inf")  # best test loss
    # iters = 0  # !!! Should it be resetting?

    # criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to(device)

    # === Change sigma data calculation ===
    # Working with running values to enable following calculation from already saved model
    running_std = []
    # sigma data mean from already processed epochs stored in config
    inp_sigma_data = float(model_params.diffusion.dist.sigma_data)
    # Count of processed epochs stored
    inp_sigma_count = start_epoch - diff_epoch if start_epoch > diff_epoch else 0

    slmadv_params = Munch(config["slmadv_params"])
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

    # Create phoneme-to-speech object for synthesizing test sentences
    # - use global noise for speed
    pts = PTS(config, model, use_glob_noise=True)

    # Total number of steps given the batch size
    tot_num_steps = len(train_list) // batch_size

    logger.info(" > Start training cycles:")
    logger.info(" | > Starting epoch:   %d", start_epoch)
    logger.info(" | > Total epochs:     %d", epochs)
    logger.info(" | > Steps per epoch:  %d", tot_num_steps)
    logger.info(" | > Input iterations: %d", iters)
    logger.info(" | > Sigma data:       %f", inp_sigma_data)
    logger.info("")

    # === Start of training loop ==============================================

    # Train model
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        # Set all models to eval mode
        _ = [model[key].eval() for key in model]

        # Set following models to train mode
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()

        for batch_idx, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            (
                spk_embs,
                texts,
                input_lengths,
                ref_texts,
                ref_lengths,
                mels,
                mel_input_length,
                ref_mels,
            ) = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except Exception as e:
                    logger.warning("Error: %s", e)
                    continue  # skip batch

                mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_st)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = t_en @ s2s_attn_mono

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # # Compute reference styles
                # ref = None
                # if multispeaker and epoch >= diff_epoch:
                #     ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                #     ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                #     ref = torch.cat([ref_ss, ref_sp], dim=1)

                # Compute reference styles
                ref = None
                if multispeaker and epoch >= diff_epoch:
                    # Vectorized computation for reference styles
                    ref_mels_batch = ref_mels.unsqueeze(1)  # Shape: [B, 1, n_mels, max_ref_len]
                    ref_ss = model.style_encoder(ref_mels_batch)
                    ref_sp = model.predictor_encoder(ref_mels_batch)
                    ref = torch.cat([spk_embs, ref_ss, ref_sp], dim=1)

            # # --- Original code ---
            # # compute the style of the entire utterance
            # # this operation cannot be done in batch because of the avgpool layer
            # # (may need to work on masked avgpool)
            # ss, gs = [], []
            # for idx, m in enumerate(mel_input_length):
            #     mel_length = int(m.item())
            #     mel = mels[idx, :, :m]
            #     ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
            #     gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

            # s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            # gs = torch.stack(gs).squeeze()  # global acoustic styles
            # s_trg = torch.cat([gs, s_dur], dim=-1).detach()  # ground truth for denoiser
            # # --- End of Original code ---

            # --- Vectorized computation of styles ---
            # The original comment about avgpool preventing batching was incorrect
            # because AdaptiveAvgPool2d handles variable lengths.

            # Add channel dimension if needed by the encoders
            ref_mels_batch = mels.unsqueeze(1)  # Shape: [B, 1, n_mels, max_len]

            # Call encoders with the entire batch
            # No mask needed due to AdaptiveAvgPool2d in the encoders
            # Global prosodic style [B, style_dim]
            s_dur = model.predictor_encoder(ref_mels_batch)
            # Global acoustic style [B, style_dim]
            gs = model.style_encoder(ref_mels_batch)
            # Set ground truth style for denoiser
            s_trg = torch.cat([spk_embs, gs, s_dur], dim=-1).detach()
            # --- End of Vectorized computation of styles ---

            try:
                # Compute contextualized embeddings from phonetic input
                bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            except RuntimeError as e:
                logger.warning("Error: %s", e)
                # print(f"[!] Error: {e}")
                continue  # skip batch

            # Encoded duration information [B, max_len, 768]
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # Denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    # Batch-wise std estimation
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item()
                    running_std.append(model.diffusion.module.diffusion.sigma_data)

                    # # Sigma data estimation from running values
                    # new_sigma_value = s_trg.std(axis=-1).mean().item()
                    # new_sigma_data = (sigma_data * sigma_count + new_sigma_value) / (
                    #     sigma_count + 1
                    # )
                    # # Update sigma data
                    # model.diffusion.module.diffusion.sigma_data = new_sigma_data
                    # sigma_data = new_sigma_data  # update sigma_data
                    # sigma_count += 1  # increment count

                if multispeaker:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        features=ref,  # reference from the same speaker as the embedding
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur, features=ref
                    ).mean()
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
                else:  # single speaker
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion.module.diffusion(
                        # loss_diff = model.diffusion.diffusion(
                        s_trg.unsqueeze(1),
                        embedding=bert_dur,
                    ).mean()
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty, loss_diff = 0, 0

            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

            # # --- Vectorized Segment Extraction ---
            # batch_size = mels.size(0)
            # # device = mels.device

            # # Calculate segment lengths, ensuring they are at least 1
            # min_mel_input_len_half = int(mel_input_length.min().item() / 2)
            # mel_len = max(1, min(min_mel_input_len_half - 1, max_len // 2))
            # mel_len_st = max(1, min_mel_input_len_half - 1)

            # # Calculate maximum possible start indices for each item in the batch
            # mel_lengths_half = mel_input_length // 2
            # max_starts1 = torch.clamp(mel_lengths_half - mel_len, min=0)
            # max_starts2 = torch.clamp(mel_lengths_half - mel_len_st, min=0)

            # # Generate random start indices for the batch (scaled from uniform random numbers)
            # # Adding a small epsilon to prevent issues with max_starts being 0
            # rand_starts1_uniform = torch.rand(batch_size, device=device)
            # random_starts1 = (rand_starts1_uniform * (max_starts1.float() + 1 - 1e-6)).long()

            # rand_starts2_uniform = torch.rand(batch_size, device=device)
            # random_starts2 = (rand_starts2_uniform * (max_starts2.float() + 1 - 1e-6)).long()

            # # --- Prepare indices for gathering ---

            # # Indices for en (length mel_len) - Keep using asr shape
            # idx_range_en = torch.arange(mel_len, device=device).unsqueeze(0)  # [1, mel_len]
            # indices_en_asr = random_starts1.unsqueeze(1) + idx_range_en  # [B, mel_len]
            # indices_en_expanded_asr = indices_en_asr.unsqueeze(1).expand(
            #     -1, asr.shape[1], -1
            # )  # [B, C_asr, mel_len]

            # # Indices for p_en (length mel_len) - Use p shape
            # # idx_range_en is the same
            # indices_en_expanded_p = indices_en_asr.unsqueeze(1).expand(  # Reuse indices_en_asr
            #     -1, p.shape[1], -1
            # )  # [B, C_p, mel_len]

            # # Indices for gt (length mel_len * 2)
            # idx_range_gt = torch.arange(mel_len * 2, device=device).unsqueeze(0)  # [1, mel_len * 2]
            # start_offset_gt = random_starts1 * 2  # [B]
            # indices_gt = start_offset_gt.unsqueeze(1) + idx_range_gt  # [B, mel_len * 2]
            # # Expand for mel dimension [B, n_mels, mel_len * 2]
            # indices_gt_expanded = indices_gt.unsqueeze(1).expand(-1, mels.shape[1], -1)

            # # Indices for st (length mel_len_st * 2) [1, mel_len_st * 2]
            # idx_range_st = torch.arange(mel_len_st * 2, device=device).unsqueeze(0)
            # start_offset_st = random_starts2 * 2  # [B]
            # indices_st = start_offset_st.unsqueeze(1) + idx_range_st  # [B, mel_len_st * 2]
            # # Expand for mel dimension
            # indices_st_expanded = indices_st.unsqueeze(1).expand(
            #     -1, mels.shape[1], -1
            # )  # [B, n_mels, mel_len_st * 2]

            # # --- Gather segments ---
            # # Check if dimensions match before gathering
            # # Ensure indices do not go out of bounds (clamp if necessary, though random generation should handle it)
            # # Clamping indices just in case of edge issues
            # indices_en_expanded_asr = torch.clamp(indices_en_expanded_asr, 0, asr.shape[2] - 1)
            # indices_en_expanded_p = torch.clamp(
            #     indices_en_expanded_p, 0, p.shape[2] - 1
            # )  # Clamp based on p length
            # indices_gt_expanded = torch.clamp(indices_gt_expanded, 0, mels.shape[2] - 1)
            # indices_st_expanded = torch.clamp(indices_st_expanded, 0, mels.shape[2] - 1)

            # en = torch.gather(asr, 2, indices_en_expanded_asr)  # Use ASR indices
            # p_en = torch.gather(p, 2, indices_en_expanded_p)  # Use P indices
            # mel_gt = torch.gather(mels, 2, indices_gt_expanded).detach()
            # mel_st = torch.gather(mels, 2, indices_st_expanded).detach()

            # # --- Waveform segment extraction (kept as loop due to 'waves' being a list) ---
            # wav_gt = []
            # wav_indices_start = random_starts1 * 2 * hop_length
            # wav_indices_end = (random_starts1 + mel_len) * 2 * hop_length
            # for idx, w in enumerate(waves):
            #     start_idx = wav_indices_start[idx].item()
            #     end_idx = wav_indices_end[idx].item()
            #     # Ensure indices are within bounds for the specific waveform
            #     start_idx = max(0, start_idx)
            #     end_idx = min(len(w), end_idx)
            #     if start_idx >= end_idx:
            #         # Handle cases where segment length becomes zero or negative
            #         # Append a zero tensor of expected type/device or handle differently
            #         # For simplicity, appending a small zero tensor. Adjust if needed.
            #         wav_gt.append(torch.zeros(1, dtype=torch.float, device=device))
            #         logger.warning(
            #             "Wave segment for index %d has zero or negative length. Appending zero.",
            #             idx,
            #         )
            #     else:
            #         y = w[start_idx:end_idx].to(device)  # Extract segment
            #         wav_gt.append(y.to(device).float())

            # # Pad waveform segments to the same length for stacking
            # # Find max length among extracted segments
            # max_wav_len = max(w_seg.shape[0] for w_seg in wav_gt)
            # # Pad and stack
            # wav_padded = []
            # for w_seg in wav_gt:
            #     pad_len = max_wav_len - w_seg.shape[0]
            #     if pad_len > 0:
            #         # Pad on the right with zeros
            #         padded_seg = F.pad(w_seg, (0, pad_len))
            #         wav_padded.append(padded_seg)
            #     else:
            #         wav_padded.append(w_seg)

            # wav_gt = torch.stack(wav_padded).float().detach()  # [B, max_segment_wav_len]

            # # --- End of Vectorized Segment Extraction ---

            # # Original non-vectorized loop (commented out) ---
            # # Set up maximum lengths based on `max_len` from config
            # # TODO: Use max and pad shorter segments?
            # mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            # mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            # en, mel_gt, mel_st, p_en, wav_gt = [], [], [], [], []

            # # Pick random segments from the batch
            # for idx, (m, w) in enumerate(zip(mel_input_length, waves)):
            #     mel_length = int(m.item() / 2)
            #     random_start = np.random.randint(0, mel_length - mel_len)
            #     en.append(asr[idx, :, random_start : random_start + mel_len])
            #     p_en.append(p[idx, :, random_start : random_start + mel_len])
            #     # Random melspetrogram segment up to `max_len`
            #     mel_gt.append(mels[idx, :, (random_start * 2) : ((random_start + mel_len) * 2)])
            #     # Random waveform segment up to `max_len` (300 is hop size)
            #     y = w[(random_start * 2) * hop_length : ((random_start + mel_len) * 2) * hop_length]
            #     # wav_gt.append(torch.from_numpy(y).to(device))
            #     wav_gt.append(y.to(device).float())
            #     # style reference (better to be different from the GT)
            #     random_start = np.random.randint(0, mel_length - mel_len_st)
            #     mel_st.append(mels[idx, :, (random_start * 2) : ((random_start + mel_len_st) * 2)])

            # wav_gt = torch.stack(wav_gt).float().detach()
            # en = torch.stack(en)
            # p_en = torch.stack(p_en)
            # mel_gt = torch.stack(mel_gt).detach()
            # mel_st = torch.stack(mel_st).detach()

            # --- End of Original non-vectorized loop ---

            # --- Pre-allocated Segment Extraction ---

            # Set up maximum lengths based on `max_len` from config
            # TODO: Use max and pad shorter segments?
            mel_len_gt = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)

            bsize = mel_input_length.shape[0]  # Use current batch size
            wav_len = (mel_len_gt * 2) * hop_length  # Calculate fixed waveform segment length

            # Pre-allocate tensors with the calculated fixed length
            en = torch.empty(bsize, asr.shape[1], mel_len_gt, device=device, dtype=asr.dtype)
            p_en = torch.empty(bsize, p.shape[1], mel_len_gt, device=device, dtype=p.dtype)
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
                p_en[bidx] = p[bidx, :, beg_gt : beg_gt + mel_len_gt]
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
            # `en` and `p_en` are not detached as they are used for gradient computation
            mel_gt = mel_gt.detach()
            mel_st = mel_st.detach()
            wav_gt = wav_gt.detach()

            # --- End of Pre-allocated Segment Extraction ---

            # Check if extracted tensors are too short or empty
            if mel_gt.size(-1) < 80:
                logger.warning(
                    "Segment is too short => skipping batch %d (gt: %d, wav: %d).",
                    batch_idx,
                    mel_gt.size(-1),
                    wav_gt.size(-1),
                )
                continue

            # Recompute styles based on the extracted segments
            # Use mel_gt for single speaker, mel_st for multispeaker reference
            style_input_mel = mel_st if multispeaker else mel_gt
            # Add channel dim for encoders
            style_input_mel_batch = style_input_mel.unsqueeze(1)
            # Compute styles for the extracted segments
            s_dur = model.predictor_encoder(style_input_mel_batch)
            s = model.style_encoder(style_input_mel_batch)
            s = torch.cat([spk_embs, s], dim=1)

            with torch.no_grad():
                # Extract F0 and normalization from the ground truth segment [B, 1, n_mels, mel_len * 2]
                # f0_real, _, f0 = model.pitch_extractor(gt.unsqueeze(1))
                f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                # f0 = f0.reshape(f0.shape[0], f0.shape[1] * 2, f0.shape[2], 1).squeeze()
                n_real = log_norm(mel_gt.unsqueeze(1)).squeeze(1)  # [B, n_mels, mel_len * 2]

                # Ground truth waveform segment is already in 'wav_gt'
                y_rec_gt = wav_gt.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, f0_real, n_real, s)

                # Use recording if decoder is tuned (joint training), otherwise use reconstruction
                wav_gt = y_rec_gt if epoch >= joint_epoch else y_rec_gt_pred

            # Predict F0 and Norm using predicted components
            f0_fake, n_fake = model.predictor.F0Ntrain(p_en, s_dur)
            # Reconstruct waveform using predicted F0/Norm
            y_rec = model.decoder(en, f0_fake, n_fake, s)

            # Calculate losses using the extracted/generated segments
            loss_f0_rec = (F.smooth_l1_loss(f0_real, f0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

            # --- Discriminator loss ---
            if epoch >= diff_epoch:
                optimizer.zero_grad()
                # Use wav_target (either real segment or reconstructed GT) for discriminator
                d_loss = dl(wav_gt.detach(), y_rec.detach()).mean()
                d_loss.backward()
                # JMa: gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.msd.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.mpd.parameters(), grad_clip)
                optimizer.step("msd")
                optimizer.step("mpd")
            else:
                d_loss = 0

            # --- Generator loss ---
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav_gt)
            loss_gen_all = gl(wav_gt, y_rec).mean() if epoch >= diff_epoch else 0
            loss_lm = wl(wav_gt.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce, loss_dur = 0, 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, : _text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(
                    _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
                )
                loss_ce += F.binary_cross_entropy_with_logits(
                    _s2s_pred.flatten(), _s2s_trg.flatten()
                )

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = (
                loss_params.lambda_mel * loss_mel
                + loss_params.lambda_F0 * loss_f0_rec
                + loss_params.lambda_ce * loss_ce
                + loss_params.lambda_norm * loss_norm_rec
                + loss_params.lambda_dur * loss_dur
                + loss_params.lambda_gen * loss_gen_all
                + loss_params.lambda_slm * loss_lm
                + loss_params.lambda_sty * loss_sty
                + loss_params.lambda_diff * loss_diff
            )

            running_loss += loss_mel.item()
            g_loss.backward()
            # JMa: gradient clipping
            if grad_clip:
                # _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                nn.utils.clip_grad_norm_(model.bert_encoder.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.bert.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.predictor.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.predictor_encoder.parameters(), grad_clip)
            if torch.isnan(g_loss):
                set_trace()

            optimizer.step("bert_encoder")
            optimizer.step("bert")
            optimizer.step("predictor")
            optimizer.step("predictor_encoder")

            if epoch >= diff_epoch:
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.diffusion.parameters(), grad_clip)
                optimizer.step("diffusion")

            if epoch >= joint_epoch:
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.style_encoder.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
                optimizer.step("style_encoder")
                optimizer.step("decoder")

                if slmadv is not None:  # None means no SLM discriminator training
                    # Do SLM discriminator training

                    # randomly pick whether to use in-distribution text
                    use_ind = np.random.rand() < 0.5

                    if use_ind:
                        ref_lengths = input_lengths
                        ref_texts = texts

                    slm_out = slmadv(
                        batch_idx,
                        y_rec_gt,
                        y_rec_gt_pred,
                        waves,
                        mel_input_length,
                        ref_texts,
                        ref_lengths,
                        use_ind,
                        s_trg.detach(),
                        ref if multispeaker else None,
                    )

                    if slm_out is None:
                        logger.warning(
                            "SLM discriminator training not performed => skipping batch %d",
                            batch_idx,
                        )
                        # Clean up memory
                        del slm_out, y_rec_gt, y_rec_gt_pred, s_trg
                        torch.cuda.empty_cache()
                        continue

                    d_loss_slm, loss_gen_lm, _ = slm_out

                    # SLM generator loss
                    optimizer.zero_grad()
                    loss_gen_lm.backward()
                    # JMa: gradient clipping
                    if grad_clip:
                        # _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                        nn.utils.clip_grad_norm_(model.bert_encoder.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.bert.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.predictor.parameters(), grad_clip)
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
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    if total_norm["predictor"] > slmadv_params.thresh:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= 1 / total_norm["predictor"]

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    optimizer.step("bert_encoder")
                    optimizer.step("bert")
                    optimizer.step("predictor")
                    optimizer.step("diffusion")

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        # d_loss_slm.backward(retain_graph=True)
                        d_loss_slm.backward()
                        # JMa: gradient clipping
                        if grad_clip:
                            nn.utils.clip_grad_norm_(model.wd.parameters(), grad_clip)
                        optimizer.step("wd")
                else:
                    # SLM discriminator training is not used
                    d_loss_slm, loss_gen_lm = 0, 0  # zero loss if not using SLM

            else:  # epoch < joint_epoch
                d_loss_slm, loss_gen_lm = 0, 0  # zero loss if not using SLM

            iters += 1

            if (batch_idx + 1) % log_interval == 0:
                mel_loss = running_loss / log_interval
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
                    "GenLM Loss: %.5f",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    tot_num_steps,
                    mel_loss,
                    d_loss,
                    loss_dur,
                    loss_ce,
                    loss_norm_rec,
                    loss_f0_rec,
                    loss_lm,
                    loss_gen_all,
                    loss_sty,
                    loss_diff,
                    d_loss_slm,
                    loss_gen_lm,
                )
                writer.add_scalar("train/mel_loss", mel_loss, iters)
                writer.add_scalar("train/gen_loss", loss_gen_all, iters)
                writer.add_scalar("train/d_loss", d_loss, iters)
                writer.add_scalar("train/ce_loss", loss_ce, iters)
                writer.add_scalar("train/dur_loss", loss_dur, iters)
                writer.add_scalar("train/slm_loss", loss_lm, iters)
                writer.add_scalar("train/norm_loss", loss_norm_rec, iters)
                writer.add_scalar("train/F0_loss", loss_f0_rec, iters)
                writer.add_scalar("train/sty_loss", loss_sty, iters)
                writer.add_scalar("train/diff_loss", loss_diff, iters)
                writer.add_scalar("train/d_loss_slm", d_loss_slm, iters)
                writer.add_scalar("train/gen_loss_slm", loss_gen_lm, iters)

                running_loss = 0
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

        # === Start of validation part ==============================================

        # Validation
        loss_test, loss_align, loss_f = 0, 0, 0
        # Set all models to eval mode
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for _, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    (
                        spk_embs,
                        texts,
                        input_lengths,
                        ref_texts,
                        ref_lengths,
                        mels,
                        mel_input_length,
                        ref_mels,
                    ) = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_st = mask_from_lens(
                            s2s_attn, input_lengths, mel_input_length // (2**n_down)
                        )
                        s2s_attn_mono = maximum_path(s2s_attn, mask_st)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = t_en @ s2s_attn_mono

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    # --- Vectorized style computation ---
                    # Add channel dimension
                    ref_mels_batch = mels.unsqueeze(1)  # Shape: [B, 1, n_mels, max_len]
                    # Call encoders with the entire batch
                    # No mask needed due to AdaptiveAvgPool2d in the encoders
                    s = model.predictor_encoder(ref_mels_batch)  # Shape: [B, style_dim]
                    # gs = model.style_encoder(mels_batch)      # Shape: [B, style_dim]
                    # --- End of vectorized style computation ---

                    # TODO: not used anymore!?
                    # s_trg = torch.cat([s, gs], dim=-1).detach()
                    # --- End of Vectorized style computation ---

                    # Original non-vectorized style computation (commented out)
                    # ss, gs = [], []
                    # for idx, m in enumerate(mel_input_length):
                    #     mel_length = int(m.item())
                    #     mel = mels[idx, :, :m]
                    #     ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                    #     gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

                    # # JMa: Fix: remove explicitly 2nd dimension
                    # # otherwise all dimensions of size 1 are removed
                    # # (resulting in error when current batch size is 1)
                    # # s = torch.stack(ss).squeeze()
                    # # s = torch.stack(ss).squeeze(dim=-1) # - not working
                    # s = torch.stack(ss).squeeze(dim=1)
                    # # # gs = torch.stack(gs).squeeze()              # !!! JMa: not used anymore?
                    # # gs = torch.stack(gs).squeeze(dim=1)        # !!! JMa: not used anymore?
                    # # s_trg = torch.cat([s, gs], dim=-1).detach() # !!! JMa: not used anymore?

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())  # [B, T, 768]
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)  # [B, 256, T]

                    # Predict duration and pitch [B, 256, T]
                    d, p = model.predictor(d_en, s, input_lengths, s2s_attn_mono, text_mask)

                    # # --- Vectorized Segment Extraction for Validation ---
                    # batch_size = mels.size(0)

                    # # Calculate segment length, ensuring it's at least 1
                    # # Use min over the batch to ensure validity for all items
                    # min_mel_input_len_half = int(mel_input_length.min().item() / 2)
                    # # Ensure mel_len is at least 1 and not larger than the smallest possible sequence halved
                    # mel_len_gt = max(1, min_mel_input_len_half - 1)

                    # # Calculate maximum possible start indices for each item in the batch
                    # mel_lengths_half = mel_input_length // 2
                    # max_starts = torch.clamp(mel_lengths_half - mel_len_gt, min=0)

                    # # Generate random start indices for the batch
                    # rand_starts_uniform = torch.rand(batch_size, device=device)
                    # random_starts = (rand_starts_uniform * (max_starts.float() + 1 - 1e-6)).long()

                    # # --- Prepare indices for gathering ---

                    # # Indices for en, p_en (length mel_len)
                    # idx_range_en = torch.arange(mel_len_gt, device=device).unsqueeze(
                    #     0
                    # )  # [1, mel_len]
                    # indices_en = random_starts.unsqueeze(1) + idx_range_en  # [B, mel_len]
                    # # Expand for channel dimension of asr/p
                    # indices_en_expanded = indices_en.unsqueeze(1).expand(
                    #     -1, asr.shape[1], -1
                    # )  # [B, C, mel_len]

                    # # Indices for mel_gt (length mel_len * 2)
                    # idx_range_gt = torch.arange(mel_len_gt * 2, device=device).unsqueeze(
                    #     0
                    # )  # [1, mel_len * 2]
                    # start_offset_gt = random_starts * 2  # [B]
                    # indices_gt = start_offset_gt.unsqueeze(1) + idx_range_gt  # [B, mel_len * 2]
                    # # Expand for mel dimension
                    # indices_gt_expanded = indices_gt.unsqueeze(1).expand(
                    #     -1, mels.shape[1], -1
                    # )  # [B, n_mels, mel_len * 2]

                    # # --- Gather segments ---
                    # # Clamp indices just in case of edge issues
                    # indices_en_expanded = torch.clamp(indices_en_expanded, 0, asr.shape[2] - 1)
                    # indices_gt_expanded = torch.clamp(indices_gt_expanded, 0, mels.shape[2] - 1)

                    # en = torch.gather(asr, 2, indices_en_expanded)
                    # p_en = torch.gather(p, 2, indices_en_expanded)
                    # mel_gt = torch.gather(mels, 2, indices_gt_expanded).detach()

                    # # --- Waveform segment extraction (kept as loop due to 'waves' being a list) ---
                    # wav_gt_list = []
                    # wav_indices_start = random_starts * 2 * hop_length
                    # wav_indices_end = (random_starts + mel_len_gt) * 2 * hop_length
                    # for idx, w in enumerate(waves):
                    #     start_idx = wav_indices_start[idx].item()
                    #     end_idx = wav_indices_end[idx].item()
                    #     # Ensure indices are within bounds for the specific waveform
                    #     start_idx = max(0, start_idx)
                    #     end_idx = min(len(w), end_idx)
                    #     if start_idx >= end_idx:
                    #         wav_gt_list.append(torch.zeros(1, dtype=torch.float, device=device))
                    #         logger.warning(
                    #             "Validation wave segment for index %d has zero or negative length. Appending zero.",
                    #             idx,
                    #         )
                    #     else:
                    #         y = w[start_idx:end_idx].to(device)  # Extract segment
                    #         wav_gt_list.append(y.float())  # Ensure float

                    # # Pad waveform segments to the same length for stacking
                    # max_wav_len = max(w_seg.shape[0] for w_seg in wav_gt_list)
                    # wav_padded = []
                    # for w_seg in wav_gt_list:
                    #     pad_len = max_wav_len - w_seg.shape[0]
                    #     if pad_len > 0:
                    #         padded_seg = F.pad(w_seg, (0, pad_len))
                    #         wav_padded.append(padded_seg)
                    #     else:
                    #         wav_padded.append(w_seg)

                    # wav_gt = torch.stack(wav_padded).float().detach()  # [B, max_segment_wav_len]

                    # # --- End of Vectorized Segment Extraction ---

                    # # Original non-vectorized loop (commented out)
                    # # Get clips
                    # mel_len = int(mel_input_length.min().item() / 2 - 1)

                    # en, mel_gt, p_en, wav_gt = [], [], [], []
                    # for idx, (m, w) in enumerate(zip(mel_input_length, waves)):
                    #     mel_length = int(m.item() / 2)

                    #     random_start = np.random.randint(0, mel_length - mel_len)
                    #     en.append(asr[idx, :, random_start : random_start + mel_len])
                    #     p_en.append(p[idx, :, random_start : random_start + mel_len])

                    #     mel_gt.append(
                    #         mels[idx, :, (random_start * 2) : ((random_start + mel_len) * 2)]
                    #     )
                    #     y = w[(random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300]
                    #     wav_gt.append(torch.from_numpy(y).to(device))

                    # wav_gt = torch.stack(wav_gt).float().detach()  # [B, T]

                    # en = torch.stack(en)  # [B, 256, T]
                    # p_en = torch.stack(p_en)  # [B, 256, T]
                    # mel_gt = torch.stack(mel_gt).detach()

                    # --- Pre-allocated Segment Extraction ---
                    # Get clips
                    mel_len_gt = int(mel_input_length.min().item() / 2 - 1)

                    bsize = mel_input_length.shape[0]  # Use current batch size
                    # Calculate fixed waveform segment length
                    wav_len = (mel_len_gt * 2) * hop_length

                    # Pre-allocate tensors with the calculated fixed length
                    # Note: Style tensor `mel_st` is not used in validation
                    en = torch.empty(
                        bsize, asr.shape[1], mel_len_gt, device=device, dtype=asr.dtype
                    )
                    p_en = torch.empty(
                        bsize, p.shape[1], mel_len_gt, device=device, dtype=asr.dtype
                    )
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
                        # Extract predicted pitch features
                        p_en[bidx] = p[bidx, :, beg_gt : beg_gt + mel_len_gt]
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
                    s = model.predictor_encoder(mel_gt.unsqueeze(1))

                    # Predict F0 and Norm using predicted components
                    f0_fake, n_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, : _text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(
                            _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
                        )
                    loss_dur /= texts.size(0)

                    # Recompute style using style_encoder for decoder input
                    s = model.style_encoder(mel_gt.unsqueeze(1))
                    s = torch.cat([spk_embs, s], dim=1)

                    y_rec = model.decoder(en, f0_fake, n_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav_gt.detach())
                    f0_real, _, _ = model.pitch_extractor(mel_gt.unsqueeze(1))
                    loss_f0 = F.l1_loss(f0_real, f0_fake) / 10

                    # Aggregate losses (loss_dur is the mean over valid elements)
                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_f0).mean()

                    iters_test += 1

                except Exception as e:
                    logger.error("[!] Error: %s", e)
                    traceback.print_exc()
                    continue

        avg_loss_test = loss_test.item() / iters_test
        avg_dur_loss = loss_align.item() / iters_test
        avg_f_loss = loss_f.item() / iters_test
        logger.info(
            "Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f",
            avg_loss_test,
            avg_dur_loss,
            avg_f_loss,
        )
        # print('\n\n\n')
        writer.add_scalar("eval/mel_loss", avg_loss_test, epoch + 1)
        writer.add_scalar("eval/dur_loss", avg_dur_loss, epoch + 1)
        writer.add_scalar("eval/F0_loss", avg_f_loss, epoch + 1)

        # Generate validation samples
        n_val_samples = min(n_val_audios, bsize)
        if epoch < joint_epoch:
            # Generating reconstruction examples with GT duration
            with torch.no_grad():
                # Iterate over the defined number of validation samples
                for idx in range(n_val_samples):
                    mel_length = int(mel_input_length[idx].item())

                    # # Reconstruct audio from ground-truth mel spectrogram and
                    # # phoneme-audio alignment
                    # wav = pts.reconstruct(mel_gt, en_gt, spk_emb=spk_embs[idx].unsqueeze(0))

                    # # Write and save val audio
                    # writer.add_audio(f"eval/y{idx}", wav, epoch, sample_rate=sr)
                    # if save_val_audio and epoch % saving_epoch == 0:
                    #     outfile = f"epoch_2nd_{epoch:0>5}_val-rec-{idx}.wav"
                    #     pts.save_wav(wav, os.path.join(test_audio_dir, outfile))

                    # Reconstruct audio from ground-truth mel spectrogram,
                    # and extracted and predicted phoneme-audio alignment encoding
                    # TODO: Enable reconstruction from multiple tensors
                    wav_pred = pts.reconstruct(
                        mels[idx, :, :mel_length].unsqueeze(0),  # Ground-truth mel spectrogram
                        # Ground-truth phonemes-audio alignment
                        asr[idx, :, : mel_length // 2].unsqueeze(0),
                        spk_embs[idx].unsqueeze(0),
                        # Predicted phonemes-audio alignment encoding
                        p[idx, :, : mel_length // 2].unsqueeze(0),
                    )

                    # Write and save val audio
                    writer.add_audio(f"pred/y{idx}", wav_pred, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav_pred, os.path.join(test_audio_dir, outfile))

                    # Save ground truth
                    if epoch == 0 or multispeaker:
                        # wav_gt = np.squeeze(waves[idx].cpu().numpy())
                        wav_gt = waves[idx].squeeze()
                        if save_val_audio and epoch % saving_epoch == 0:
                            outfile = f"epoch_2nd_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))
                        writer.add_audio(f"gt/y{idx}", wav_gt, epoch, sample_rate=sr)

        else:
            # Generating sampled speech from text directly
            with torch.no_grad():
                ref_s = None
                # # Compute reference styles from ground truth mel spectrogram
                # if multispeaker and epoch >= diff_epoch:
                #     ref_ss = model.style_encoder(ref_mels.unsqueeze(1))  # Timbre style
                #     ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))  # Prosody style
                #     ref_s = torch.cat([ref_ss, ref_sp], dim=1)  # Combined style [B, 256, T]

                # --- Vectorized style computation ---
                if multispeaker and epoch >= diff_epoch:
                    # Take only the first `n_val_samples` samples
                    # Add channel dimension
                    # Shape: [n_val_samples, 1, n_mels, max_len]
                    ref_mels_val = ref_mels[:n_val_samples].unsqueeze(1)
                    spk_embs_val = spk_embs[:n_val_samples]

                    # Call encoders with the entire batch
                    ref_ss = model.style_encoder(ref_mels_val)  # Shape: [ref_mels_val, style_dim]
                    # Shape: [ref_mels_val, style_dim]
                    ref_sp = model.predictor_encoder(ref_mels_val)
                    # Combined style [B, 256+512, T]
                    ref_s = torch.cat([spk_embs_val, ref_ss, ref_sp], dim=1)
                # --- End of Vectorized style computation ---

                # Iterate over the defined number of validation samples
                for idx in range(n_val_samples):
                    # Generate audio from phoneme features of the `idx`-th validation file
                    # TODO: Enable reconstruction from multiple tensors
                    wav_pred, _ = pts.infer_from_ph_features(
                        input_lengths[idx, ...].unsqueeze(0),
                        text_mask[idx, : input_lengths[idx]].unsqueeze(0),
                        t_en[idx, :, : input_lengths[idx]].unsqueeze(0),
                        bert_dur[idx].unsqueeze(0),
                        d_en[idx, :, : input_lengths[idx]].unsqueeze(0),
                        ref_s=ref_s[idx].unsqueeze(0) if multispeaker else None,
                    )

                    # Write and save val audio
                    writer.add_audio(f"pred/y{idx}", wav_pred, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav_pred, os.path.join(test_audio_dir, outfile))

                    # Save ground truth
                    if epoch == 0 or multispeaker:
                        wav_gt = waves[idx].squeeze()
                        if save_val_audio and epoch % saving_epoch == 0:
                            outfile = f"epoch_2nd_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav_gt, os.path.join(test_audio_dir, outfile))
                        writer.add_audio(f"gt/y{idx}", wav_gt, epoch, sample_rate=sr)

        # --- End of validation part ------------------------------------------

        # --- Start of saving part --------------------------------------------

        # Save progress
        if epoch % saving_epoch == 0:
            curr_loss = loss_test.item() / iters_test
            if curr_loss < best_loss:
                best_loss = curr_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                iters,
                curr_loss,
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
                logger.info(
                    "Estimated sigma: %f", config["model_params"]["diffusion"]["dist"]["sigma_data"]
                )

                #     config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(np.mean(running_std))
                #     config["model_params"]["diffusion"]["dist"]["sigma_data"] = sigma_data

                # Save config file updated with estimated sigma
                cfg_path = osp.join(log_dir, f"{cfg_name}.processed{cfg_ext}")
                with open(cfg_path, "w", encoding="utf-8") as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)

            # Synthesize test audios to evaluate the model's performance after diffusion training has started.
            if save_test_audio and epoch >= diff_epoch:
                # Set up number of speakers to test if multispeaker is enabled
                n_speakers = min(3, len(ref_s)) if multispeaker else 1
                # Iterate over the defined number of validation test speakers
                for sidx in range(n_speakers):
                    # Generate test sentences for each speaker
                    test_wavs = pts(
                        test_sentences,
                        ref_s=ref_s[sidx].unsqueeze(0) if multispeaker else None,
                    )
                    # Save test sentences
                    for widx, w in enumerate(test_wavs):
                        outfile = f"epoch_2nd_{epoch:0>5}_test-{sidx}{widx}.wav"
                        pts.save_wav(w, os.path.join(test_audio_dir, outfile))

        # Save milestone models
        if save_milestones:
            if epoch == diff_epoch - 1:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    iters,
                    loss_test / iters_test,
                    "stage2_pre-diff",
                    log_dir,
                    use_epoch_in_name=False,
                )
            if epoch == joint_epoch - 1:
                save_checkpoint(
                    model,
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
    final_filepath = save_checkpoint(
        model,
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
            if not multispeaker:
                del model["style_encoder"]
                del model["predictor_encoder"]
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
    # # if estimate sigma, save the estimated sigma
    # if epoch >= diff_epoch and model_params.diffusion.dist.estimate_sigma_data:
    #     # config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(np.mean(running_std))
    #     config["model_params"]["diffusion"]["dist"]["sigma_data"] = sigma_data
    #     logger.info(
    #         "Estimated sigma: %f", config["model_params"]["diffusion"]["dist"]["sigma_data"]
    #     )

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
    logger.info("NVLM shutdown")


if __name__ == "__main__":
    main()
