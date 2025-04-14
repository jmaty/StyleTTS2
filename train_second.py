# load packages
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
        level=logging.INFO,
        file=osp.join(log_dir, "train.log"),
        formatter_file=formatter_file,
    )
    logger = get_logger(__name__)  # Get a logger
    writer = SummaryWriter(osp.join(log_dir, "tensorboard"))

    # Init NVLM
    nvidia_smi.nvmlInit()
    n_gpus = nvidia_smi.nvmlDeviceGetCount()
    logger.info("NVLM initialized")

    # Set up training parameters
    batch_size = config.get("batch_size", 10)
    epochs = config.get("epochs_2nd", 200)
    log_interval = config.get("log_interval", 10)
    saving_epoch = config.get("save_freq", 2)
    max_saved_models = config.get("max_saved_models", 2)
    save_milestones = config.get("save_milestones", False)
    max_len = config.get("max_len", 200)
    grad_clip = config.get("grad_clip", None)  # JMa: gradient clipping support
    device = config.get("cuda", "cuda")  # Set to cuda

    data_params = config.get("data_params", None)
    sr = config["preprocess_params"].get("sr", 24000)
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
        config["log_dir"], config["data_params"].get("test_audio_dir", "test_audios")
    )

    # # Define pre-processing function and apply to test sentences
    # preprocess_text_fn = add_spaces_around_punctuation
    # print(f"Text pre-processing function: {preprocess_text_fn}")
    # test_sentences = list(map(preprocess_text_fn, data_params.get("test_sentences", [])))
    # print("\n".join(test_sentences))
    test_sentences = data_params.get("test_sentences", [])

    # Set up loss and optimizer parameters
    loss_params = Munch(config["loss_params"])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
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
    }

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

    val_dataloader = build_dataloader(
        val_list,
        root_path,
        text_cleaner=text_cleaner,
        ood_data=ood_data,
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
        logger.info("Starting epoch:      %d", start_epoch)
        logger.info("Starting iterations: %d", iters)
        logger.info("")

    n_down = model.text_aligner.n_down

    best_loss = float("inf")  # best test loss
    # iters = 0  # !!! Should it be resetting?

    # criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to(device)

    # print("BERT", optimizer.optimizers["bert"])
    # print("decoder", optimizer.optimizers["decoder"])

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
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

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

                # Compute reference styles
                ref = None
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer
            # (may need to work on masked avgpool)
            ss, gs = [], []
            for idx, m in enumerate(mel_input_length):
                mel_length = int(m.item())
                mel = mels[idx, :, :m]
                ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze()  # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach()  # ground truth for denoiser

            try:
                bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            except RuntimeError as e:
                logger.warning("Error: %s", e)
                # print(f"[!] Error: {e}")
                continue  # skip batch

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

            # Set up maximum lengths based on `max_len` from config
            # TODO: Use max and pad shorter segments?
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en, gt, st, p_en, wav = [], [], [], [], []

            # Pick random segments from the batch
            for idx, (m, w) in enumerate(zip(mel_input_length, waves)):
                mel_length = int(m.item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[idx, :, random_start : random_start + mel_len])
                p_en.append(p[idx, :, random_start : random_start + mel_len])
                # Random melspetrogram segment up to `max_len`
                gt.append(mels[idx, :, (random_start * 2) : ((random_start + mel_len) * 2)])
                # Random waveform segment up to `max_len` (300 is hop size)
                y = w[(random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[idx, :, (random_start * 2) : ((random_start + mel_len_st) * 2)])

            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            if gt.size(-1) < 80:
                continue

            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))

            with torch.no_grad():
                f0_real, _, f0 = model.pitch_extractor(gt.unsqueeze(1))
                f0 = f0.reshape(f0.shape[0], f0.shape[1] * 2, f0.shape[2], 1).squeeze()
                n_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = model.decoder(en, f0_real, n_real, s)

                if epoch >= joint_epoch:
                    # ground truth from recording => use recording since decoder is tuned
                    wav = y_rec_gt
                else:
                    # ground truth from reconstruction => use reconstruction since decoder is fixed
                    wav = y_rec_gt_pred

            f0_fake, n_fake = model.predictor.F0Ntrain(p_en, s_dur)
            y_rec = model.decoder(en, f0_fake, n_fake, s)

            loss_f0_rec = (F.smooth_l1_loss(f0_real, f0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

            # --- Discriminator loss ---
            if epoch >= diff_epoch:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                d_loss.backward()
                # JMa: gradient clipping
                if grad_clip:
                    # _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                    nn.utils.clip_grad_norm_(model.msd.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.mpd.parameters(), grad_clip)
                optimizer.step("msd")
                optimizer.step("mpd")
            else:
                d_loss = 0

            # --- Generator loss ---
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean() if epoch >= diff_epoch else 0
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

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

                    ss, gs = [], []
                    for idx, m in enumerate(mel_input_length):
                        mel_length = int(m.item())
                        mel = mels[idx, :, :m]
                        ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                        gs.append(model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

                    # JMa: Fix: remove explicitly 2nd dimension
                    # otherwise all dimensions of size 1 are removed
                    # (resulting in error when current batch size is 1)
                    # s = torch.stack(ss).squeeze()
                    # s = torch.stack(ss).squeeze(dim=-1) # - not working
                    s = torch.stack(ss).squeeze(dim=1)
                    # # gs = torch.stack(gs).squeeze()              # !!! JMa: not used anymore?
                    # gs = torch.stack(gs).squeeze(dim=1)        # !!! JMa: not used anymore?
                    # s_trg = torch.cat([s, gs], dim=-1).detach() # !!! JMa: not used anymore?

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())  # [B, T, 768]
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)  # [B, 256, T]

                    # decode
                    d, p = model.predictor(
                        d_en, s, input_lengths, s2s_attn_mono, text_mask
                    )  # [B, 256, T]

                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)

                    en, gt, p_en, wav = [], [], [], []
                    for idx, (m, w) in enumerate(zip(mel_input_length, waves)):
                        mel_length = int(m.item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[idx, :, random_start : random_start + mel_len])
                        p_en.append(p[idx, :, random_start : random_start + mel_len])

                        gt.append(mels[idx, :, (random_start * 2) : ((random_start + mel_len) * 2)])
                        y = w[(random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()  # [B, T]

                    en = torch.stack(en)  # [B, 256, T]
                    p_en = torch.stack(p_en)  # [B, 256, T]
                    gt = torch.stack(gt).detach()
                    s = model.predictor_encoder(gt.unsqueeze(1))

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
                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, f0_fake, n_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())
                    f0_real, _, f0 = model.pitch_extractor(gt.unsqueeze(1))
                    loss_f0 = F.l1_loss(f0_real, f0_fake) / 10
                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_f0).mean()

                    iters_test += 1

                except Exception as e:
                    logger.error("[!] Error: %s", e)
                    traceback.print_exc()
                    continue

        # print('Epochs:', epoch + 1)
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
        if epoch < joint_epoch:
            # Generating reconstruction examples with GT duration
            with torch.no_grad():
                # Iterate over the defined number of validation samples
                for idx in range(min(n_val_audios, len(mel_input_length))):
                    mel_length = int(mel_input_length[idx].item())
                    # Ground-truth mel spectrogram
                    mel_gt = mels[idx, :, :mel_length].unsqueeze(0)
                    # Ground-truth phonemes-audio alignment
                    en_gt = asr[idx, :, : mel_length // 2].unsqueeze(0)
                    # # Reconstruct audio from ground-truth mel spectrogram and
                    # # phoneme-audio alignment
                    # wav = pts.reconstruct(mel_gt, en_gt)

                    # # Write and save val audio
                    # writer.add_audio(f"eval/y{idx}", wav, epoch, sample_rate=sr)
                    # if save_val_audio and epoch % saving_epoch == 0:
                    #     outfile = f"epoch_2nd_{epoch:0>5}_val-rec-{idx}.wav"
                    #     pts.save_wav(wav, os.path.join(test_audio_dir, outfile))

                    # Predicted phonemes-audio alignment encoding
                    p_en = p[idx, :, : mel_length // 2].unsqueeze(0)
                    # Reconstruct audio from ground-truth mel spectrogram,
                    # and extracted and predicted phoneme-audio alignment encoding
                    wav = pts.reconstruct(mel_gt, en_gt, p_en)

                    # Write and save val audio
                    writer.add_audio(f"pred/y{idx}", wav, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav, os.path.join(test_audio_dir, outfile))

                    # Save ground truth
                    if epoch == 0:
                        wav = waves[idx].squeeze()
                        if save_val_audio:
                            outfile = f"epoch_2nd_{epoch:0>5}_gt-{idx}.wav"
                            pts.save_wav(wav, os.path.join(test_audio_dir, outfile))
                        writer.add_audio(f"gt/y{idx}", wav, epoch, sample_rate=sr)

        else:
            # Generating sampled speech from text directly
            with torch.no_grad():
                ref_s = None
                # Compute reference styles from ground truth mel spectrogram
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))  # Timbre style
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))  # Prosody style
                    ref_s = torch.cat([ref_ss, ref_sp], dim=1)  # Combined style [B, 256, T]

                # Iterate over the defined number of validation samples
                for idx in range(min(n_val_audios, len(mel_input_length))):
                    # Generate audio from phoneme features of the `idx`-th validation file
                    wav, _ = pts.infer_from_ph_features(
                        input_lengths[idx, ...].unsqueeze(0),
                        text_mask[idx, : input_lengths[idx]].unsqueeze(0),
                        t_en[idx, :, : input_lengths[idx]].unsqueeze(0),
                        bert_dur[idx].unsqueeze(0),
                        d_en[idx, :, : input_lengths[idx]].unsqueeze(0),
                        ref_s=ref_s[idx].unsqueeze(0) if multispeaker else None,
                    )

                    # Write and save val audio
                    writer.add_audio(f"pred/y{idx}", wav, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile = f"epoch_2nd_{epoch:0>5}_val-pred-{idx}.wav"
                        pts.save_wav(wav, os.path.join(test_audio_dir, outfile))

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
