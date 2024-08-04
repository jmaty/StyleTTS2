# load packages
import logging
import os
import os.path as osp
import shutil
import time
import traceback
import warnings
from logging import StreamHandler
import copy

import click
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from IPython.core.debugger import set_trace
from monotonic_align import mask_from_lens
from munch import Munch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from losses import *
from meldataset import build_dataloader
from models import *
from Modules.diffusion.sampler import (ADPM2Sampler, DiffusionSampler,
                                       KarrasSchedule)
from Modules.slmadv import SLMAdversarialLoss
from optimizers import build_optimizer
from utils import *
from Utils.PLBERT.util import load_plbert

warnings.simplefilter('ignore')

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    with open(config_path, encoding="utf-8") as fr:
        config = yaml.safe_load(fr)

    log_dir = config['log_dir']
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs_2nd', 200)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    max_saved_models = config.get('max_saved_models', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    ood_data = data_params['OOD_data']
    save_val_audio = data_params.get('save_val_audio', False)
    save_test_audio = data_params.get('save_test_audio', False)
    test_sentences = data_params.get('test_sentences', [])
    test_audio_dir = os.path.join(config['log_dir'], config['data_params'].get('test_audio_dir', 'test_audios'))

    max_len = config.get('max_len', 200)
    # JMa: gradient clipping support
    grad_clip = config.get('grad_clip', None)
    # JMa: gradient accumulation
    grad_accum_steps = config.get('grad_accum_steps', 1)

    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch

    optimizer_params = Munch(config['optimizer_params'])

    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=ood_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=ood_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})

    # load pretrained ASR model
    asr_config = config.get('ASR_config', False)
    asr_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(asr_path, asr_config)

    # load pretrained F0 model
    f0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(f0_path)

    # load PL-BERT model
    bert_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(bert_path)

    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    # DP
    for key in model:
        if key not in ("mpd", "msd", "wd"):
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)

    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print(f'Loading the first stage model at {first_stage_path} ...')
            model, _, start_epoch, iters = load_checkpoint(model,
                None,
                first_stage_path,
                load_only_params=True,
                # keep starting epoch for tensorboard log
                ignore_modules=[
                    'bert', 
                    'bert_encoder',
                    'predictor',
                    'predictor_encoder',
                    'msd',
                    'mpd',
                    'wd',
                    'diffusion'
                    ])

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.')

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model,
                   model.wd,
                   sr,
                   model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        # empirical parameters
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)

    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4

    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(
            model,
            optimizer,
            config['pretrained_model'],
            load_only_params=config.get('load_only_params', True))

    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    iters = 0

    # criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to(device)

    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False    # Diffusion sampling

    running_std = []

    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(
        model, wl, sampler,
        slmadv_params.min_len,
        slmadv_params.max_len,
        batch_percentage=slmadv_params.batch_percentage,
        skip_update=slmadv_params.iter,
        sig=slmadv_params.sig
    )

    # Create test audio dir under log/eval dir
    if (save_val_audio or save_test_audio) and not os.path.exists(test_audio_dir):
        os.makedirs(test_audio_dir, exist_ok=True)

    # Total number of steps given the batch size
    tot_num_steps = len(train_list)//batch_size

    # Train model
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        # Set all models to eval mode
        _ = [model[key].eval() for key in model]

        # Set specified models to train mode
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()

        # Start diffusion sampling at a specified epoch
        if epoch >= diff_epoch:
            start_ds = True

        # JMa: Zero gradients at each epoch start
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except Exception:
                    continue    # skip batch

                mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_st)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = t_en @ s2s_attn_mono

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss, gs = [], []
            for idx, mel_input_length_item in enumerate(mel_input_length):
                mel_length = int(mel_input_length_item.item())
                mel = mels[idx, :, :mel_input_length_item]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze() # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    # batch-wise std estimation
                    model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item()
                    running_std.append(model.diffusion.module.diffusion.sigma_data)
 
                if multispeaker:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        features=ref, # reference from the same speaker as the embedding
                        embedding_mask_proba=0.1,
                        num_steps=num_steps).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion(s_trg.unsqueeze(1),
                                                embedding=bert_dur,
                                                features=ref).mean()
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach()) 
                else:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    # EDM loss
                    loss_diff = model.diffusion.module.diffusion(s_trg.unsqueeze(1),
                                                                 embedding=bert_dur).mean()
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty, loss_diff = 0, 0

            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en, gt, st, p_en, wav = [], [], [], [], []

            for idx, (mel_input_length_item, wave_item) in enumerate(zip(mel_input_length, waves)):
                mel_length = int(mel_input_length_item.item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[idx, :, random_start:random_start+mel_len])
                p_en.append(p[idx, :, random_start:random_start+mel_len])
                gt.append(mels[idx, :, (random_start * 2):((random_start+mel_len) * 2)])
                y = wave_item[(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[idx, :, (random_start * 2):((random_start+mel_len_st) * 2)])

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
                    # ground truth from recording
                    wav = y_rec_gt # use recording since decoder is tuned
                else:
                    # ground truth from reconstruction
                    wav = y_rec_gt_pred # use reconstruction since decoder is fixed

            f0_fake, n_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, f0_fake, n_fake, s)

            loss_f0_rec =  (F.smooth_l1_loss(f0_real, f0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

            #--- Discriminator loss ---
            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach(), y_rec.detach()).mean()
                d_loss = d_loss / grad_accum_steps  # JMa: normalize loss
                # JMa: Compute and accumulate gradients only for discriminators
                d_loss.backward(inputs=list(model.mpd.parameters()) + list(model.msd.parameters()))
                # JMa: Gradient accumulation
                if (i + 1) % grad_accum_steps == 0:
                    # JMa: gradient clipping
                    if grad_clip:
                        # _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                        nn.utils.clip_grad_norm_(model.msd.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.mpd.parameters(), grad_clip)
                    optimizer.step('msd')
                    optimizer.step('mpd')
                    optimizer.zero_grad('msd')   # zero gradient before another accumulation
                    optimizer.zero_grad('mpd')   # zero gradient before another accumulation
            else:
                d_loss = 0

            #--- Generator loss ---
            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean() if start_ds else 0
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce, loss_dur = 0, 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = loss_params.lambda_mel * loss_mel + \
                     loss_params.lambda_F0 * loss_f0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_gen * loss_gen_all + \
                     loss_params.lambda_slm * loss_lm + \
                     loss_params.lambda_sty * loss_sty + \
                     loss_params.lambda_diff * loss_diff
            g_loss = g_loss / grad_accum_steps  # JMa: normalize loss

            inputs = list(model.bert_encoder.parameters()) + \
                     list(model.bert.parameters()) + \
                     list(model.predictor.parameters()) + \
                     list(model.predictor_encoder.parameters())
            if epoch >= diff_epoch:
                inputs += list(model.diffusion.parameters())
            if epoch >= joint_epoch:
                inputs += list(model.style_encoder.parameters()) + list(model.decoder.parameters())
            g_loss.backward(inputs=inputs)

            running_loss += loss_mel.item()

            # JMa: Gradient accumulation
            if (i + 1) % grad_accum_steps == 0:
                # JMa: gradient clipping
                if grad_clip:
                    # _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                    nn.utils.clip_grad_norm_(model.bert_encoder.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.bert.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.predictor.parameters(), grad_clip)
                    nn.utils.clip_grad_norm_(model.predictor_encoder.parameters(), grad_clip)
                if torch.isnan(g_loss):
                    set_trace()

                optimizer.step('bert_encoder')
                optimizer.step('bert')
                optimizer.step('predictor')
                optimizer.step('predictor_encoder')
                optimizer.zero_grad('bert_encoder')
                optimizer.zero_grad('bert')
                optimizer.zero_grad('predictor')
                optimizer.zero_grad('predictor_encoder')
            
                if epoch >= diff_epoch:
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.diffusion.parameters(), grad_clip)
                    optimizer.step('diffusion')
                    optimizer.zero_grad('diffusion')

                # Joint training
                if epoch >= joint_epoch:
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.style_encoder.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
                    optimizer.step('style_encoder')
                    optimizer.step('decoder')
                    optimizer.zero_grad('style_encoder')
                    optimizer.zero_grad('decoder')

            if epoch >= joint_epoch:
                # randomly pick whether to use in-distribution text
                use_ind = np.random.rand() < 0.5

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts

                slm_out = slmadv(i,
                                y_rec_gt,
                                y_rec_gt_pred,
                                waves,
                                mel_input_length,
                                ref_texts,
                                ref_lengths, use_ind, s_trg.detach(),
                                ref if multispeaker else None)

                if slm_out is None:
                    continue

                d_loss_slm, loss_gen_lm, y_pred = slm_out

                # SLM generator loss
                # optimizer.zero_grad()
                loss_gen_lm /= grad_accum_steps  # JMa: normalize loss
                inputs = list(model.bert_encoder.parameters()) + \
                         list(model.bert.parameters()) + \
                         list(model.predictor.parameters()) + \
                         list(model.diffusion.parameters())
                loss_gen_lm.backward(inputs=inputs)

                # JMa: gradient clipping
                if grad_clip:
                    _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]

                # compute the gradient norm
                total_norm = {}
                for key in model.keys():
                    total_norm[key] = 0
                    parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                    for p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm[key] += param_norm.item() ** 2
                    total_norm[key] = total_norm[key] ** 0.5

                # gradient scaling
                if total_norm['predictor'] > slmadv_params.thresh:
                    for key in model.keys():
                        for p in model[key].parameters():
                            if p.grad is not None:
                                p.grad *= (1 / total_norm['predictor'])

                for p in model.predictor.duration_proj.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.predictor.lstm.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.diffusion.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale
                
                # JMa: Gradient accumulation
                if (i + 1) % grad_accum_steps == 0:
                    # JMa: gradient clipping
                    if grad_clip:
                        # _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                        nn.utils.clip_grad_norm_(model.bert_encoder.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.bert.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.predictor.parameters(), grad_clip)
                        nn.utils.clip_grad_norm_(model.diffusion.parameters(), grad_clip)

                    optimizer.step('bert_encoder')
                    optimizer.step('bert')
                    optimizer.step('predictor')
                    optimizer.step('diffusion')
                    optimizer.zero_grad('bert_encoder')
                    optimizer.zero_grad('bert')
                    optimizer.zero_grad('predictor')
                    optimizer.zero_grad('diffusion')

                # SLM discriminator loss
                if d_loss_slm != 0:
                    # optimizer.zero_grad()
                    d_loss_slm /= grad_accum_steps  # JMa: normalize loss
                    d_loss_slm.backward(inputs=list(model.wd.parameters()), retain_graph=True)
                    # JMa: Gradient accumulation
                    if (i + 1) % grad_accum_steps == 0:
                        # JMa: gradient clipping
                        if grad_clip:
                            # _ = [nn.utils.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                            nn.utils.clip_grad_norm_(model.wd.parameters(), grad_clip)
                        # JMa: gradient clipping
                        optimizer.step('wd')
                        optimizer.zero_grad('wd')
            else:
                d_loss_slm, loss_gen_lm = 0, 0

            iters += 1

            if (i+1)%log_interval == 0:
                mel_loss = running_loss / log_interval
                logger.info(f'Epoch [{epoch+1:3}/{epochs}], Step [{i+1:4}/{tot_num_steps}], Mel Loss: {mel_loss:.5f}, Disc Loss: {d_loss:.5f}, Dur Loss: {loss_dur:.5f}, CE Loss: {loss_ce:.5f}, Norm Loss: {loss_norm_rec:.5f}, F0 Loss: {loss_f0_rec:.5f}, LM Loss: {loss_lm:.5f}, Gen Loss: {loss_gen_all:.5f}, Sty Loss: {loss_sty:.5f}, Diff Loss: {loss_diff:.5f}, DiscLM Loss: {d_loss_slm:.5f}, GenLM Loss: {loss_gen_lm:.5f}')  
                writer.add_scalar('train/mel_loss', mel_loss, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/slm_loss', loss_lm, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_f0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                writer.add_scalar('train/d_loss_slm', d_loss_slm, iters)
                writer.add_scalar('train/gen_loss_slm', loss_gen_lm, iters)

                running_loss = 0

                print('Time elapsed:', time.time() - start_time)

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
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_st)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss, gs = [], []

                    for idx, mel_input_length_item in enumerate(mel_input_length):
                        mel_length = int(mel_input_length_item.item())
                        mel = mels[idx, :, :mel_input_length_item]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze()
                    gs = torch.stack(gs).squeeze()
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
                    d, p = model.predictor(d_en,
                                           s,
                                           input_lengths,
                                           s2s_attn_mono,
                                           text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en, gt, p_en, wav = [], [], [], []

                    for idx, (mel_input_length_item, wav_item) in enumerate(zip(mel_input_length, waves)):
                        mel_length = int(mel_input_length_item.item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[idx, :, random_start:random_start+mel_len])
                        p_en.append(p[idx, :, random_start:random_start+mel_len])

                        gt.append(mels[idx, :, (random_start * 2):((random_start+mel_len) * 2)])

                        y = wav_item[(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))

                    f0_fake, n_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for idx in range(_s2s_trg.shape[0]):
                            _s2s_trg[idx, :_text_input[idx]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1],
                                              _text_input[1:_text_length-1])

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
                    print(f"run into exception {e}")
                    traceback.print_exc()
                    continue

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f', loss_test/iters_test, loss_align/iters_test, loss_f/iters_test)
        # print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)

        if epoch < joint_epoch:
            # generating reconstruction examples with GT duration
            with torch.no_grad():
                for idx, mel_input_length_item in enumerate(mel_input_length):
                    mel_length = int(mel_input_length_item.item())
                    gt = mels[idx, :, :mel_length].unsqueeze(0)
                    en = asr[idx, :, :mel_length // 2].unsqueeze(0)

                    f0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    f0_real = f0_real.unsqueeze(0)
                    s = model.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                    y_rec = model.decoder(en, f0_real, real_norm, s)

                    # Write and save val audio
                    wav = y_rec.cpu().numpy().squeeze()
                    writer.add_audio(f'eval/y{idx}', wav, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile_template = f'epoch_2nd_{epoch:0>5}'
                        out_file = f'{outfile_template}_val-{idx}.wav'
                        scipy.io.wavfile.write(filename=os.path.join(test_audio_dir, out_file),
                                               rate=config['preprocess_params']['sr'],
                                               data=wav)

                    s_dur = model.predictor_encoder(gt.unsqueeze(1))
                    p_en = p[idx, :, :mel_length // 2].unsqueeze(0)

                    f0_fake, n_fake = model.predictor.F0Ntrain(p_en, s_dur)

                    y_pred = model.decoder(en, f0_fake, n_fake, s)
                    writer.add_audio(
                        f'pred/y{idx}',
                        y_pred.cpu().numpy().squeeze(),
                        epoch,
                        sample_rate=sr
                    )

                    if epoch == 0:
                        writer.add_audio(
                            f'gt/y{idx}',
                            waves[idx].squeeze(),
                            epoch,
                            sample_rate=sr
                        )
                    # Use up to 5 validation samples
                    if idx >= 5:
                        break
        else:
            # generating sampled speech from text directly
            with torch.no_grad():
                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref_s = torch.cat([ref_ss, ref_sp], dim=1)

                for idx, _ in enumerate(d_en):
                    if multispeaker:
                        s_pred = sampler(
                            noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                            embedding=bert_dur[idx].unsqueeze(0),
                            embedding_scale=1,
                            # reference from the same speaker as the embedding
                            features=ref_s[idx].unsqueeze(0),
                            num_steps=5).squeeze(1)
                    else:
                        s_pred = sampler(
                            noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                            embedding=bert_dur[idx].unsqueeze(0),
                            embedding_scale=1,
                            num_steps=5).squeeze(1)

                    s = s_pred[:, 128:]
                    ref = s_pred[:, :128]

                    d = model.predictor.text_encoder(
                        d_en[idx, :, :input_lengths[idx]].unsqueeze(0),
                        s,
                        input_lengths[idx, ...].unsqueeze(0),
                        text_mask[idx, :input_lengths[idx]].unsqueeze(0)
                    )

                    x, _ = model.predictor.lstm(d)
                    duration = model.predictor.duration_proj(x)

                    duration = torch.sigmoid(duration).sum(axis=-1)
                    pred_dur = torch.round(duration.squeeze()).clamp(min=1)

                    pred_dur[-1] += 5

                    pred_aln_trg = torch.zeros(input_lengths[idx], int(pred_dur.sum().data))
                    c_frame = 0
                    for i in range(pred_aln_trg.size(0)):
                        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                        c_frame += int(pred_dur[i].data)

                    # encode prosody
                    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device)
                    f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
                    out = model.decoder(t_en[idx, :, :input_lengths[idx]].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(texts.device),
                                        f0_pred,
                                        n_pred,
                                        ref.squeeze().unsqueeze(0))
                    
                    # Write and save val audio
                    wav = out.cpu().numpy().squeeze()
                    writer.add_audio('pred/y' + str(idx), wav, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile_template = f'epoch_2nd_{epoch:0>5}'
                        out_file = f'{outfile_template}_val-{idx}.wav'
                        scipy.io.wavfile.write(filename=os.path.join(test_audio_dir, out_file),
                                               rate=config['preprocess_params']['sr'],
                                               data=wav)
                    # Use up to 5 validation samples
                    if idx >= 5:
                        break

        if epoch % saving_epoch == 0:
            curr_loss = loss_test / iters_test
            if curr_loss < best_loss:
                best_loss = curr_loss
            # Prepare model state for saving
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': curr_loss,
                'epoch': epoch,
            }
            # Save model
            save_model(state, '2nd', epoch, log_dir, max_saved_models)

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))

                with open(osp.join(log_dir, osp.basename(config_path)), 'w', encoding='utf-8') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

            # JMa: synthesize test audios
            if save_test_audio:
                synth_test_files(model,
                                test_sentences,
                                test_audio_dir,
                                f'epoch_2nd_{epoch:0>5}_test',
                                sr,
                                sampler=None,
                                diffusion_steps=5,
                                embedding_scale=1,
                                device=device)

        # Save auxiliary models
        if epoch in (diff_epoch-1, joint_epoch-1):
            # Prepare model state fo saving
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            if epoch == diff_epoch-1:
                save_path = osp.join(log_dir, f'pre-diff_2nd_{epoch:0>5}.pth')
                phase = 'Pre-diffusion'
            else:
                save_path = osp.join(log_dir, f'pre-joint_2nd_{epoch:0>5}.pth')
                phase = 'Pre-joint'
            torch.save(state, save_path)
            print(f'{phase} phase model saved (epoch) {epoch}')


if __name__=="__main__":
    main(None)
