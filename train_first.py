import logging
import os
import os.path as osp
import random
import shutil
import time
import warnings
import numpy as np
import scipy

import click
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from monotonic_align import mask_from_lens
from munch import Munch
from torch.utils.tensorboard import SummaryWriter

from losses import *
from meldataset import build_dataloader
from models import *
from optimizers import build_optimizer
from utils import (get_data_path_list, get_image, length_to_mask, log_norm,
                   log_print, maximum_path, recursive_munch, save_model,
                   synth_test_files)
from Utils.PLBERT.util import load_plbert

warnings.simplefilter('ignore')

logger = get_logger(__name__, log_level="DEBUG")

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    with open(config_path, encoding="utf-8") as fr:
        config = yaml.safe_load(fr)

    log_dir = config['log_dir']
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = accelerator.device

    epochs = config.get('epochs_1st', 200)
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

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

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

    with accelerator.main_process_first():
        # load pretrained ASR model
        asr_config = config.get('ASR_config', False)
        asr_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(asr_path, asr_config)

        # load pretrained F0 model
        f0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(f0_path)

        # load BERT model
        bert_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(bert_path)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    best_loss = float('inf')  # best test loss

    loss_params = Munch(config['loss_params'])
    tma_epoch = loss_params.TMA_epoch

    for k in model:
        model[k] = accelerator.prepare(model[k])

    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )

    # Move models to device (cuda)
    _ = [model[key].to(device) for key in model]

    # initialize optimizers after preparing models for compatibility with FSDP
    parameters_dict = {key: model[key].parameters() for key in model}
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    lr = float(config['optimizer_params'].get('lr', 1e-4))
    optimizer = build_optimizer(parameters_dict, scheduler_params_dict, lr)

    for k, _ in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(
                model,
                optimizer,
                config['pretrained_model'],
                load_only_params=config.get('load_only_params', True))
            print(f'Loading pre-trained model: {config["pretrained_model"]}')
            print(f'Starting epoch:      {start_epoch}')
            print(f'Starting iterations: {iters}')
            print()
        else:
            start_epoch = 0
            iters = 0

    # in case not distributed computing
    try:
        n_down = model.text_aligner.module.n_down
    except AttributeError:
        print("Distributed computing NOT used")
        n_down = model.text_aligner.n_down

    # wrapped losses for compatibility with mixed precision
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model,
                   model.wd,
                   sr,
                   model_params.slm.sr).to(device)

    # Create test audio dir under log/eval dir
    if (save_val_audio or save_test_audio) and not os.path.exists(test_audio_dir):
        os.makedirs(test_audio_dir, exist_ok=True)

    # Total number of steps given the batch size
    tot_num_steps = len(train_list)//batch_size

    # Train model
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        # Set all models to train mode
        _ = [model[key].train() for key in model]

        # JMa: Zero gradients of all optimizers at each epoch start
        optimizer.zero_grad()

        # Train loop for each epoch
        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, _, _, mels, mel_input_length, _ = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(mel_input_length.device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

            _, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)

            s2s_attn.masked_fill_(attn_mask, 0.0)

            with torch.no_grad():
                mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_st)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = t_en @ s2s_attn
            else:
                asr = t_en @ s2s_attn_mono

            # get clips
            mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
            mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)

            en, gt, wav, st = [], [], [], []

            for idx, (mel_input_length_item, wave_item) in enumerate(zip(mel_input_length, waves)):
                mel_length = int(mel_input_length_item.item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[idx, :, random_start:random_start+mel_len])
                gt.append(mels[idx, :, (random_start * 2):((random_start+mel_len) * 2)])

                y = wave_item[(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
 
                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[idx, :, (random_start * 2):((random_start+mel_len_st) * 2)])

            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            wav = torch.stack(wav).float().detach()

            # clip too short to be used by the style encoder
            if gt.shape[-1] < 80:
                continue

            with torch.no_grad():    
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                f0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))

            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))

            y_rec = model.decoder(en, f0_real, real_norm, s)

            #--- Discriminator loss ---
            if epoch >= tma_epoch:
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                d_loss = d_loss / grad_accum_steps  # JMa: normalize loss
                # JMa: Compute gradients only for discriminators
                accelerator.backward(d_loss, inputs=list(model.mpd.parameters()) + list(model.msd.parameters()))
                # JMa: Gradient accumulation
                if (i + 1) % grad_accum_steps == 0:
                    # JMa: gradient clipping
                    if grad_clip:
                        _ = [accelerator.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]
                    optimizer.step('msd')
                    optimizer.step('mpd')
                    optimizer.zero_grad('msd')
                    optimizer.zero_grad('mpd')
            else:
                d_loss = 0

            #--- Generator loss ---
            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

            if epoch >= tma_epoch: # start TMA training
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                    loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                loss_s2s /= texts.size(0)

                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

                loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                loss_slm = wl(wav.detach(), y_rec).mean()

                g_loss = loss_params.lambda_mel * loss_mel + \
                         loss_params.lambda_mono * loss_mono + \
                         loss_params.lambda_s2s * loss_s2s + \
                         loss_params.lambda_gen * loss_gen_all + \
                         loss_params.lambda_slm * loss_slm

            else:
                loss_s2s = 0
                loss_mono = 0
                loss_gen_all = 0
                loss_slm = 0
                g_loss = loss_mel
            
            g_loss = g_loss / grad_accum_steps  # JMa: normalize loss
            # JMa: Compute gradients only for generator
            inputs = list(model.decoder.parameters()) + \
                     list(model.style_encoder.parameters()) + \
                     list(model.text_encoder.parameters())
            if epoch >= tma_epoch:
                inputs += list(model.text_aligner.parameters())
            accelerator.backward(g_loss, inputs=inputs)

            running_loss += accelerator.gather(loss_mel).mean().item()

            # JMa: Gradient accumulation
            if (i + 1) % grad_accum_steps == 0:
                # JMa: gradient clipping
                if grad_clip:
                    _ = [accelerator.clip_grad_norm_(model[k].parameters(), grad_clip) for k in model]

                optimizer.step('text_encoder')
                optimizer.step('style_encoder')
                optimizer.step('decoder')
                # optimizer.zero_grad('text_encoder')
                # optimizer.zero_grad('style_encoder')
                # optimizer.zero_grad('decoder')

                if epoch >= tma_epoch:
                    optimizer.step('text_aligner')
                    # JMa: pitch extractor should not be updated, see:
                    # https://github.com/yl4579/StyleTTS2/issues/10#issuecomment-1783701686
                    # optimizer.step('pitch_extractor')
                    # optimizer.zero_grad('text_aligner')
                # Zero all gradients

                optimizer.zero_grad()

            iters = iters + 1

            if (i+1)%log_interval == 0 and accelerator.is_main_process:
                mel_loss = running_loss / log_interval
                log_print(f'Epoch [{epoch+1:3}/{epochs}], Step [{i+1:4}/{tot_num_steps}], Mel Loss: {mel_loss:.5f}, Gen Loss: {loss_gen_all:.5f}, Disc Loss: {d_loss:.5f}, Mono Loss: {loss_mono:.5f}, S2S Loss: {loss_s2s:.5f}, SLM Loss: {loss_slm:.5f}', logger)
                writer.add_scalar('train/mel_loss', mel_loss, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/mono_loss', loss_mono, iters)
                writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                writer.add_scalar('train/slm_loss', loss_slm, iters)
                print('Time elapsed:', time.time()-start_time)
                running_loss = 0

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
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                    _, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                    s2s_attn.masked_fill_(attn_mask, 0.0)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)

                asr = t_en @ s2s_attn

                # get clips
                mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
                mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])

                en, gt, wav = [], [], []
                for idx, (mel_input_length_item, wave_item) in enumerate(zip(mel_input_length, waves)):
                    mel_length = int(mel_input_length_item.item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[idx, :, random_start:random_start+mel_len])
                    gt.append(mels[idx, :, (random_start * 2):((random_start+mel_len) * 2)])
                    y = wave_item[(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                    wav.append(torch.from_numpy(y).to('cuda'))

                wav = torch.stack(wav).float().detach()

                en = torch.stack(en)
                gt = torch.stack(gt).detach()

                f0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                s = model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec = model.decoder(en, f0_real, real_norm, s)

                loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                loss_test += accelerator.gather(loss_mel).mean().item()
                iters_test += 1

        if accelerator.is_main_process:
            # print('Epochs:', epoch + 1)
            log_print(f'Epoch [{epoch+1:3}/{epochs}]: validation loss: {loss_test/iters_test:.3f}', logger)
            # print('\n\n\n')
            writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
            attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            writer.add_figure('eval/attn', attn_image, epoch)

            with torch.no_grad():
                for idx, (mel_input_length_item, wave_item) in enumerate(zip(mel_input_length, waves)):
                    mel_length = int(mel_input_length_item.item())
                    gt = mels[idx, :, :mel_length].unsqueeze(0)
                    en = asr[idx, :, :mel_length // 2].unsqueeze(0)

                    f0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    f0_real = f0_real.unsqueeze(0)    # JMa
                    s = model.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                    y_rec = model.decoder(en, f0_real, real_norm, s)

                    # Write and save val audio
                    wav = y_rec.cpu().numpy().squeeze()
                    writer.add_audio('eval/y' + str(idx), wav, epoch, sample_rate=sr)
                    if save_val_audio and epoch % saving_epoch == 0:
                        outfile_template = f'epoch_1st_{epoch:0>5}'
                        out_file = f'{outfile_template}_val-{idx}.wav'
                        scipy.io.wavfile.write(filename=os.path.join(test_audio_dir, out_file),
                                               rate=config['preprocess_params']['sr'],
                                               data=wav)
                    # Write and save ground-truth audio
                    if epoch == 0:
                        wav = wave_item.squeeze()
                        writer.add_audio('gt/y' + str(idx), wav, epoch, sample_rate=sr)
                        if save_val_audio:
                            out_file = f'{outfile_template}_gt-{idx}.wav'
                            scipy.io.wavfile.write(filename=os.path.join(test_audio_dir, out_file),
                                                   rate=config['preprocess_params']['sr'],
                                                   data=wav)
                    # Use up to 6 validation samples
                    if idx >= 6:
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
                save_model(state, '1st', epoch, log_dir, max_saved_models)

                # JMa: synthesize test audios
                if save_test_audio:
                    synth_test_files(model,
                                     test_sentences,
                                     test_audio_dir,
                                    f'epoch_1st_{epoch:0>5}_test',
                                    sr,
                                    sampler=None,
                                    diffusion_steps=5,
                                    embedding_scale=1,
                                    device=device)
            # Save pre-TMA model
            if epoch == tma_epoch - 1:
                # Prepare model state fo saving
                state = {
                    'net':  {key: model[key].state_dict() for key in model}, 
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': loss_test / iters_test,
                    'epoch': epoch,
                }
                save_path = osp.join(log_dir, f'pre-tma_1st_{epoch:0>5}.pth')
                torch.save(state, save_path)
                print('Pre-TMA model saved')

    if accelerator.is_main_process:
        state = {
            'net':  {key: model[key].state_dict() for key in model}, 
            'optimizer': optimizer.state_dict(),
            'iters': iters,
            'val_loss': loss_test / iters_test,
            'epoch': epoch,
        }
        save_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
        torch.save(state, save_path)
        print('Final first-stage model saved')


if __name__=="__main__":
    main(None)
