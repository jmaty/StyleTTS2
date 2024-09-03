# load packages
import copy
import logging
import os
import os.path as osp
import random
import shutil
import time
import warnings
from logging import StreamHandler
import nvidia_smi

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import yaml
from IPython.core.debugger import set_trace
from monotonic_align import mask_from_lens
# from munch import Munch
# from torch import nn
from torch.utils.tensorboard import SummaryWriter

from losses import (DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss,
                    WavLMLoss)
from meldataset import build_dataloader
from models import (build_model, load_ASR_models, load_checkpoint,
                    load_F0_models, save_checkpoint)
from Modules.diffusion.sampler import (ADPM2Sampler, DiffusionSampler,
                                       KarrasSchedule)
from Modules.slmadv import SLMAdversarialLoss
from optimizers import build_optimizer
from utils import (get_data_path_list, get_image, length_to_mask, log_norm,
                   maximum_path, recursive_munch, synth_test_files)
from Utils.PLBERT_mlng.util import load_plbert

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


class StyleTTS2Finetune():
    def __init__(self, config_path, accelerator):

        self.config_path = config_path

        # Load config
        with open(config_path, encoding="utf-8") as fr:
            self.config = yaml.safe_load(fr)
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        shutil.copy(config_path, osp.join(self.log_dir, osp.basename(config_path)))
        self.accelerator = accelerator

        self.writer = SummaryWriter(self.log_dir + "/tensorboard")
        self.device = accelerator.device

        self.start_epoch = 0
        self.iters = 0

        self.train_dataloader, self.val_dataloader = None, None

        self.model = None
        self.optimizer = None
        self.init_model()
        self.sampler = None
        self.slmadv = None

        self.train_list, self.val_list = get_data_path_list(
            self.train_path,
            self.val_path
        )
        self.build_dataloaders()

        # Init losses
        self.gl, self.dl, self.wl = None, None, None

        # Create test audio dir under log/eval dir
        if (self.save_val_audio or self.save_test_audio) and \
            not os.path.exists(self.test_audio_dir):
            os.makedirs(self.test_audio_dir, exist_ok=True)

        self.best_loss = float('inf')
        self.stft_loss = MultiResolutionSTFTLoss().to(self.device)
        self.running_std = []

        # Init NVLM
        nvidia_smi.nvmlInit()
        self.n_gpus = nvidia_smi.nvmlDeviceGetCount()
        logger.info('NVLM initialized')


    def shutdown(self):
        # Ending work with NVIDIA NVLM
        nvidia_smi.nvmlShutdown()
        logger.info('NVLM shutdown')


    # Init model
    def init_model(self):
        # load pretrained ASR model
        asr_config = self.config.get('ASR_config', False)
        asr_path = self.config.get('ASR_path', False)
        text_aligner = load_ASR_models(asr_path, asr_config)
        # load pretrained F0 model
        f0_path = self.config.get('F0_path', False)
        pitch_extractor = load_F0_models(f0_path)
        # load PL-BERT model
        bert_path = self.config.get('PLBERT_dir', False)
        plbert = load_plbert(bert_path)
        # build model
        self.model = build_model(
            recursive_munch(self.config['model_params']),
            text_aligner, pitch_extractor,
            plbert
        )
        _ = [self.model[key].to(self.device) for key in self.model]

        # DP
        for key in self.model:
            if key not in ("mpd", "msd", "wd"):
                self.model[key] = MyDataParallel(self.model[key])


    # Load 1st-stage pretrained model
    def load_stage1_model(self):
        # If no pretrained model in the 2nd stage => load final 1st model
        first_stage_path = self.config.get('first_stage_path', None)
        if not first_stage_path:
            raise ValueError('You need to specify the path to the first stage model.')

        # Load 1st-stage model
        first_stage_path = osp.join(
            self.log_dir,
            first_stage_path
        )
        print(f'Loading the first stage model at {first_stage_path} ...')
        self.model, _, self.start_epoch, self.iters = load_checkpoint(
            self.model,
            None,
            first_stage_path,
            load_only_params=True,
            ignore_modules=[    # keep starting epoch for tensorboard log
                'bert',
                'bert_encoder',
                'predictor',
                'predictor_encoder',
                'msd',
                'mpd',
                'wd',
                'diffusion'
            ]
        )

        # these epochs should be counted from the start epoch
        self.diff_epoch += self.start_epoch
        self.joint_epoch += self.start_epoch
        self.epochs += self.start_epoch

        self.model.predictor_encoder = copy.deepcopy(self.model.style_encoder)


    def load_stage2_model(self):
        self.model, self.optimizer, self.start_epoch, self.iters = load_checkpoint(
            self.model,
            self.optimizer,
            self.config['pretrained_model'],
            load_only_params=self.config.get('load_only_params', True)
        )
        # advance start epoch or we'd re-train and rewrite the last epoch file
        self.start_epoch += 1
        print(f'Loading pre-trained model: {self.config["pretrained_model"]}')
        print(f'Starting epoch:      {self.start_epoch}')
        print(f'Starting iterations: {self.iters}')
        print()


    def init_losses(self):
        self.gl = GeneratorLoss(self.model.mpd, self.model.msd).to(self.device)
        self.dl = DiscriminatorLoss(self.model.mpd, self.model.msd).to(self.device)
        self.wl = WavLMLoss(
            self.config['model_params']['slm']['model'],
            self.model.wd,
            self.sr,
            self.config['model_params']['slm']['sr']
        ).to(self.device)
        self.gl = MyDataParallel(self.gl)
        self.dl = MyDataParallel(self.dl)
        self.wl = MyDataParallel(self.wl)


    def init_sampler(self):
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            # empirical parameters
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
    )

    def init_slmadv_loss(self):
        self.slmadv = SLMAdversarialLoss(
            self.model,
            self.wl,
            self.sampler,
            self.slmadv_params['min_len'],
            self.slmadv_params['max_len'],
            batch_percentage=self.slmadv_params['batch_percentage'],
            skip_update=self.slmadv_params['iter'],
            sig=self.slmadv_params['sig']
        )

    def prep_optimizer(self):
        optimizer_params = self.config['optimizer_params']
        scheduler_params = {
            "max_lr": optimizer_params['lr'],
            "pct_start": float(0),
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }
        scheduler_params_dict= {key: scheduler_params.copy() for key in self.model}
        scheduler_params_dict['bert']['max_lr'] = optimizer_params['bert_lr'] * 2
        scheduler_params_dict['decoder']['max_lr'] = optimizer_params['ft_lr'] * 2
        scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params['ft_lr'] * 2

        self.optimizer = build_optimizer(
            {key: self.model[key].parameters() for key in self.model},
            scheduler_params_dict=scheduler_params_dict, lr=optimizer_params['lr'])

        # adjust BERT learning rate
        for g in self.optimizer.optimizers['bert'].param_groups:
            g['betas'] = (0.9, 0.99)
            g['lr'] = optimizer_params['bert_lr']
            g['initial_lr'] = optimizer_params['bert_lr']
            g['min_lr'] = 0
            g['weight_decay'] = 0.01

        # adjust acoustic module learning rate
        for module in ["decoder", "style_encoder"]:
            for g in self.optimizer.optimizers[module].param_groups:
                g['betas'] = (0.0, 0.99)
                g['lr'] = optimizer_params['ft_lr']
                g['initial_lr'] = optimizer_params['ft_lr']
                g['min_lr'] = 0
                g['weight_decay'] = 1e-4


    def build_dataloaders(self):
        self.train_dataloader = build_dataloader(
            self.train_list,
            self.root_path,
            OOD_data=self.ood_data,
            min_length=self.min_length,
            batch_size=self.batch_size,
            num_workers=2,
            dataset_config={},
            device=self.device
        )

        self.val_dataloader = build_dataloader(
            self.val_list,
            self.root_path,
            OOD_data=self.ood_data,
            min_length=self.min_length,
            batch_size=self.batch_size,
            validation=True,
            num_workers=0,
            dataset_config={},
            device=self.device
        )


    def prep_accelerator(self):
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader
        )

    # compute the gradient norm
    def grad_norm(self):
        total_norm = {}
        for k in self.model.keys():
            total_norm[k] = 0
            parameters = [p for p in self.model[k].parameters()\
                          if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm[k] += param_norm.item() ** 2
            total_norm[k] = total_norm[k] ** 0.5
        return total_norm


    # gradient scaling
    def grad_scaling(self, total_norm):
        if total_norm['predictor'] > self.slmadv_params['thresh']:
            for key in self.model.keys():
                for p in self.model[key].parameters():
                    if p.grad is not None:
                        p.grad *= (1 / total_norm['predictor'])

        for p in self.model.predictor.duration_proj.parameters():
            if p.grad is not None:
                p.grad *= self.slmadv_params['scale']

        for p in self.model.predictor.lstm.parameters():
            if p.grad is not None:
                p.grad *= self.slmadv_params['scale']

        for p in self.model.diffusion.parameters():
            if p.grad is not None:
                p.grad *= self.slmadv_params['scale']


    def eval_mode(self, key=None):
        if key is not None:
            self.model[key].eval()
        else:
            # Set all models to eval mode
            _ = [self.model[k].eval() for k in self.model]


    def train_mode(self, key=None):
        if key is not None:
            self.model[key].train()
        else:
            # Set all models to eval mode
            _ = [self.model[k].train() for k in self.model]


    def clip_grad_norm(self, key=None):
        if key is not None:
            self.accelerator.clip_grad_norm_(self.model[key].parameters(), self.grad_clip)
        else:
            # Set clip all model parameters
            _ = [self.accelerator.clip_grad_norm_(self.model[k].parameters(), self.grad_clip) \
                 for k in self.model]


    def finetune(self):
        self.iters = 0
        self.best_loss = float('inf')   # Init best loss
        self.running_std = []

        for epoch in range(self.start_epoch, self.epochs):
            # Set all models to eval mode
            self.eval_mode()

            # Training loop
            self._training_loop(epoch)

            # Validation loop
            loss_test, iters_test = self._validation_loop(epoch)

            # Save progress
            if (epoch+1) % self.save_freq == 0:
                self._save_progress(epoch, loss_test, iters_test)
                # JMa: synthesize test audios
                if self.save_test_audio:
                    logger.info("Synthesizing %d test sentences to %s",
                                len(self.test_sentences),
                                self.test_audio_dir)
                    synth_test_files(
                        self.model,
                        self.test_sentences,
                        self.test_audio_dir,
                        f'epoch_2nd_{epoch:0>5}_test',
                        self.sr,
                        sampler=None,
                        diffusion_steps=5,
                        embedding_scale=1,
                        device=self.device
                    )

            # Save milestone models
            if epoch == self.diff_epoch - 1:
                state = {
                    'net':  {key: self.model[key].state_dict() for key in self.model}, 
                    'optimizer': self.optimizer.state_dict(),
                    'iters': self.iters,
                    'val_loss': loss_test / iters_test,
                    'epoch': epoch,
                }
                save_checkpoint(state, 'pre-diff', epoch, self.log_dir)
            if epoch == self.joint_epoch - 1:
                state = {
                    'net':  {key: self.model[key].state_dict() for key in self.model}, 
                    'optimizer': self.optimizer.state_dict(),
                    'iters': self.iters,
                    'val_loss': loss_test / iters_test,
                    'epoch': epoch,
                }
                save_checkpoint(state, 'pre-joint', epoch, self.log_dir)


    def _training_loop(self, epoch):
        # Set following models to train mode
        self.train_mode('text_aligner')
        self.train_mode('text_encoder')
        self.train_mode('predictor')
        self.train_mode('bert_encoder')
        self.train_mode('bert')
        self.train_mode('msd')
        self.train_mode('mpd')

        running_loss = 0
        start_time = time.time()
        for i, batch in enumerate(self.train_dataloader):
            waves = batch[0]
            batch = [b.to(self.device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** self.n_down)).to(self.device)
                # mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                # compute reference styles
                if self.multispeaker and epoch >= self.diff_epoch:
                    ref_ss = self.model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = self.model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            try:
                _, s2s_pred, s2s_attn = self.model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)
            except Exception as e:
                print(f"[!] Error: {e}")
                continue    # skip batch

            mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**self.n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_st)

            # encode
            t_en = self.model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            asr = t_en @ s2s_attn if bool(random.getrandbits(1)) else t_en @ s2s_attn_mono

            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer
            # (may need to work on masked avgpool)
            ss, gs = [], []
            for idx, m in enumerate(mel_input_length):
                mel_length = int(m.item())
                mel = mels[idx, :, :m]
                ss.append(self.model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                gs.append(self.model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze() # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

            bert_dur = self.model.bert(texts, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            # denoiser training
            if epoch >= self.diff_epoch:
                num_steps = np.random.randint(3, 5)

                if self.model_params['diffusion']['dist']['estimate_sigma_data']:
                    # batch-wise std estimation
                    self.model.diffusion.module.diffusion.sigma_data = s_trg.std(axis=-1).mean().item()
                    self.running_std.append(self.model.diffusion.module.diffusion.sigma_data)

                if self.multispeaker:
                    s_preds = self.sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(self.device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        # reference from the same speaker as the embedding
                        features=ref,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = self.model.diffusion(
                        s_trg.unsqueeze(1),
                        embedding=bert_dur,
                        features=ref
                    ).mean()
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
                else:   # single-speaker
                    s_preds = self.sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(self.device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps
                    ).squeeze(1)
                    # EDM loss
                    loss_diff = self.model.diffusion.module.diffusion(
                        s_trg.unsqueeze(1),
                        embedding=bert_dur
                    ).mean()
                    # style reconstruction loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty, loss_diff = 0, 0

            d, p = self.model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

            mel_len_st = int(mel_input_length.min().item()/2 - 1)
            mel_len = min(int(mel_input_length.min().item()/2 - 1),
                          self.max_len // 2)
            en, gt, st, p_en, wav = [], [], [], [], []

            for idx, (m, w) in enumerate(zip(mel_input_length, waves)):
                mel_length = int(m.item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[idx, :, random_start:random_start+mel_len])
                p_en.append(p[idx, :, random_start:random_start+mel_len])
                gt.append(mels[idx, :, (random_start * 2):((random_start+mel_len) * 2)])

                y = w[(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(self.device))

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

            # s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            # s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            s = self.model.style_encoder(gt.unsqueeze(1))     
            s_dur = self.model.predictor_encoder(gt.unsqueeze(1))

            with torch.no_grad():
                f0_real, _, f0 = self.model.pitch_extractor(gt.unsqueeze(1))
                f0 = f0.reshape(f0.shape[0], f0.shape[1] * 2, f0.shape[2], 1).squeeze()
                n_real = log_norm(gt.unsqueeze(1)).squeeze(1)

                y_rec_gt = wav.unsqueeze(1)
                y_rec_gt_pred = self.model.decoder(en, f0_real, n_real, s)

                wav = y_rec_gt

            f0_fake, n_fake = self.model.predictor.F0Ntrain(p_en, s_dur)
            y_rec = self.model.decoder(en, f0_fake, n_fake, s)

            loss_f0_rec =  (F.smooth_l1_loss(f0_real, f0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

            #--- Discriminator loss ---
            self.optimizer.zero_grad()
            d_loss = self.dl(wav.detach(), y_rec.detach()).mean()
            self.accelerator.backward(d_loss)
            # JMa: gradient clipping
            if self.grad_clip:
                self.clip_grad_norm('msd')
                self.clip_grad_norm('mpd')
            self.optimizer.step('msd')
            self.optimizer.step('mpd')

            #--- Generator loss ---
            self.optimizer.zero_grad()

            loss_mel = self.stft_loss(y_rec, wav)
            loss_gen_all = self.gl(wav, y_rec).mean()
            loss_lm = self.wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce, loss_dur = 0, 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(
                    _s2s_pred.flatten(),
                    _s2s_trg.flatten()
                )

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            g_loss = self.loss_params['lambda_mel'] * loss_mel + \
                     self.loss_params['lambda_F0'] * loss_f0_rec + \
                     self.loss_params['lambda_ce'] * loss_ce + \
                     self.loss_params['lambda_norm'] * loss_norm_rec + \
                     self.loss_params['lambda_dur'] * loss_dur + \
                     self.loss_params['lambda_gen'] * loss_gen_all + \
                     self.loss_params['lambda_slm'] * loss_lm + \
                     self.loss_params['lambda_sty'] * loss_sty + \
                     self.loss_params['lambda_diff'] * loss_diff + \
                     self.loss_params['lambda_mono'] * loss_mono + \
                     self.loss_params['lambda_s2s'] * loss_s2s

            running_loss += loss_mel.item()
            self.accelerator.backward(g_loss)
            # JMa: gradient clipping
            if self.grad_clip:
                self.clip_grad_norm('bert_encoder')
                self.clip_grad_norm('bert')
                self.clip_grad_norm('predictor')
                self.clip_grad_norm('predictor_encoder')
                self.clip_grad_norm('style_encoder')
                self.clip_grad_norm('decoder')
                self.clip_grad_norm('text_encoder')
                self.clip_grad_norm('text_aligner')
            if torch.isnan(g_loss):
                set_trace()

            self.optimizer.step('bert_encoder')
            self.optimizer.step('bert')
            self.optimizer.step('predictor')
            self.optimizer.step('predictor_encoder')
            self.optimizer.step('style_encoder')
            self.optimizer.step('decoder')
            self.optimizer.step('text_encoder')
            self.optimizer.step('text_aligner')

            if epoch >= self.diff_epoch:
                if self.grad_clip:
                    self.clip_grad_norm('diffusion')
                self.optimizer.step('diffusion')

            d_loss_slm, loss_gen_lm = 0, 0
            if epoch >= self.joint_epoch:
                # randomly pick whether to use in-distribution text
                use_ind = np.random.rand() < 0.5

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts

                slm_out = self.slmadv(
                    i,
                    y_rec_gt,
                    y_rec_gt_pred,
                    waves,
                    mel_input_length,
                    ref_texts,
                    ref_lengths,
                    use_ind,
                    s_trg.detach(),
                    ref if self.multispeaker else None
                )

                if slm_out is not None:
                    d_loss_slm, loss_gen_lm, _ = slm_out

                    # SLM generator loss
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss_gen_lm)
                    # JMa: gradient clipping
                    if self.grad_clip:
                        self.clip_grad_norm('bert_encoder')
                        self.clip_grad_norm('bert')
                        self.clip_grad_norm('predictor')
                        self.clip_grad_norm('diffusion')

                    # compute the gradient norm
                    total_norm = self.grad_norm()
                    # gradient scaling
                    self.grad_scaling(total_norm)

                    self.optimizer.step('bert_encoder')
                    self.optimizer.step('bert')
                    self.optimizer.step('predictor')
                    self.optimizer.step('diffusion')

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        self.optimizer.zero_grad()
                        self.accelerator.backward(d_loss_slm)
                        # JMa: gradient clipping
                        if self.grad_clip:
                            self.clip_grad_norm('wd')
                        self.optimizer.step('wd')

            self.iters += 1

            if (i+1) % self.log_interval == 0:
                mel_loss = running_loss / self.log_interval
                logger.info(
                    'Epoch [%d/%d], ' \
                    'Step [%d/%d], ' \
                    'Mel Loss: %.5f, ' \
                    'Disc Loss: %.5f, ' \
                    'Dur Loss: %.5f, ' \
                    'CE Loss: %.5f, '  \
                    'Norm Loss: %.5f, ' \
                    'F0 Loss: %.5f, ' \
                    'LM Loss: %.5f, ' \
                    'Gen Loss: %.5f, ' \
                    'Sty Loss: %.5f, ' \
                    'Diff Loss: %.5f, ' \
                    'DiscLM Loss: %.5f, ' \
                    'GenLM Loss: %.5f, ' \
                    'S2S Loss: %.5f, ' \
                    'Mono Loss: %.5f',
                    epoch+1,
                    self.epochs,
                    i+1,
                    self.tot_num_steps,
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
                    loss_s2s,
                    loss_mono,
                )
                self.writer.add_scalar('train/mel_loss', mel_loss, self.iters)
                self.writer.add_scalar('train/gen_loss', loss_gen_all, self.iters)
                self.writer.add_scalar('train/d_loss', d_loss, self.iters)
                self.writer.add_scalar('train/dur_loss', loss_dur, self.iters)
                self.writer.add_scalar('train/ce_loss', loss_ce, self.iters)
                self.writer.add_scalar('train/slm_loss', loss_lm, self.iters)
                self.writer.add_scalar('train/norm_loss', loss_norm_rec, self.iters)
                self.writer.add_scalar('train/F0_loss', loss_f0_rec, self.iters)
                self.writer.add_scalar('train/sty_loss', loss_sty, self.iters)
                self.writer.add_scalar('train/diff_loss', loss_diff, self.iters)
                self.writer.add_scalar('train/d_loss_slm', d_loss_slm, self.iters)
                self.writer.add_scalar('train/gen_loss_slm', loss_gen_lm, self.iters)

                running_loss = 0
                for device_idx in range(self.n_gpus):
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_idx)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    print(f'Device {device_idx} VRAM usage: {info.used>>30}/{info.total>>30} GB ({info.used/info.total:.2%})')
                print('Time elapsed:', time.time()-start_time)


    def _validation_loop(self, epoch):
        loss_test, loss_align, loss_f = 0, 0, 0
        self.eval_mode()

        with torch.no_grad():
            iters_test = 0

            for val_idx, batch in enumerate(self.val_dataloader):
                self.optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(self.device) for b in batch[1:]]
                    texts, input_lengths, _, _, mels, mel_input_length, _ = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2**self.n_down)).to(self.device)
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = self.model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_st = mask_from_lens(
                            s2s_attn, input_lengths,
                            mel_input_length // (2**self.n_down)
                        )
                        s2s_attn_mono = maximum_path(s2s_attn, mask_st)

                        # encode
                        t_en = self.model.text_encoder(texts, input_lengths, text_mask)
                        asr = t_en @ s2s_attn_mono

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    for idx, m in enumerate(mel_input_length):
                        mel_length = int(m.item())
                        mel = mels[idx, :, :m]
                        s = self.model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = self.model.style_encoder(mel.unsqueeze(0).unsqueeze(1))

                    # JMa: Fix: remove explicitly 2nd dimension
                    # otherwise all dimensions of size 1 are removed
                    # (resulting in error when current batch size is 1)
                    # s = torch.stack(ss).squeeze()
                    s = torch.stack(ss).squeeze(dim=1)

                    bert_dur = self.model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = self.model.predictor(
                        d_en,
                        s,
                        input_lengths,
                        s2s_attn_mono,
                        text_mask
                    )
                    # get clips
                    mel_len = int(mel_input_length.min().item()/2 - 1)

                    en, gt, p_en, wav = [], [], [], []
                    for idx, (m, w) in enumerate(zip(mel_input_length, waves)):
                        mel_length = int(m.item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[idx, :, random_start:random_start+mel_len])
                        p_en.append(p[idx, :, random_start:random_start+mel_len])

                        gt.append(mels[idx, :, (random_start * 2):((random_start+mel_len) * 2)])
                        y = w[(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(self.device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()
                    s = self.model.predictor_encoder(gt.unsqueeze(1))

                    f0_fake, n_fake = self.model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for idx in range(_s2s_trg.shape[0]):
                            _s2s_trg[idx, :_text_input[idx]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(
                            _dur_pred[1:_text_length-1],
                            _text_input[1:_text_length-1]
                        )

                    loss_dur /= texts.size(0)
                    s = self.model.style_encoder(gt.unsqueeze(1))

                    y_rec = self.model.decoder(en, f0_fake, n_fake, s)
                    loss_mel = self.stft_loss(y_rec.squeeze(), wav.detach())
                    f0_real, _, _ = self.model.pitch_extractor(gt.unsqueeze(1))
                    loss_f0 = F.l1_loss(f0_real, f0_fake) / 10
                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_f0).mean()

                    # Generate validation sample (up to the defined number)
                    if self.save_val_audio and val_idx < self.n_val_audios:
                        self._create_val_sample(
                            val_idx,
                            epoch,
                            mel_input_length,
                            mels,
                            asr,
                            p,
                            idx_in_batch=0,
                        )
                    # Generate ground-truth sample only at the beginning
                    if epoch == 0 and val_idx < self.n_val_audios and self.save_val_audio:
                        self._create_gt_sample(
                            val_idx,
                            epoch,
                            waves,
                            idx_in_batch=0,
                        )

                    iters_test += 1

                except Exception as e:
                    print(f"[!] Error: {e}")
                    continue    # skip batch

        avg_loss_test = loss_test/iters_test
        avg_dur_loss = loss_align/iters_test
        avg_f_loss = loss_f/iters_test
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f',
                    avg_loss_test, avg_dur_loss, avg_f_loss)
        self.writer.add_scalar('eval/mel_loss', avg_loss_test, epoch+1)
        self.writer.add_scalar('eval/dur_loss', avg_dur_loss, epoch+1)
        self.writer.add_scalar('eval/F0_loss', avg_f_loss, epoch+1)
        attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
        self.writer.add_figure('eval/attn', attn_image, epoch+1)

        return loss_test, iters_test


    # Create 1 validation sample for the validation batch (typically the first one (0))
    # Skipping if desired number of validation samples was reached
    def _create_val_sample(self,
                           val_idx,
                           epoch,
                           mel_input_length,
                           mels,
                           asr,
                           p,
                           idx_in_batch=0):
        # Compute for the whole utterance
        mel_length = int(mel_input_length[idx_in_batch].item())
        gt = mels[idx_in_batch, :, :mel_length].unsqueeze(0)
        en = asr[idx_in_batch, :, :mel_length // 2].unsqueeze(0)
        s = self.model.style_encoder(gt.unsqueeze(1))

        # Use real characteristics
        # f0_real, _, _ = self.model.pitch_extractor(gt.unsqueeze(1))
        # f0_real = f0_real.unsqueeze(0)
        # real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
        # y_rec = self.model.decoder(en, f0_real, real_norm, s)

        # Use predicted characteristics
        p_en = p[idx_in_batch, :, :mel_length // 2].unsqueeze(0)
        f0_fake, n_fake = self.model.predictor.F0Ntrain(p_en, s)
        y_rec = self.model.decoder(en, f0_fake, n_fake, s)

        # Write and save val audio
        wav = y_rec.cpu().numpy().squeeze()
        self.writer.add_audio('eval/y' + str(val_idx), wav, epoch+1, sample_rate=self.sr)
        if (epoch+1) % self.save_freq == 0:
            outfile_template = f'epoch_2nd_{epoch+1:0>5}'
            out_file = f'{outfile_template}_val-{val_idx}.wav'
            scipy.io.wavfile.write(
                filename=os.path.join(self.test_audio_dir, out_file),
                rate=self.sr,
                data=wav,
            )

    # Create 1 ground-truth sample for the validation batch (typically the first one (0))
    # Skipping if desired number of validation samples was reached
    # Do it only once at the very beginning (epoch 0)
    def _create_gt_sample(self, val_idx, epoch, waves, idx_in_batch=0):
        wav = waves[idx_in_batch].squeeze()
        self.writer.add_audio('gt/y' + str(val_idx), wav, epoch+1, sample_rate=self.sr)
        outfile_template = f'epoch_2nd_{epoch+1:0>5}'
        out_file = f'{outfile_template}_gt-{val_idx}.wav'
        scipy.io.wavfile.write(
            filename=os.path.join(self.test_audio_dir, out_file),
            rate=self.sr,
            data=wav,
        )


    def _save_progress(self, epoch, loss_test, iters_test):
        curr_loss = loss_test / iters_test
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
        state = {
            'net':  {key: self.model[key].state_dict() for key in self.model}, 
            'optimizer': self.optimizer.state_dict(),
            'iters': self.iters,
            'val_loss': loss_test / iters_test,
            'epoch': epoch,
        }
        # Save model
        save_checkpoint(state, '2nd', epoch, self.log_dir, self.max_saved_models)

        # if estimate sigma, save the estimated sigma
        if self.config['model_params']['diffusion']['dist']['estimate_sigma_data']:
            self.config['model_params']['diffusion']['dist']['sigma_data'] = \
                float(np.mean(self.running_std))

            with open(
                osp.join(self.log_dir, osp.basename(self.config_path)),
                'w',
                encoding='utf-8'
            ) as outfile:
                yaml.dump(self.config, outfile, default_flow_style=True)

    # # Save model
    # def _save_model(self, path, epoch, loss_test, iters_test):
    #     # Prepare model state fo saving
    #     state = {
    #         'net':  {key: self.model[key].state_dict() for key in self.model}, 
    #         'optimizer': self.optimizer.state_dict(),
    #         'iters': self.iters,
    #         'val_loss': loss_test / iters_test,
    #         'epoch': epoch,
    #     }
    #     torch.save(state, path)

    @property
    def log_dir(self):
        return self.config['log_dir']

    @property
    def batch_size(self):
        return self.config.get('batch_size', 10)

    @property
    def model_params(self):
        return self.config['model_params']

    @property
    def multispeaker(self):
        return self.config['model_params']['multispeaker']

    @property
    def loss_params(self):
        return self.config['loss_params']
    
    @property
    def preprocess_params(self):
        return self.config['preprocess_params']

    @property
    def diff_epoch(self):
        return self.config['loss_params']['diff_epoch']
    @diff_epoch.setter
    def diff_epoch(self, value):
        self.config['loss_params']['diff_epoch'] = value

    @property
    def joint_epoch(self):
        return self.config['loss_params']['joint_epoch']
    @joint_epoch.setter
    def joint_epoch(self, value):
        self.config['loss_params']['joint_epoch'] = value

    @property
    def epochs(self):
        return self.config.get('epochs', 200)
    @epochs.setter
    def epochs(self, value):
        self.config['epochs'] = value

    @property
    # Check if pretrained model for the 2nd stage exists
    def load_pretrained_for_stage2(self):
        return self.config.get('pretrained_model', '') != '' \
            and self.config.get('second_stage_load_pretrained', False)

    @property
    def sr(self):
        return self.config['preprocess_params'].get('sr', 24000)

    @property
    def save_freq(self):
        return self.config.get('save_freq', 2)

    @property
    def max_saved_models(self):
        return self.config.get('max_saved_models', 2)

    @property
    def log_interval(self):
        return self.config.get('log_interval', 10)

    # data_params = config.get('data_params', None)
    # sr = config['preprocess_params'].get('sr', 24000)

    @property
    def train_path(self):
        return self.config['data_params']['train_data']

    @property
    def val_path(self):
        return self.config['data_params']['val_data']

    @property
    def root_path(self):
        return self.config['data_params']['root_path']

    @property
    def min_length(self):
        return self.config['data_params']['min_length']

    @property
    def ood_data(self):
        return self.config['data_params']['OOD_data']

    @property
    def save_val_audio(self):
        return self.config['data_params'].get('save_val_audio', False)
    
    @property
    def n_val_audios(self):
        return self.config['data_params'].get('n_val_audios', 5)

    @property
    def save_test_audio(self):
        return self.config['data_params'].get('save_test_audio', False)

    @property
    def test_sentences(self):
        return self.config['data_params'].get('test_sentences', [])

    @property
    def test_audio_dir(self):
        return os.path.join(
            self.config['log_dir'],
            self.config['data_params'].get('test_audio_dir', 'test_audios')
    )

    @property
    def max_len(self):
        return self.config.get('max_len', 200)

    # JMa: gradient clipping support
    @property
    def grad_clip(self):
        return self.config.get('grad_clip', None)

    @property
    def steps_per_epoch(self):
        return len(self.train_dataloader)

    @property
    # Total number of steps given the batch size
    def tot_num_steps(self):
        return len(self.train_list) // self.batch_size

    @property
    def n_down(self):
        return self.model.text_aligner.n_down

    @property
    def slmadv_params(self):
        return self.config['slmadv_params']
