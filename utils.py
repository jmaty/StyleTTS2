import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from monotonic_align.core import maximum_path_c
from munch import Munch
from nltk.tokenize import word_tokenize

from Modules.diffusion.sampler import (ADPM2Sampler, DiffusionSampler,
                                       KarrasSchedule)
from text_utils import TextCleaner


def maximum_path(neg_cent, mask):
    """ Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    if isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d

def log_print(message, logger):
    logger.info(message)
    print(message)

# JMa: Infere a single sentence written in phonemes
def inference(sentence,
              model,
              textcleaner,
              sampler,
              noise,
              diffusion_steps=5,
              embedding_scale=1,
              phonemizer=None,
              device='cuda'):
    # Phonemize text if phonemizer is given;
    # otherwise phonemes expected at the input
    if phonemizer:
        sentence = sentence.strip()
        ps = phonemizer.phonemize([sentence])
    ps = word_tokenize(sentence)
    ps = ' '.join(ps)
    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise,
                         embedding=bert_dur[0].unsqueeze(0),
                         num_steps=diffusion_steps,
                         embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                            f0_pred, n_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()

# JMa: Synthesize test files written in phonemes
def synth_test_files(model,
                     test_sentences,
                     outdir,
                     outfile_template,
                     sr,
                     sampler=None,
                     diffusion_steps=5,
                     embedding_scale=1,
                     device='cuda'):
    textcleaner = TextCleaner()
    # Generate noise
    noise = torch.randn(1,1,256).to(device)
    # Set up sampler
    if not sampler:
        sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        # empirical parameters
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    for idx, snt in enumerate(test_sentences):
        wav = inference(snt,
                        model,
                        textcleaner,
                        sampler,
                        noise,
                        diffusion_steps,
                        embedding_scale,
                        device)
        outfile = f'{outfile_template}-{idx}.wav'
        scipy.io.wavfile.write(filename=os.path.join(outdir, outfile),
                               rate=sr,
                               data=wav,
        )

# JMa: Save model and delete old models
def save_model(model_state, stage, epoch, save_dir, max_saved_models=3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the model
    filename = f"epoch_{stage}_{epoch:05d}.pth"
    filepath = os.path.join(save_dir, filename)
    torch.save(model_state, filepath)
    print(f"New model saved to {filepath}")

    # Get list of all saved models and sort by epoch number
    saved_models = sorted(
        [f for f in os.listdir(save_dir) if f.startswith(f"epoch_{stage}") and f.endswith(".pth")],
        key=lambda x: int(x.split('_')[2].split('.')[0])
    )

    # Remove old models if exceeding max_saved_models
    while len(saved_models) > max_saved_models:
        old_model = saved_models.pop(0)
        os.remove(os.path.join(save_dir, old_model))
        print(f"Old model {old_model} removed")
