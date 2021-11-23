import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from data_utils import TextAudioTagLoader, TextAudioTagCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from tqdm import tqdm
from scipy.io.wavfile import write
from pathlib import Path

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_audio(filename):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    if sampling_rate != hps.data.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, hps.data.sampling_rate))
    audio_norm = audio / hps.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
        center=False)
    spec = torch.squeeze(spec, 0)
    return spec, audio_norm

hps = utils.get_hparams_from_file("./configs/IITP.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    use_ref=True,
    use_sdp=True,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/0929/G_356000.pth", net_g, None)

outdir = Path('./outputs_small_same_text_ref_20/sample')
outdir.mkdir(parents=True, exist_ok=True)

refdir = Path('./outputs_small_same_text_ref_20/ref')
refdir.mkdir(parents=True, exist_ok=True)

test_meta_loc = hps.data.testing_files
test_dataset = TextAudioTagLoader(test_meta_loc, hps.data, inf=True)
collate_fn = TextAudioTagCollate()
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

for batch_idx, (x, x_lengths, mel, mel_lengths, spec, spec_lengths, y, y_lengths, tag_emb, sid) in enumerate(test_loader):
    x, x_lengths = x.cuda(), x_lengths.cuda()
    spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
    tag_emb, sid = tag_emb.cuda(), sid.cuda()

    audio_, _, _, (_, _, _, _) = net_g.infer(x, x_lengths, spec, spec_lengths, tag_emb, sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)
    audio = audio_.detach().data.cpu().float().numpy()
    write(os.path.join(outdir, 'SMART_TTSv2_MULTI_{}.wav'.format(batch_idx+1)), hps.data.sampling_rate, audio)
#    attn_fig = utils.plot_attn_to_numpy(attn.detach().squeeze().cpu().numpy())
#    attn_fig.savefig(os.path.join(outdir, 'sample/{}'.format(batch_idx)))

    print("Sample saved at ", os.path.join(outdir, '{}.wav'.format(batch_idx+1)))
    write(os.path.join(refdir, 'REF_MULTI_{}.wav'.format(batch_idx+1)), hps.data.sampling_rate, y.data.cpu().float().numpy())

#   recon = net_g.recon(spec, spec_lengths, spec, spec_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1).detach()
#   recon = recon.data.cpu().float().numpy()
#   print("audio", audio.shape, "y", y.shape, "recon", recon.shape)
#   write(os.path.join(outdir, 'recon/{}.wav'.format(batch_idx)), hps.data.sampling_rate, recon)
   

'''
    y_hat_mel = mel_spectrogram_torch(audio_.squeeze(1), hps.data.filter_length,
			hps.data.n_mel_channels,
			hps.data.sampling_rate,
			hps.data.hop_length,
			hps.data.win_length,
			hps.data.mel_fmin,
			hps.data.mel_fmax)
    _, mel_fig = utils.plot_spectrogram_to_numpy(y_hat_mel.squeeze().cpu().numpy())
    mel_fig.savefig(f'{outdir}/fig/{batch_idx}')


with open(test_meta_loc, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
    for i, line in tqdm(enumerate(lines)):
        _, ref_wav_loc, text = line.split('|')
        ref_mel, ref_wav = get_audio(ref_wav_loc)
        stn_tst = get_text(text, hps)
        #stn_tst = get_text("VITS is Awesome!", hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            ref = ref_mel.cuda().unsqueeze(0)
            ref_lengths = torch.LongTensor([ref_mel.size(0)]).cuda()
#            print("x length", x_tst_lengths, "ref_length", ref_lengths)
#            sid = torch.LongTensor([4]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, ref, ref_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            write(os.path.join(outdir, '{}.wav'.format(i)), hps.data.sampling_rate, audio)
#ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
'''
