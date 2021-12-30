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
from text.symbols import hangul_symbols as symbols
from tqdm import tqdm
from scipy.io.wavfile import write
from pathlib import Path

hps = utils.get_hparams_from_file("./configs/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    use_ref=True,
    use_sdp=True,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/[model_name]", net_g, None)

outdir = Path('./sample')
outdir.mkdir(parents=True, exist_ok=True)

test_meta_loc = hps.data.testing_files
test_dataset = TextAudioTagLoader(test_meta_loc, hps.data, inf=True)
collate_fn = TextAudioTagCollate()
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

for batch_idx, (x, x_lengths, mel, mel_lengths, spec, spec_lengths, y, y_lengths, tag_emb, sid, spec_ref, spec_ref_lengths) in enumerate(test_loader, 1500):
    x, x_lengths = x.cuda(), x_lengths.cuda()
    spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
    tag_emb, sid = tag_emb.cuda(), sid.cuda()
    spec_ref, spec_ref_lengths = spec_ref.cuda(), spec_ref_lengths.cuda()

    audio_, _, _, (_, _, _, _) = net_g.infer(x, x_lengths, spec_ref, spec_ref_lengths, tag_emb, sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)
    audio = audio_.detach().data.cpu().float().numpy()
    write(os.path.join(outdir, 'SMART_TTSv2_MULTI_{}.wav'.format(batch_idx+1)), hps.data.sampling_rate, audio)

    print("Sample saved at ", os.path.join(outdir, '{}.wav'.format(batch_idx+1)))
