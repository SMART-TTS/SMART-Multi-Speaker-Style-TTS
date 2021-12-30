import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import sys

import commons 
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text.text import TextProcessor, TagProcessor

sys.path.append('KoSentenceBERT_SKTBERT')
from sentence_transformers import SentenceTransformer


def get_spkrs(filename):
	spkrs = list()
	with open(filename, encoding='utf-8') as f:
		lines = f.readlines()
		for line in lines:
			fileloc, text, emo, tag = line.split('|')
			cur_spkr = fileloc.split('/')[6]
			if cur_spkr not in spkrs:
				spkrs.append(cur_spkr)
	return spkrs

ALL_spkr_ls = get_spkrs('filelists/[ALL_SPKRS.csv]')

class TextAudioTagLoader(torch.utils.data.Dataset):
    """
        1) loads audio, reference audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_text_emo_tag_sid, hparams, inf=False):
        self.audiopaths_text_emo_tag_sid = load_filepaths_and_text(audiopaths_text_emo_tag_sid)
        self.spkr_ls = ALL_spkr_ls
        self.max_wav_value = hparams.max_wav_value 
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.num_mels       = hparams.n_mel_channels
        self.fmin           = hparams.mel_fmin
        self.fmax           = hparams.mel_fmax

        self.get_text = TextProcessor()
        self.get_tag = TagProcessor()

        self.min_text_len = getattr(hparams, "min_text_len", 10)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        if not inf:
            self._filter()
        else:
            self._new_()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_text_emo_tag_sid_new = []
        lengths = []
        ctr = 0
        for audiopath, text, emo, tag in self.audiopaths_text_emo_tag_sid:
            spkr = audiopath.split('/')[6]
            sid = self.spkr_ls.index(spkr)
            spec_length = os.path.getsize(audiopath)//(2*self.hop_length)
            wav_length = os.path.getsize(audiopath)//2
            if spec_length > 32 and wav_length < 10*self.sampling_rate:
                if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                    audiopaths_text_emo_tag_sid_new.append([audiopath, text, emo, tag, sid])			# sid is appended at the end
                    lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
                else:
                    ctr += 1
            else:
                ctr += 1
        print("audio smaller than 32 frames or larger than 10 sec and text between 10 and 190:", str(ctr))
        self.audiopaths_text_emo_tag_sid = audiopaths_text_emo_tag_sid_new
        self.lengths = lengths

    def _new_(self):
        audiopaths_text_emo_tag_sid_new = []
        for audiopath, text, emo, tag in self.audiopaths_text_emo_tag_sid:
            spkr = audiopath.split('/')[6]
            sid = self.spkr_ls.index(spkr)
            audiopaths_text_emo_tag_sid_new.append([audiopath, text, emo, tag, sid])
            self.audiopaths_text_emo_tag_sid = audiopaths_text_emo_tag_sid_new

    def get_audio_text_emo_tag_pair(self, audiopath_text_emo_tag_sid):
        # separate filename, ref_filename and text
        audiopath, text,  emo, tag, sid = audiopath_text_emo_tag_sid[0], audiopath_text_emo_tag_sid[1], audiopath_text_emo_tag_sid[2], audiopath_text_emo_tag_sid[3], audiopath_text_emo_tag_sid[4]
        text = self.get_text.syllables_to_cjj(text)
        text = torch.LongTensor(torch.cat([torch.as_tensor(x) for x in text]))
        tag = self.get_tag.tag_augment(tag)
        mel, spec, wav = self.get_audio(audiopath)
        return (text, mel, spec, wav, tag, sid)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        assert sampling_rate == self.sampling_rate
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        mel = spec_to_mel_torch(spec, self.filter_length, self.num_mels, self.sampling_rate, self.fmin, self.fmax)

        return mel, spec, audio_norm	# c, t

    def __getitem__(self, index):
        return self.get_audio_text_emo_tag_pair(self.audiopaths_text_emo_tag_sid[index])

    def __len__(self):
        return len(self.audiopaths_text_emo_tag_sid)


class TextAudioTagCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids
        self.sbert = SentenceTransformer('KoSentenceBERT_SKTBERT/output/training_sts')

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and reference audio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, spec_ref_normalized, wav_ref_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)
        
        max_text_len = max([len(x[0]) for x in batch])
        max_mel_len = max([x[1].size(1) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_mel_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        sids = torch.LongTensor(len(batch))
       
        text_padded.zero_()
        mel_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        tag_ls = list()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            mel = row[1]
            mel_padded[i, :, :mel.size(1)] = mel	# b, c, t
            mel_lengths[i] = mel.size(1)

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec	# b, c, t
            spec_lengths[i] = spec.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            tag = row[4]
            tag_ls.append(self.sbert.tokenize(tag))
        
            sid = row[5]
            sids[i] = sid
        tag_info = self.sbert.my_collate(tag_ls)
        with torch.no_grad():
            tag_emb = self.sbert(tag_info)["sentence_embedding"].unsqueeze(1).clone().detach()
        tag_emb = torch.FloatTensor(tag_emb)
        
        if self.return_ids:
            return text_padded, text_lengths, mel_padded, mel_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, tag_emb, sids, ids_sorted_decreasing
        return text_padded, text_lengths, mel_padded, mel_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, tag_emb, sids


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)

      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))

      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]

          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]

          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)

      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches

      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1

      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
