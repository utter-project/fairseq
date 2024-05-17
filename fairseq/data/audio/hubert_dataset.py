# Additional functions for multilingual batching with upsampling and for loading from multiple sources
# mHubertDataset class with numpy storing of intermediate variables, flags for checking alignment and new ordered_indices function
# Modified by Marcely Zanon Boito, 2023
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import time
import sys
import glob
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
)
import io

logger = logging.getLogger(__name__)


def load_audios(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    
    flat_names, all_inds, flat_sizes, flat_langs, flat_datasets, all_tot = [], [], [], [], [], []
    paths = glob.glob(manifest_path + "/*.tsv")

    id_names = [path.split("/")[-1].split(".tsv")[0] for path in paths]
    for path in paths:
        names, inds, tot, sizes, langs, datasets = load_audio(path, max_keep, min_keep)
        flat_names += names #file name
        flat_sizes +=  sizes #frames between [min_keep, max_keep]
        flat_langs += langs #language information
        flat_datasets += datasets #dataset information
        all_inds.append(inds) #store indexes per file
        all_tot.append(tot) #saves file length for comparison with .km file

    #print([len(info) for info in [flat_names, all_inds, flat_sizes, flat_langs, flat_datasets, all_tot]])
    #root is always the same, so just get the last one
    return "", flat_names, all_inds, all_tot, flat_sizes, flat_langs, flat_datasets, id_names

#removed root from load_audio
def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes, langs, datasets = [], [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) >= 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(root + items[0])
                inds.append(ind)
                sizes.append(sz)
                if len(items) > 2:
                    langs.append(items[2])
                if len(items) > 3:
                    datasets.append(items[3])
    tot = ind + 1
    logger.info(
        (
            f"loaded {manifest_path},"
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return names, inds, tot, sizes, langs, datasets

def load_labels(label_files, inds, tot, offset=False):
    labels = []
    assert len(label_files) == len(inds) and len(inds) == len(tot)
    for i in range(len(label_files)):
        if offset:
            labels.append(load_label_offset(label_files[i], inds[i], tot[i]))
            
        else:
            labels += load_label(label_files[i], inds[i], tot[i])
    if offset:
        return None, labels
    return labels, None

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels

def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets

def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        #this should filter out invalid inds
        lengths = [lengths[i] for i in inds]
    
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            exit(1)
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )
        exit(1)

def get_language_distribution(sizes, language_indexing, upsampling_factor):
    prob_dictionary = dict()

    #get duration for lang
    for lang in language_indexing:
        prob_dictionary[lang] = len(language_indexing[lang])

    total = len(sizes)
    for lang in prob_dictionary:
        prob_dictionary[lang] = (prob_dictionary[lang]/total)**upsampling_factor

    #normalization
    probs_sum = sum([prob_dictionary[lang] for lang in prob_dictionary])
    for lang in prob_dictionary:
        prob_dictionary[lang] /= probs_sum 

    return prob_dictionary

def get_dataset_distribution(dataset_indexing, upsampling_factor):
    prob_dictionary = dict()

    #get duration for (lang, dataset)
    for lang in dataset_indexing:
        for dataset in dataset_indexing[lang]:
            if lang not in prob_dictionary:
                prob_dictionary[lang] = dict()
            prob_dictionary[lang][dataset] = len(dataset_indexing[lang][dataset])

    for lang in prob_dictionary:
        #language total
        total = sum([prob_dictionary[lang][dataset] for dataset in prob_dictionary[lang]])
        for dataset in prob_dictionary[lang]:
            #sum of durations becomes probability
            prob_dictionary[lang][dataset] = (prob_dictionary[lang][dataset]/total)**upsampling_factor

    #normalization
    for lang in prob_dictionary:
        probs_sum = sum([prob_dictionary[lang][dataset] for dataset in prob_dictionary[lang]])
        for dataset in prob_dictionary[lang]:
            prob_dictionary[lang][dataset] /= probs_sum 

    return prob_dictionary

def get_dataset_indexing(langs, datasets):
    dictionary = dict()
    assert len(langs) == len(datasets)
    for i in range(len(langs)):
        lang = langs[i]
        dataset = datasets[i]
        if not lang in dictionary:
            dictionary[lang] = dict()
        if not dataset in dictionary[lang]:
            dictionary[lang][dataset] = list()
        dictionary[lang][dataset].append(i)
    return dictionary

def get_language_indexing(langs):
    #saves general index order in language dictionary for sampling
    dictionary = dict()
    for i in range(len(langs)):
        key = langs[i]
        if not key in dictionary:
            dictionary[key] = list()
        dictionary[key].append(i)
    return dictionary

def load_variable_from_numpy(numpy_path):
    return np.load(numpy_path, allow_pickle=True)

def dump_variable_to_numpy(numpy_path, variable):
    np.save(numpy_path, variable)


class HubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes, self.langs = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )

        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )

        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        _path, slice_ptr = parse_path(wav_path)
        if len(slice_ptr) == 0:
            wav, cur_sample_rate = sf.read(_path)
        else:
            assert _path.endswith(".zip")
            data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            f = io.BytesIO(data)
            wav, cur_sample_rate = sf.read(f)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index, label_idx):
        def load_offset():
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)
                return label
        
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            try:
                label = load_offset()
            except OSError:
                #scratch instability fixer
                time.sleep(0.1)
                label = load_offset()

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source": collated_audios, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1.0:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav


class mHubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        language_upsampling_factor: float,
        dataset_upsampling_factor: float,
        manifest_numpy_file: str,
        label_numpy_file: str,
        label_suffix: Optional[str] = ".km",
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        check_alignment: bool = False,
    ):
        

        try:
            logger.info(f"trying to load existing manifest variables from:{manifest_numpy_file}")
            self.audio_root, self.audio_names, inds, tot, self.sizes, self.langs, self.datasets, id_names = load_variable_from_numpy(manifest_numpy_file)
            logger.info(f"successfuly loaded the manifest variables from numpy file")
        except FileNotFoundError:
            logger.info(f"Could not find manifest numpy, creating manifest variables from scratch")
            self.audio_root, self.audio_names, inds, tot, self.sizes, self.langs, self.datasets, id_names = load_audios(
                manifest_path, max_keep_sample_size, min_keep_sample_size
            )
            logger.info(f"saving manifest variables at:{manifest_numpy_file}")
            dump_variable_to_numpy(manifest_numpy_file, (self.audio_root, self.audio_names, inds, tot, self.sizes, self.langs, self.datasets, id_names))

        self.language_upsampling_factor = language_upsampling_factor
        self.dataset_upsampling_factor = dataset_upsampling_factor
        self.sample_rate = sample_rate

        self.language_index = get_language_indexing(self.langs)
        self.dataset_index = get_dataset_indexing(self.langs, self.datasets)

        self.language_distribution = get_language_distribution(self.sizes, self.language_index, self.language_upsampling_factor)
        self.dataset_distribution = get_dataset_distribution(self.dataset_index, self.dataset_upsampling_factor)

        self.shuffle = shuffle
        self.random_crop = random_crop
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.store_labels = store_labels

        #avoids costly glob + makes sure tot is aligned with labels
        label_files = [label_paths + "/" + id_name + "." + label_suffix for id_name in id_names]  

        # this is a bit hard-coded: we have several label_files, but only one single dictionary for everything
        #final interpretation of this parameter: this was used at some point to allow several different labels per audio sample
        self.num_labels = 1 #len(label_files) 
        
        self.label_rates = (
            [label_rates for _ in range(len(label_files))]
            if isinstance(label_rates, float)
            else label_rates
        )
        
        self.label_paths = label_files

        #adding loading of numpy offsets due to slow loading for massive datasets
        try:
            logger.info(f"trying to load existing label (or offset) list from:{label_numpy_file}")
            self.label_list, self.label_offsets_list = load_variable_from_numpy(label_numpy_file)
            logger.info(f"successfuly loaded the labels (or offsets) from numpy file")
        except FileNotFoundError:
            logger.info(f"Could not find existing label (or offset) numpy, creating label (or offsets) list from scratch")
            self.label_list, self.label_offsets_list = load_labels(label_files, inds, tot, offset=(not self.store_labels))
            logger.info(f"saving label offsets list at:{label_numpy_file}")
            dump_variable_to_numpy(label_numpy_file, (self.label_list, self.label_offsets_list))

        if check_alignment:
            for i in range(len(label_files)):
                this_tot = tot[i]
                removed_indices = this_tot - len(inds[i])
                shift = sum([tot[y] - (tot[y] - len(inds[y])) for y in range(0,i)]) if i > 0 else 0
                
                chunk_end = shift + this_tot - removed_indices 
                this_audio_size_chunk = self.sizes[shift:chunk_end]

                logger.info(
                f"verifying label length match to audio frames={label_files[i]}"
                )
                verify_label_lengths(
                    this_audio_size_chunk, sample_rate, label_files[i], self.label_rates[i], inds[i], this_tot
                )


        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        _path, slice_ptr = parse_path(wav_path) #necessary because sometimes we have a zip index instead of file

        #this is necessary because sometimes os.path.join bugs on audio_names starting with "/"
        if self.audio_root not in _path:
            _path = self.audio_root + "/" + wav_path
        if len(slice_ptr) == 0:
            wav, cur_sample_rate = sf.read(_path)
        else:
            assert _path.endswith(".zip")
            data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            f = io.BytesIO(data)
            wav, cur_sample_rate = sf.read(f)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def translate_label_index(self, index):
        #offset case only
        labels_length_list = [len(labels) for labels in self.label_offsets_list]
        
        new_index = index
        for i in range(len(labels_length_list)):
            #sums everything that came before
            if i > 0:
                labels_length_list[i] += labels_length_list[i-1]
            
            if index < labels_length_list[i]:
                if i > 0:
                    new_index = index - labels_length_list[i-1]
                return i, new_index
            
    def get_label(self, index, label_idx):
        def load_offset():
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)
                return label
        
        if self.store_labels:
            label = self.label_list[index]
        else:
            old_index = index
            label_idx, index = self.translate_label_index(index)
            try:
                label = load_offset()
            except OSError:
                #scratch instability fixer
                time.sleep(0.1)
                label = load_offset()

        if self.label_processors is not None:
            #only one label processor
            label = self.label_processors[0](label)
        
        index = index if self.store_labels else old_index
        assert abs(len(label)/self.label_rates[0] - self.sizes[index]/self.sample_rate) < 0.1
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        try:
            wav = self.get_audio(index)
        except Exception:
            logger.info(f"Failed to load audio file {self.audio_names[index]}")
            exit(1)
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source": collated_audios, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1.0:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        '''
        This function produces "ordered" indices list considering upsampling factors for language and dataset
        This function considers that get_batch_iterator is updating the random seed before call as in the code below
        with data_utils.numpy_seed(seed + epoch):
                indices = dataset.ordered_indices()
        '''

        #original behavior sorted by length when both upsampling factors are equal to one
        if self.language_upsampling_factor == self.dataset_upsampling_factor and self.dataset_upsampling_factor == 1:   
            if self.shuffle:
                order = [np.random.permutation(len(self))]
            else:
                order = [np.arange(len(self))]

            order.append(self.sizes)
            return np.lexsort(order)[::-1]

        #creating batches sampling languages and datasets
        epoch_indices = np.empty(0, int)

        def sample_from_list(index_list, l_number_samples):
            indices = np.empty(0, int)
            #if downsampling, just shuffle and crop
            if l_number_samples <= len(index_list):
                indices = np.random.permutation(index_list)[:l_number_samples]
            #if upsampling
            else:
                while l_number_samples >= len(index_list):
                    #shuffle and concatenate 
                    permuted = np.random.permutation(index_list)
                    indices  = np.append(indices, permuted)
                    l_number_samples -= len(index_list)
                if l_number_samples > 0:
                    #then generate random permutation for the rest
                    new_indices = np.random.choice(index_list, l_number_samples, replace=False)
                    indices = np.append(indices, new_indices)
            return indices

        #sampling by the length of the dataset (some examples will not be seen during one given epoch)
        n_samples = len(self)
        language_samples = np.random.choice(list(self.language_distribution.keys()), n_samples, p=list(self.language_distribution.values()))
        #for each language, get the datasets to drawn from
        for language in self.language_distribution.keys():
            #number of samples in epoch
            l_number_samples = np.count_nonzero(language_samples == language)
            #if language was drawn
            if l_number_samples > 0:
                #get the dataset distribution
                dataset_samples = np.random.choice(list(self.dataset_distribution[language].keys()), l_number_samples, p=list(self.dataset_distribution[language].values()))
                #for each dataset
                language_indices = np.empty(0, int)
                for dataset in self.dataset_distribution[language].keys():
                    l_number_samples_dataset = np.count_nonzero(dataset_samples == dataset)
                    if l_number_samples_dataset > 0:
                        language_indices = np.append(language_indices, sample_from_list(self.dataset_index[language][dataset], l_number_samples_dataset))
                assert len(language_indices) == l_number_samples
                epoch_indices = np.append(epoch_indices, np.random.permutation(language_indices))
        
        assert len(epoch_indices) == n_samples
        
        #mixes the languages
        epoch_indices = np.random.permutation(epoch_indices)

        lengths = [self.sizes[index] for index in epoch_indices]

        sorted_indices = [x for _, x in sorted(zip(lengths, epoch_indices))]

        return sorted_indices

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
