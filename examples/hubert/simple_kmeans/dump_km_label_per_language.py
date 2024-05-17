# Based on dump_km_label.py
# Modified by Marcely Zanon Boito, 2023
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import glob

import numpy as np

import joblib
import torch
import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(feat_path, leng_path):
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)


def dump_label(feat_dir, km_path, lab_dir):
    apply_kmeans = ApplyKmeans(km_path)
    
    languages = glob.glob(feat_dir + "*.npy")
    print(languages)
    #exit(1)
    for lang in languages:
        generator, num = get_feat_iterator(lang, lang.replace(".npy",".len"))
        iterator = generator()
        file_name = lang.split("/")[-1].replace(".npy",".km")
        lab_path = lab_dir + "/" + file_name
        os.makedirs(lab_dir, exist_ok=True)
        logger.info("writing " + lab_path)
        with open(lab_path, "w") as f:
            for feat in tqdm.tqdm(iterator, total=num):
                lab = apply_kmeans(feat).tolist()
                try:
                    f.write(" ".join(map(str, lab)) + "\n")
                except TypeError:
                    print(lang)
                    print(feat)
                    print(lab)
                    exit(1)
        logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir")
    parser.add_argument("km_path")
    parser.add_argument("lab_dir")
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(**vars(args))
