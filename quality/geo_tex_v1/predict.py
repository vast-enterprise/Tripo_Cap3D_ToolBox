# -*- coding=utf-8
import os
os.environ["HF_HOME"] = "/mnt/pfs/share/pretrained_model/.cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = '1'
os.environ["DIFFUSERS_OFFLINE"] = '1'



from sklearn.decomposition import PCA
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
from baidubce.services.bos import storage_class
from baidubce.auth import bce_credentials
from typing import Iterable, List, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler
import open_clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import torch.nn as nn
import math
import aiofiles
import traceback
import torch
import json
import sys

TRAIN_FILE = "data/train_list.txt"
VAL_FILE = "data/val_list.txt"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", type=str,
                        default="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384")
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--input_features", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.txt")
    return parser.parse_args()

args = parse_args()

def cal_roc(pred, y):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, pred)
    # print(fpr, tpr, thresholds)
    for i in range(len(fpr)):
        if fpr[i] > 0.02 or (abs(fpr[i] - 0.02) < 0.003):
            print(f'fpr: {fpr[i]} tpr: {tpr[i]} threshold: {thresholds[i]}')
            break
    roc_auc = auc(fpr, tpr)
    print(f'roc auc: ', roc_auc)

def train(features, labels):
    # PCA
    pca = PCA(n_components=512)
    # features = features.cpu().numpy()
    pca.fit(features)
    features = pca.transform(features)
    features = torch.tensor(features)

    # Linear Regression
    model = LinearRegression()
    model.fit(features, labels)
    return model, pca


def evaluate(model, pca, features, labels):
    features = pca.transform(features)
    preds = model.predict(features)
    # MSE
    mse = np.mean((preds - labels) ** 2)
    print(f"MSE: {mse}")

    # AUC
    labels = np.array(labels)
    preds = (preds > 2).astype(np.int64)
    labels = (labels > 2).astype(np.int64)
    return cal_roc(preds, labels)


def predict(reg_model, pca, features, device):
    components_torch = torch.from_numpy(pca.components_).to(device)
    mean_torch = torch.from_numpy(pca.mean_).to(device)

    reg_weight_torch = torch.from_numpy(reg_model.coef_).to(device)
    reg_bias_torch = torch.from_numpy(
        np.array([reg_model.intercept_])).to(device)

    results_linear = []
    for i in tqdm(range(0, len(features), 10000)):
        beg = i
        end = min(i+10000, len(features))

        batch_features = features[beg:end]
        batch_features = batch_features.astype(np.float32)
        batch_features = torch.from_numpy(batch_features).to(device)

        # PCA
        batch_features = batch_features - mean_torch
        batch_features = torch.matmul(batch_features, components_torch.T)

        # Linear Regression
        pred_linear_2 = torch.matmul(
            batch_features, reg_weight_torch.T) + reg_bias_torch
        pred_linear = pred_linear_2.cpu().numpy().tolist()
        results_linear.extend(pred_linear)
    return results_linear

def is_json(input):
    try:
        json.loads(input)
    except ValueError:
        return False
    return True
    

def load_data(filename: str, with_label=True) -> List[str]:
    uuids = []
    if with_label:
        geo_scores = []
        tex_scores = []
        for uuid, geo_score, tex_score in [line.strip().split() for line in open(filename, 'r')]:
            uuids.append(uuid)
            geo_scores.append(float(geo_score)-1)
            tex_scores.append(float(tex_score)-1)
        return uuids, geo_scores, tex_scores

    elif filename.endswith(".jsonl"):
        with open(filename, "r") as f:
            for line in f:
                uuids.append(json.loads(line.strip())["model_id"])
        return uuids
    else:
        with open(filename, "r") as f:
            for line in f:
                if is_json(line):
                    uuids.append(json.loads(line.strip())["model_id"])
                else:
                    uuids.append(line.strip().split()[0])
        return uuids

def main():
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


    # Load model and data
    train_uuids, train_geo_scores, train_tex_scores = load_data(TRAIN_FILE)
    val_uuids, val_geo_scores, val_tex_scores = load_data(VAL_FILE)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            train_features = torch.load("train.feature.bin", map_location=device).reshape(-1, 1024 * 4)
            val_features = torch.load("val.feature.bin", map_location=device).reshape(-1, 1024 * 4)
    reg_geo, pca_geo = train(train_features, train_geo_scores)
    reg_tex, pca_tex = train(train_features, train_tex_scores)
    val_auc_geo = evaluate(
        reg_geo, pca_geo, val_features, val_geo_scores)
    val_auc_tex = evaluate(
        reg_tex, pca_tex, val_features, val_tex_scores)

    # Predict
    test_uuids = load_data(args.input, with_label=False)
    print(f"Predicting {len(test_uuids)} samples")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            test_features = np.load(args.input_features, mmap_mode='r')

            geo_preds = predict(reg_geo, pca_geo, test_features, device)
            tex_preds = predict(reg_tex, pca_tex, test_features, device)
    # Save
    with open(args.output, "w") as f:
        for uuid, geo_pred, tex_pred in zip(test_uuids, geo_preds, tex_preds):
            f.write(f"{uuid} {geo_pred} {tex_pred}\n")
    f.close()

if __name__ == "__main__":
    main()
