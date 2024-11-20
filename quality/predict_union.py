import os 
from typing import Iterable, List, Optional, Union
from sklearn.decomposition import PCA
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


QUALITY_DIR = '/mnt/pfs/share/yuzhipeng/Quality/'

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", type=str,
                        default="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384")
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--input_features", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.txt")
    return parser.parse_args()

def is_json(input):
    try:
        json.loads(input)
    except ValueError:
        return False
    return True

def load_data(filename: str, with_label=True, score_offset = 0) -> List[str]:
    uuids = []
    if with_label:
        geo_scores = []
        tex_scores = []
        for uuid, geo_score, tex_score in [line.strip().split() for line in open(filename, 'r')]:
            uuids.append(uuid)
            geo_scores.append(float(geo_score) + score_offset)
            tex_scores.append(float(tex_score)+ score_offset)
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

def predict(reg_model, pca, features, device='cuda'):
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

def geo_tex_v1(test_uuids, test_features, device='cuda'):

    TRAIN_FILE = f"{QUALITY_DIR}/data/v1/train_list.txt"
    VAL_FILE = f"{QUALITY_DIR}/data/v1/val_list.txt"        
    train_uuids, train_geo_scores, train_tex_scores = load_data(TRAIN_FILE, score_offset = -1)
    val_uuids, val_geo_scores, val_tex_scores = load_data(VAL_FILE, score_offset = -1)
    
    data_dict = {}
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            train_features = torch.load(f"{QUALITY_DIR}/data/v1/train.feature.bin", map_location=device).reshape(-1, 1024 * 4)
            val_features = torch.load(f"{QUALITY_DIR}/data/v1/val.feature.bin", map_location=device).reshape(-1, 1024 * 4)
            reg_geo, pca_geo = train(train_features, train_geo_scores)
            reg_tex, pca_tex = train(train_features, train_tex_scores)    
            
            geo_scores = predict(reg_geo, pca_geo, test_features, device)
            tex_scores = predict(reg_tex, pca_tex, test_features, device)
            for i, uuid in enumerate(test_uuids):
                data_dict[uuid] = [geo_scores[i], tex_scores[i]]
    return data_dict

def geo_v2(test_uuids, test_features):
    def select_scorer(clip_train_features, train_data, extra_low=False):
        scorers_infos = json.load(open(f'{QUALITY_DIR}/data/v2/geo_1w_202407.json'))
        scorer_dict = {}
        for line in scorers_infos:
            scorer_dict.setdefault(line['scorer'], []).append(line['source_id'])        
        selected_users = ['liangding','zouzixin','liyangguang','huangzehuan','liuyingtian','guoyuanchen']
        selected_source_ids = []
        for user in selected_users:
            selected_source_ids.extend(scorer_dict[user])
        selected_clip_train_features = []
        selected_train_data = []
        for i in tqdm(range(len(train_data))):
            if train_data[i][0] in selected_source_ids or (extra_low and train_data[i][1] <= 2):
                selected_clip_train_features.append(clip_train_features[i])
                selected_train_data.append(train_data[i])
        return np.array(selected_clip_train_features), selected_train_data


    clip_train_features = np.load(f'{QUALITY_DIR}/data/v2/train_data_sketch.clip-378.txt.bin.npy').astype(np.float32)
    clip_test_features = np.load(f'{QUALITY_DIR}/data/v2/test_data_sketch.clip-378.txt.bin.npy').astype(np.float32)

    train_data = torch.load(f'{QUALITY_DIR}/data/v2/train_data.pt')
    clip_train_features, train_data = select_scorer(clip_train_features, train_data, extra_low=True)
    train_labels =[line[1] for line in train_data]
    test_data = torch.load(f'{QUALITY_DIR}/data/v2/test_data.pt')
    test_labels = [line[1] for line in test_data]    

    # select label [1,5]
    train_labels = np.array(train_labels)
    low_index = np.where(train_labels <= 2)
    print('low_index:', len(low_index[0]))

    high_index = np.where(train_labels >= 5)
    print('high_index:', len(high_index[0]))


    clip_all_features = np.concatenate((clip_train_features, clip_test_features), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)

    reg_geo, pca_geo = train(clip_all_features, all_labels)
    geo_scores = predict(reg_geo, pca_geo, test_features)
    data_dict = {}
    for i, uuid in enumerate(test_uuids):
        data_dict[uuid] = [geo_scores[i]]
    return data_dict

def geo_v3(test_uuids, test_features):
    datas = json.load(open(f'{QUALITY_DIR}/data/v3/v2_all_datas.json'))
    clip_all_features = np.load(f'{QUALITY_DIR}/data/v3/v2_all_datas_clip_378_feature.json.npy').astype(np.float32)

    train_features = []
    train_labels = []

    for data_i, feature in zip(datas, clip_all_features):
        train_features.append(feature)
        train_labels.append(data_i['geo_score'])
    train_features = np.array(train_features)
    print('train_features:', train_features.shape)
    train_labels = np.array(train_labels)


    reg_geo, pca_geo = train(train_features, train_labels)
    geo_scores = predict(reg_geo, pca_geo, test_features)
    data_dict = {}
    for i, uuid in enumerate(test_uuids):
        data_dict[uuid] = [geo_scores[i]]
    return data_dict


def main():
    input_args = parse_args()
    input_features = np.load(input_args.input_features)
    test_uuids = load_data(input_args.input, with_label=False)
    data_dict_v1 = geo_tex_v1(test_uuids, input_features)
    data_dict_v2 = geo_v2(test_uuids, input_features)
    data_dict_v3 = geo_v3(test_uuids, input_features)

    with open(input_args.output, "w") as f:
        for uuid in test_uuids:
            f.write(f"{uuid} {data_dict_v1[uuid][0]} {data_dict_v1[uuid][1]} {data_dict_v2[uuid][0]} {data_dict_v3[uuid][0]}\n")
    f.close()
    
if __name__ == "__main__":
    main()





