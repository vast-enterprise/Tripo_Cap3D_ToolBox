import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from tqdm import tqdm
import json
from sklearn.model_selection import StratifiedKFold 
from sklearn.svm import SVR
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

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

def is_json(input):
    try:
        json.loads(input)
    except ValueError:
        return False
    return True
 
def load_data(filename: str, with_label=True):
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


scorers_infos = json.load(open('geo_1w_202407.json'))
scorer_dict = {}
for line in scorers_infos:
    scorer_dict.setdefault(line['scorer'], []).append(line['source_id'])

def select_scorer(clip_train_features, train_data, extra_low=False):
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


def cal_good_bad_acc(scores, pred_scores):
    good_correct = 0
    good_total = 0
    bad_correct = 0
    bad_total = 0
    pred_scores = np.clip(pred_scores, 0, 5)
    
    for score, pred_score in zip(scores, pred_scores):
        if score >= 3:
            good_total += 1
            # if np.abs(score - pred_score) <= 0.5:
            #     good_correct += 1
            if pred_score >= 2.5:
                good_correct += 1
        else:
            bad_total += 1
            # if np.abs(score - pred_score) <= 0.5:
            #     bad_correct += 1
            if pred_score < 2.5:
                bad_correct += 1
    acc_good = good_correct / good_total if good_total > 0 else 0
    acc_bad = bad_correct / bad_total if bad_total > 0 else 0
    acc_all = (good_correct + bad_correct) / (good_total + bad_total) if (good_total + bad_total) > 0 else 0
    print('good_total:', good_total, 'bad_total:', bad_total)
    return acc_good, acc_bad, acc_all

clip_train_features = np.load('train_data_sketch.clip-378.txt.bin.npy').astype(np.float32)
clip_test_features = np.load('test_data_sketch.clip-378.txt.bin.npy').astype(np.float32)

# select users
train_data = torch.load('train_data.pt')
clip_train_features, train_data = select_scorer(clip_train_features, train_data, extra_low=True)
train_labels =[line[1] for line in train_data]
test_data = torch.load('test_data.pt')
test_labels = [line[1] for line in test_data]

# select label [1,5]
train_labels = np.array(train_labels)
low_index = np.where(train_labels <= 2)
print('low_index:', len(low_index[0]))

high_index = np.where(train_labels >= 5)
print('high_index:', len(high_index[0]))

clip_all_features = np.concatenate((clip_train_features, clip_test_features), axis=0)
all_labels = np.concatenate((train_labels, test_labels), axis=0)

print('clip_all_features:', clip_all_features.shape)
print('all_labels:', all_labels.shape)

def predict_model(train_features, train_labels, test_features,pca_num = 512):
    # PCA
    pca = PCA(n_components=pca_num)
    pca.fit(train_features)

    train_features_pca = pca.transform(train_features)
    # Linear Regression
    reg_linear = LinearRegression().fit(train_features_pca, train_labels)

    test_features_pca = pca.transform(clip_test_features)
    test_pred_linear = reg_linear.predict(test_features_pca)
    print('acc_linear:', cal_good_bad_acc(test_labels, test_pred_linear))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    components_torch = torch.from_numpy(pca.components_).to(device)
    mean_torch = torch.from_numpy(pca.mean_).to(device)

    reg_weight_torch = torch.from_numpy(reg_linear.coef_).to(device)
    reg_bias_torch = torch.from_numpy(np.array([reg_linear.intercept_])).to(device)

    results_linear = []
    results_svr = []
    for i in tqdm(range(0, len(test_features), 10000)):
        beg = i
        end = min(i+10000, len(test_features))
        
        batch_features = test_features[beg:end]
        batch_features = batch_features.astype(np.float32)
        batch_features = torch.from_numpy(batch_features).to(device)
        batch_features = batch_features - mean_torch
        batch_features = torch.matmul(batch_features, components_torch.mT)

        pred_linear_2 = torch.matmul(batch_features, reg_weight_torch.T) + reg_bias_torch
        pred_linear = pred_linear_2.cpu().numpy().tolist()
        results_linear.extend(pred_linear)

    return results_linear #, results_svr

all_data = load_data(args.input, with_label=False)
print('all_data:', len(all_data))
all_featuers = np.load(args.input_features, mmap_mode='r')
print('all_featuers:', all_featuers.shape)
all_results_linear = predict_model(clip_all_features, all_labels, all_featuers)

with open(args.output, 'w') as f:
    for line, result_linear in zip(all_data, all_results_linear):
        f.write(f'{line} {result_linear}\n')
f.close()
