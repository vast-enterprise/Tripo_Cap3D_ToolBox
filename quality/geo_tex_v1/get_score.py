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
    parser.add_argument("--output", type=str, default="output.txt")
    return parser.parse_args()

args = parse_args()


class SimpleDataset(Dataset):
    def __init__(self, uuids: List[str], transform=None,
            bucket_name="vast-data-platform-render"):
        self.uuids = uuids
        self.transform = transform
        self.bos_client = None
        # self.bucket_name = "vast-data-platform-render"
        self.bucket_name = bucket_name
        if os.path.exists("fail.uuids"):
            self.fail_uuids = [line.strip()
                               for line in open("fail.uuids", "r")]
        else:
            self.fail_uuids = []
        self.fail_dict = {k: 1 for k in self.fail_uuids}

    def init_client(self):
        if self.bos_client is not None:
            return
        else:
            bos_endpoint = "bj.bcebos.com"
            access_key_id = "XXXXX"
            secret_access_key = "XXXX"
            config = BceClientConfiguration(
                credentials=BceCredentials(
                    access_key_id=access_key_id,
                    secret_access_key=secret_access_key
                ),
                endpoint=bos_endpoint
            )
            self.bos_client = BosClient(config)

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        self.init_client()
        uuid = self.uuids[idx]
        images = []
        if uuid in self.fail_dict:
            return [torch.zeros(3, 378, 378)] * 4, idx
        for i in range(4):
            object_key = f"4view/{uuid}/render_000{i}.webp" if self.bucket_name == "vast-data-platform-render" else f"objaverse/{uuid[:2]}/{uuid}/render_000{i+4}.webp"
            try:
                response = self.bos_client.get_object(
                    self.bucket_name, object_key)
                image_byte = response.data
                image = Image.open(image_byte).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                images.append(image)
                response.data.close()
            except Exception as e:
                print(
                    f"Error: {e} {object_key} failed to load. Retry {i+1} times。")
                with open("error.log", "a") as f:
                    f.write(f"{object_key} failed\n")
                f.close()
                if self.transform:
                    images.append(torch.zeros(3, 378, 378))
                else:
                    images.append(Image.new("RGB", (378, 378)))
        return images, idx


class SequentialDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with `torch.nn.parallel.DistributedDataParallel`. In such a case, each process can pass a
    DistributedSampler instance as a DataLoader sampler, and load a subset of the original dataset that is exclusive to it.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        # Add extra samples to make it evenly divisible
        indices = self.indices[:self.total_size]
        # Subsample
        indices = indices[self.rank *
                          self.num_samples:(self.rank+1)*self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    result = []
    idxs = []
    for images, idx in batch:
        result.extend(images)
        idxs.append(idx)
    if isinstance(result[0], torch.Tensor):
        result = torch.stack(result)
    return result, idxs


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


def get_feature(image_paths: List[str],
                clip_model,
                preprocess=None,
                device="cuda",
                bucket_name="vast-data-platform-render"
                ):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            dataset = SimpleDataset(image_paths, preprocess, bucket_name)
            sampler = SequentialDistributedSampler(dataset)
            data_loader = DataLoader(
                dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn, sampler=sampler)
            features_all = []
            idx = 0
            for images, idxs in tqdm(data_loader):
                images = images.cuda(non_blocking=True).half()
                features = clip_model(images, idx)
                features_all.append(features.cpu().numpy())
                idx += 1
            features_all = np.concatenate(features_all, axis=0) 
            features_all = features_all.reshape(-1, 4 * features.shape[-1])
    print(f"features shape: {features_all.shape}")
    return features_all


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
    preds = (preds > 3).astype(np.int64)
    labels = (labels > 3).astype(np.int64)
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


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            args.clip_model)

    def forward(self, x, iteration=0):
        return self.model.encode_image(x)


def load_data(filename: str, with_label=True) -> List[str]:
    uuids = []
    if with_label:
        geo_scores = []
        tex_scores = []
        for uuid, geo_score, tex_score in [line.strip().split() for line in open(filename, 'r')]:
            uuids.append(uuid)
            geo_scores.append(float(geo_score))
            tex_scores.append(float(tex_score))
        return uuids, geo_scores, tex_scores

    elif filename.endswith(".json"):
        with open(filename, "r") as f:
            for line in f:
                uuids.append(json.loads(line.strip())["model_id"])
        return uuids
    else:
        with open(filename, "r") as f:
            for line in f:
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
            clip_model = SimpleModel()
            clip_model = clip_model.to(device)
            clip_model = DDP(clip_model, device_ids=[
                             local_rank], output_device=local_rank)
            if os.path.exists("train.feature.bin") and os.path.exists("val.feature.bin"):
                train_features = torch.load("train.feature.bin", map_location=device).reshape(-1, 1024 * 4)
                val_features = torch.load("val.feature.bin", map_location=device).reshape(-1, 1024 * 4)
            else:
                train_features = get_feature(train_uuids,
                                            clip_model,
                                            preprocess=clip_model.module.preprocess,
                                            device=device,
                                            bucket_name="v3-render")

                val_features = get_feature(val_uuids,
                                        clip_model,
                                        preprocess=clip_model.module.preprocess,
                                        device=device,
                                        bucket_name="v3-render")
                torch.save(train_features, "train.feature.bin")
                torch.save(val_features, "val.feature.bin")

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
            test_features = get_feature(test_uuids,
                                        clip_model,
                                        preprocess=clip_model.module.preprocess,
                                        device=device,
                                        bucket_name="v3-render"
                                        )
            geo_preds = predict(reg_geo, pca_geo, test_features, device)
            tex_preds = predict(reg_tex, pca_tex, test_features, device)

    # Save
    with open(args.output, "w") as f:
        for uuid, geo_pred, tex_pred in zip(test_uuids, geo_preds, tex_preds):
            f.write(f"{uuid} {geo_pred} {tex_pred}\n")
    f.close()

if __name__ == "__main__":
    main()
