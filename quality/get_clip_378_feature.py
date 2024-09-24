# -*- coding=utf-8
import aiofiles
from baidubce.auth import bce_credentials
from baidubce.services.bos import storage_class
from baidubce.services.bos.bos_client import BosClient
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
import traceback

import json
import os
import sys
from typing import Iterable, List, Optional, Union
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import open_clip
import time

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler
import math

class SimpleDataset(Dataset):
    def __init__(self, uuids: List[str], transform=None):
        self.uuids = uuids
        self.transform = transform
        self.bos_client = None
        self.bucket_name = "vast-data-platform-render"
        # self.bucket_name = "vast-liangyuan"
        self.fail_uuids = [line.strip() for line in open("fail.uuids", "r")]
        self.fail_dict = {k: 1 for k in self.fail_uuids}
 
    def init_client(self):
        if self.bos_client is not None:
            return
        else:
            bos_endpoint = "bj.bcebos.com"
            access_key_id = "XXXX"
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
            # object_key = f"4view/{uuid}/render_000{i}.webp"
            object_key = f"4view_normal/{uuid}/normal_000{i}.webp"
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

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        #self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        #    'convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup', precision='fp16')
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')

    
    def forward(self, x, iteration=0):
        # print(f"iteration: {iteration}")
        # if iteration == 0:`
        #     self.check_device()
        #     print(f"Check device Model device: {self.dummy_param.device}")
        return self.model.encode_image(x)


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
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        # Add extra samples to make it evenly divisible
        indices = self.indices[:self.total_size]
        # Subsample
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def load_data(filename: str) -> List[str]:
    results = []
    with open(filename, "r") as f:
        for line in f:
            #results.append(json.loads(line.strip())["model_id"])
            results.append(line.strip())
    return results

def collate_fn(batch):
    result = []
    idxs = []
    for images,idx in batch:
        result.extend(images)
        idxs.append(idx)
    if isinstance(result[0], torch.Tensor):
        result = torch.stack(result)
    return result, idxs

def get_feature(image_paths: List[str], model, preprocess=None, device="cuda") -> List[Union[None, List[float]]]:
    dataset = SimpleDataset(image_paths, preprocess)
    sampler = SequentialDistributedSampler(dataset)
    data_loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn, sampler=sampler)
    features_all = []
    idx = 0
    for images,idxs in tqdm(data_loader):
        images = images.cuda(non_blocking=True).half()
        features = model(images, idx)
        # print(features.shape, images.shape)
        features_all.append(features.cpu().numpy())
        idx += 1
    return features_all

def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    begin = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast():        
            clip_model = SimpleModel()
            clip_model = clip_model.to(device)
            clip_model = DDP(clip_model, device_ids=[local_rank], output_device=local_rank)
            end = time.time()
            print(f"Init Time cost: {end - begin}")
            clip_model.eval()
            image_paths = load_data(sys.argv[1])
            print('Total: ',len(image_paths))
            features = get_feature(image_paths, clip_model, preprocess=clip_model.module.preprocess, device=device)
            

    # Gather features from all ranks
    # features_tensor = torch.tensor(features, device=device)
    # gathered_features = [torch.zeros_like(features_tensor) for _ in range(dist.get_world_size())]
    # dist.all_gather(gathered_features, features_tensor)
    
    # Only save on rank 0
    # if local_rank == 0:
        # gathered_features = torch.cat(gathered_features, dim=0).cpu().numpy()
        # feat_dim = gathered_features[0].shape[-1]
        # gathered_features = gathered_features.reshape(-1, 4 * feat_dim)
        # print(gathered_features.shape, gathered_features.dtype)
        # np.save(sys.argv[2], gathered_features)

    features = np.concatenate(features, axis=0)
    features = features.reshape(-1, 4 * features.shape[-1])
    print(features.shape)
    output_path = f'cache/{sys.argv[2]}.{local_rank}'
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'cache/{sys.argv[2]}.{local_rank}', features)

    torch.distributed.barrier()

    if local_rank == 0:
        # Merge all features
        all_features = []
        for i in range(dist.get_world_size()):
            features = np.load(f'cache/{sys.argv[2]}.{i}.npy')
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        np.save(sys.argv[2], all_features)
        for i in range(dist.get_world_size()):
            os.remove(f'cache/{sys.argv[2]}.{i}.npy')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
