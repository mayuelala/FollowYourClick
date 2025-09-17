import pyarrow as pa
import os
import random
import re
from PIL import ImageFile
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

import csv
from einops import rearrange
import torchvision.transforms as T
from tqdm import tqdm
import traceback
import numpy as np
import glob
import io
import einops


def get_proper_resize_size(img, size):
    from PIL import Image
    if isinstance(size, int):
        size = (size, size)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    W, H = img.size
    h, w = size
    r = max(h / H, w / W)
    return int(H * r), int(W * r)

class LaionDataset():

    def __init__(self,
         resolution,
         resolution_h=256,
         resolution_w=256,
         **kwargs
    ):
        self.root = 'YOUR PATH/share_1367250/0_public_datasets/laion-high-resolution/arrows'
        self.resolution = resolution
        names = glob.glob(f'{self.root}/*.arrow')

        if len(names) != 0:
            tables = []
            new_names = []
            for name in tqdm(names, desc='Reading arrow: ', ncols=100):
                try:
                    tables.append(pa.ipc.RecordBatchFileReader(pa.memory_map(f"{name}", "r")).read_all())
                    new_names.append(name)
                except:
                    print(name)

            names = new_names
            # print(1)
            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)

            self.label_columb_name = 'caption'
            self.labels = self.table[self.label_columb_name].to_pandas().tolist()


    def __len__(self):
        return len(self.table)

    def get_raw_image(self, index, image_key="image"):
        # index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        # image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image,
        }

    def get_text(self, raw_index):
        text = str(self.table['caption'][raw_index][0])

        return {
            "text": text,
        }

    def __getitem__(self, item):
        try:
            return self.getitem(item)
        except Exception as e:
            print (f'Failed to load data with error {traceback.format_exc()}-{e}')
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

    def getitem(self, item):
        example = {}
        img = self.get_raw_image(item)
        import typing
        if not isinstance(self.resolution, int):
            W, H = img.size
            h, w = self.resolution
            r = max(h / H, w / W)
            new_size = int(H * r), int(W * r)
        else:
            new_size = self.resolution
        transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Resize(new_size),
            T.CenterCrop(self.resolution),
            T.Normalize(0.5, 0.5)
        ])
        img = einops.rearrange(transform(img), 'c h w -> 1 c h w')
        txt = self.get_text(item)['text']


        example["pixel_values"] = img
        example["text"] = txt
        example["ori_text"] = txt
        example["sample_index"] = item
        example["datasetname"] = 'laion'
        example["fps"] = 1

        return example

class LaionAesthetic5:
    # 大概 20M
    def __init__(self,
                 resolution,
                 **kwargs
                 ):
        from tqdm import tqdm
        # root_1024 = 'YOUR PATH/share_1367250/0_public_datasets/laion-aesthetic5/metadata/1024'
        # root_512 = 'YOUR PATH/share_1367250/0_public_datasets/laion-aesthetic5/metadata/512to1024'
        data = []
        # for idx, f_path in tqdm(enumerate(os.listdir(root_1024)), desc='Reading LaionAesthetic5 1024'):
        #     f_path = os.path.join(root_1024, f_path)
        #     # if idx > 1:
        #     #     break
        #     with open(f_path, 'r') as f:
        #         for line in f:
        #             _, img_path, prompt, *_  = line.split('|*|')
        #             data.append((img_path, prompt))
                    
        # for idx, f_path in tqdm(enumerate(os.listdir(root_512)), desc='Reading LaionAesthetic5 512'):
        #     f_path = os.path.join(root_512, f_path)
        #     # if idx > 1:
        #     #     break
        #     with open(f_path, 'r') as f:
        #         for line in f:
        #             _, img_path, prompt, *_  = line.split('|*|')
        #             data.append((img_path, prompt))
        
        # root_512 = "YOUR PATH/share_1367250/0_public_datasets/laion-aesthetic5/metadata/512to1024_all.txt"
        # root_1024 = "YOUR PATH/share_1367250/0_public_datasets/laion-aesthetic5/metadata/1024_all.txt"
        
        root_512 = "YOUR PATH/share_301124792/0_public_datasets/laion-aesthetic5/metadata/512to1024_all.txt"
        root_1024 = "YOUR PATH/share_301124792/0_public_datasets/laion-aesthetic5/metadata/1024_all.txt"
        
        with open(root_512, 'r') as f:
            for line in tqdm(f):
                _, img_path, prompt, *_  = line.split('|*|')
                data.append((img_path, prompt))
                
        with open(root_1024, 'r') as f:
            for line in tqdm(f):
                _, img_path, prompt, *_  = line.split('|*|')
                data.append((img_path, prompt))
                
        # loguru.logger.info(f'{len(data)} items')
        self.data = data
        self.resolution = resolution

    def __getitem__(self, item):
        path, prompt = self.data[item]
        if not path.startswith('/mnt/'):
            path = '/mnt/' + path
        img = Image.open(path)
        target_res = get_proper_resize_size(img, self.resolution)

        transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Resize(target_res),
            T.CenterCrop(self.resolution),
            T.Normalize(0.5, 0.5)
        ])
        img = einops.rearrange(transform(img), 'c h w -> 1 c h w')

        example = {}
        example["pixel_values"] = img
        example["text"] = prompt
        example["ori_text"] = prompt
        example["sample_index"] = item
        example["datasetname"] = 'LaionAesthetic5'
        example["fps"] = 1

        return example

    def __len__(self):
        return len(self.data)
    
class AllImageDataset:
    def __init__(self, 
                resolution,
                # resolution_w=256,
                # resolution_h=256,
                **kwargs):
        self.laion = LaionDataset(resolution)
        self.laion_aes = LaionAesthetic5(resolution)
        
        from torch.utils.data import ConcatDataset, ChainDataset
        # self.concat_dataset = ConcatDataset([self.laion, self.laion_aes, self.webvid, self.pexel])
        self.concat_dataset = ConcatDataset([self.laion, self.laion_aes])
        # self.concat_dataset = ConcatDataset([self.laion])
        # self.concat_dataset = ConcatDataset([self.laion, self.webvid])

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, item):
        return self.concat_dataset[item]