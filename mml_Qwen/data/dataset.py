"""
    Module contains Dataset class, collate function for DataLoader and loader getter function.

    * ImageCaptionDataset loads data from pickle file and returns image embedding and caption.
    * cl_fn is used to process batch of data and return tensors.
    * get_loader returns DataLoader object.
"""

import os, json
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2Tokenizer

MODEL_PATH = "/data/chy/others/MML-Assignment3/models"

# 原始的返回的应该是 image id
# 但是在经过 cl_fn 之后应该返回 image embedding

# Dataset 返回的是 image_id, id, caption
# cl_fn 之后返回的是 img_emb, cap, att_mask (img_emb 应该直接保存之后 load)
# 只提供了 train_caption, 所以根本不用使用 val

class ImageCaptionDataset(Dataset):
    # TODO: 需要实现一个 ImageCaptionDataset

    def __init__(
            self, 
            meta_path,
            image_cache_path,
            dataset_len,
            max_len
        ):
        # 限制 max_len 便于进行 evaluate
        # 甚至只需要 meta_path, 不需要 img_path
        super().__init__()
        self.meta_path = meta_path
        #BUG: 原本使用的是 'rb', 表示以二进制读取, 但是这里是 json 文件, 应该使用 'r'
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        with open(image_cache_path, 'rb') as f:
            self.image_cache = pickle.load(f)
        
        self.max_len = max_len
        self.meta = self.meta[:dataset_len] # 使用固定的部分, 减少随机性


    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # __getitem__ 针对单独 idx 实现 (增强的时候只能进行单个增强)
        image_id = int(self.meta[idx]['image_id'])
        id = self.meta[idx]['id'] # id 应该表示的是 data id, 没有用, 不用返回
        caption = self.meta[idx]['caption']
        image_name = f"COCO_train2014_{image_id:012d}.jpg" # 填充到 12 位
        return image_name, image_id, caption


def cl_fn(batch, image_cache, tokenizer, max_len):
    # TODO: 需要实现一个 collate function

    # 输入的格式为 [(image_name, id, caption)]
    # collate_fn 用于处理 dataloader 从 dataset返回的多个样本的列表, 并将它们合并成 batch
    # 定制数据的批量化过程, 将数据组织成可以直接输入到 model 中的格式
    
    img_ids = [image_id for _, image_id, _ in batch]
    img_embs = image_cache[img_ids]

    captions = [caption for _, _, caption in batch]
    tok = tokenizer.batch_encode_plus(
        captions,
        max_length=max_len,
        padding='max_length',  # 填充到最大长度
        truncation=True,       # 如果超出长度就截断
        return_tensors='pt'    # 返回 pt(tensor) 或者 np
    )
    input_ids = tok['input_ids']
    attention_mask = tok['attention_mask']

    # encode_plus 只能针对单个 caption 处理
    # input_ids = []
    # attention_mask = []
    # for _, _, caption in batch:
    #     tok = tokenizer.encode_plus(
    #         caption,
    #         max_length=max_len,
    #         padding='max_length', # 填充到最大长度
    #         truncation=True,      # 如果超出长度就截断
    #         return_tensors='pt'   # 返回 pt(tensor) 或者 np
    #     )
    #     input_ids.append(tok['input_ids'])
    #     attention_mask.append(tok['attention_mask'])

    # input_ids = torch.cat(input_ids, dim=0)
    # attention_mask = torch.cat(attention_mask, dim=0)

    assert input_ids.shape == (len(batch), max_len)
    assert attention_mask.shape == (len(batch), max_len)

    # print("tok shape", input_ids.shape, attention_mask.shape)
    # encode_plus 返回的结果是 dict, 包含 input_ids, attention_mask, token_type_ids 等信息
    return img_embs, input_ids, attention_mask


def get_loader(
        dataset, bs_exp=5, shuffle=True, num_workers=0, pin_memory=False,
        max_len=None, image_cache=None
    ):
    # 根据 Dataset 返回 DataLoader
    # 所以输入的数据是什么格式? image embedding 到底是谁来进行的?

    # GPT2 使用的 tokenizer 都是相同的, 所以可以直接使用
    model_path = os.path.join(MODEL_PATH, "Qwen2.5-0.5B")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # image_cache = dataset.image_cache
    # max_len = dataset.max_len

    return DataLoader(
        dataset,
        batch_size=2**bs_exp,
        collate_fn=lambda b: cl_fn(b, image_cache, tokenizer, max_len),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
