import os
import json
import torch
import pickle
import argparse
from PIL import Image
from tqdm import tqdm

from model import ImageEncoder


def check_image_cache(
        clip_model, meta_path, image_dir, cache_dir,
    ):
    image_encoder = ImageEncoder(model=clip_model, device="cuda")
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    cache_path = os.path.join(cache_dir, f"{clip_model}.pkl")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    # random sample 50 in all meta
    choose_ids = torch.randperm(len(meta))[:50]
    for idx in tqdm(choose_ids, desc="Checking cache"):
        item = meta[idx]
        image_id = int(item['image_id'])
        image_name = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        image_data = Image.open(image_path)
        with torch.no_grad():
            image_features = image_encoder(image_data)
            assert torch.allclose(image_features.view(-1), cache[image_id])


def get_image_cache(
        clip_model, meta_path, image_dir, cache_dir, 
        dataset_len, calc_emb
    ):
    image_encoder = ImageEncoder(model=clip_model, device="cuda") # 注意 device
    with open(meta_path, 'r', encoding='utf-8') as f: # 应该是 r, 不是 wb
        meta = json.load(f)[:dataset_len] #TODO: 先尝试前50个, 跑通代码
    
    # 需要转换成 int 进行比较
    max_id = int(max(meta, key=lambda x: int(x['image_id']))['image_id'])
    print(f"max_id: {max_id}") # 581921

    cache = torch.zeros((max_id+1, image_encoder.model.config.hidden_size))
    vis = torch.zeros((max_id+1), dtype=torch.uint8)
    print(f"hidden_size: {image_encoder.model.config.hidden_size}") # 1024

    filtered_meta = []
    for item in tqdm(meta, desc="Encoding images"):
        image_id = int(item['image_id']) # 转换成 int
        # :012d 用于格式化字符串, 使得 image_id 总是 12 位
        image_name = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        filtered_meta.append(item)
        if not calc_emb:
            continue

        # 使用 caption data 可能会出现重复
        if vis[image_id] == 1:
            continue
        vis[image_id] = 1

        # 根据 evaluate.py 和 model.py 中的 foward 中实现的计算方式
        image_data = Image.open(image_path) # 这里不需要 todevice
        with torch.no_grad():
            image_features = image_encoder(image_data)
            # 这里输出的 tensor 形状是什么?
            # assert image_features.shape == (1, image_encoder.model.config.hidden_size)
            # 不管输出形状是什么, view 就完事了
            cache[image_id] = image_features.view(-1).cpu() # 转换到 cpu 上

    if calc_emb:
        cache_path = os.path.join(cache_dir, f"{clip_model}.pkl")
        with open(cache_path, 'wb') as f:
           pickle.dump(cache, f)

    # 需要将 string list 合并起来
    # 注意使用 "" 和 '', 如果不存在不能直接 open, 但是写了 "w" 就可以

    new_meta_path = f"{''.join(meta_path.split('.')[:-1])}_{clip_model}.json"
    with open(new_meta_path, "w") as f:
        json.dump(filtered_meta, f, indent=4)


if __name__ == '__main__':
    # clip model = "openai/clip-vit-large-patch14" or "openai/clip-vit-base-patch32"
    # 不需要前缀, 本地保存的都是对应名称的模型

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="clip-vit-base-patch32", 
        choices=["clip-vit-base-patch32", "clip-vit-large-patch14"]
    )
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--dataset_len", type=int, default=-1)
    parser.add_argument("--calc_emb", type=bool, default=True)
    parser.add_argument("--check", type=bool, default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    if args.check:
        check_image_cache(
            clip_model=args.model,
            meta_path="/data/chy/others/MML-Assignment3/datasets/train_caption_filtered.json",
            image_dir="/data/chy/others/MML-Assignment3/datasets/train2014",
            cache_dir="/data/chy/others/MML-Assignment3/cache",
        )
    else:
        get_image_cache(
            clip_model=args.model,
            meta_path="/data/chy/others/MML-Assignment3/datasets/train_caption.json",
            image_dir="/data/chy/others/MML-Assignment3/datasets/train2014",
            cache_dir="/data/chy/others/MML-Assignment3/cache",
            dataset_len=args.dataset_len,
            calc_emb=args.calc_emb
        )