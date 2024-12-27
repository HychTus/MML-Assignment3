import os
import json
import torch
import pickle
from PIL import Image
from tqdm import tqdm

from model import ImageEncoder


def get_image_cache(clip_model, meta_path, image_path, cache_dir, dataset_len):
    image_encoder = ImageEncoder(clip_model, device="cuda") # 注意 device
    with open(meta_path, 'r') as f: # 应该是 r, 不是 wb
        meta = json.load(f) #TODO: 先尝试前50个, 跑通代码
    
    # 需要转换成 int 进行比较
    max_id = int(max(meta, key=lambda x: int(x['image_id']))['image_id'])
    print(f"max_id: {max_id}") # 581921

    cache = torch.zeros((max_id+1, image_encoder.model.config.hidden_size))
    vis = torch.zeros((max_id+1), dtype=torch.uint8)
    print(f"hidden_size: {image_encoder.model.config.hidden_size}") # 1024

    for item in tqdm(meta, desc="Encoding images"):
        image_id = int(item['image_id']) # 转换成 int
        # 使用 caption data 可能会出现重复
        if vis[image_id] == 1:
            continue
        vis[image_id] = 1

        # :012d 用于格式化字符串, 使得 image_id 总是 12 位
        image_name = f"COCO_train2014_{image_id:012d}.jpg"

        # 根据 evaluate.py 和 model.py 中的 foward 中实现的计算方式
        image_data = Image.open(os.path.join(image_path, image_name)) # 这里不需要 todevice
        with torch.no_grad():
            image_features = image_encoder(image_data)
            # 这里输出的 tensor 形状是什么?
            # assert image_features.shape == (1, image_encoder.model.config.hidden_size)
            # 不管输出形状是什么, view 就完事了
            cache[image_id] = image_features.view(-1)

    cache_path = os.path.join(cache_dir, f"{clip_model}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


if __name__ == '__main__':
    # clip model = "openai/clip-vit-large-patch14" or "openai/clip-vit-base-patch32"
    # 不需要前缀, 本地保存的都是对应名称的模型

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    get_image_cache(
        clip_model="clip-vit-base-patch32",
        meta_path="/data/chy/others/mml-assignment3/datasets/train_caption.json",
        image_path="/data/chy/others/mml-assignment3/datasets/train2014",
        cache_dir="/data/chy/others/mml-assignment3/cache",
        dataset_len=-1,
    )