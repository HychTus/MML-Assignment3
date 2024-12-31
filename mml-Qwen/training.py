"""
    Script that contains whole training process.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split

import wandb
from mml.data import ImageCaptionDataset, get_loader
from mml.model import Net, Trainer
from mml.utils import ConfigS, ConfigL, LRWarmup

parser = argparse.ArgumentParser()

# checkpoint name
parser.add_argument(
    "-C", "--checkpoint-name", type=str, default="", help="Checkpoint name"
)

parser.add_argument("--cuda_device", type=str, default="0", help="Cuda device")

# model size
parser.add_argument(
    "-S",
    "--size",
    type=str,
    default="S",
    help="Model size [S, L]",
    choices=["S", "L", "s", "l"],
)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
os.environ["WANDB_API_KEY"] = "0a67c749b5589ca32092b047bdbdcea9d2f8facf"

# 根据参数选择不同的 config, ConfigL 和 ConfigS 定义在 utils/config.py 中
config = ConfigL() if args.size.upper() == 'L' else ConfigS()

# set seed
# 对于所有的 package 都固定随机数种子
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    # TODO: 需要你自己实现一个ImageCaptionDataset在`data/dataset.py`中
    # 根据 Dataset 的实现来调整相关代码
    # 不需要进行 {config.clip_model.split('/')[-1]}
    dataset = ImageCaptionDataset(
        meta_path="/data/chy/others/MML-Assignment3/datasets/train_caption_filtered.json",
        image_cache_path=f"/data/chy/others/MML-Assignment3/cache/{config.clip_model}.pkl",
        max_len=config.max_len, # 填充到的 maxlen
        dataset_len=config.dataset_len # 需要在 config 中添加
    )

    # train_size 表示使用的比例
    # train:val:test = 0.84:0.13:0.03
    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    # 随机划分数据集
    #NOTE: 划分之后的 Subset 失去了相关的性质
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [config.train_size, config.val_size, config.test_size]
    )

    # 从 Dataset 构建 Dataloader, 用于模型训练
    # 为什么没有定义 test_loader?
    train_loader = get_loader(
        train_dataset,
        bs_exp=config.batch_size_exp if is_cuda else 2,
        shuffle=True,
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
        max_len=dataset.max_len,
        image_cache=dataset.image_cache,
    )

    valid_loader = get_loader(
        val_dataset,
        bs_exp=config.batch_size_exp if is_cuda else 2,
        shuffle=False,
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
        max_len=dataset.max_len,
        image_cache=dataset.image_cache,
    )

    # Net 定义在 model.py, 根据 config 中参数构建模型
    model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()

    # checkpoint_name 如果对应的是 epoch
    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

    # 使用的是 test_dataset
    # 将 model 和 data 都传入构建 Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_dataset=test_dataset,
        # test_path=os.path.join("data", "raw"), # TODO: 请你修改这里为你自己的目录
        test_path="/data/chy/others/MML-Assignment3/datasets/train2014", # 图片的路径
        ckp_path=ckp_path,
        device=device,
    )

    # QA 这部分 wandb 是在如何使用? 为什么对于 model 进行 watch?
    # build train model process with experiment tracking from wandb
    wandb.init(project="captioner", config=config.__dict__) # 还是先 online mode="offline"
    wandb.watch(trainer.model, log="all")

    # range 迭代的是 trainer.epoch (还有这种用法)
    # 并不是使用这个迭代, 而是设置初始值, 从之前的进度开始继续训练
    for epoch in range(trainer.epoch, config.epochs):
        trainer.train_epoch()
        trainer.valid_epoch()
        trainer.test_step()

        metadata = trainer.get_training_data()

        # log loss to wandb
        wandb.log(
            {
                "train_loss/loss": metadata["train_loss"][-1],
                "valid_loss/loss": metadata["valid_loss"][-1],
                "lr": metadata["lr"],
                "examples": wandb.Image(metadata["examples"]),
            }
        )

        if not os.path.exists(config.weights_dir):
            os.makedirs(config.weights_dir)

        # 训练完 6 个 epoch 保存一次模型 (提前检测路径是否存在)
        # if (epoch + 1) % 6 == 0:
        trainer.save_ckp(os.path.join(config.weights_dir, f"epoch_{epoch + 1}.pt"))
        # 这里存储的位置感觉有问题, 完全没有使用到 checkpoint_name
