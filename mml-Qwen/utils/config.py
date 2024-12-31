"""
    Project's main config.
"""

import os
from dataclasses import dataclass


@dataclass
class ConfigS:
    clip_model: str = "clip-vit-base-patch32"
    text_model: str = "gpt2"
    seed: int = 100
    num_workers: int = 2
    train_size: int = 0.84
    val_size: int = 0.13
    epochs: int = 30
    lr: int = 1e-4
    k: float = 0.33
    batch_size_exp: int = 6
    ep_len: int = 4
    num_layers: int = 6
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 40
    dropout: float = 0.1
    weights_dir: str = "/data/chy/others/MML-Assignment3/results/weights/Qwen_small"
    dataset_len: int = -1


@dataclass
class ConfigL:
    """
    Project's main config.
    """

    clip_model: str = "clip-vit-large-patch14"
    text_model: str = "gpt2-medium"
    seed: int = 100
    num_workers: int = 2
    train_size: int = 0.84
    val_size: int = 0.13
    epochs: int = 30
    lr: int = 1e-4
    k: float = 0.3
    batch_size_exp: int = 5 # 这里设置的是5
    ep_len: int = 4
    num_layers: int = 5
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 40
    dropout: float = 0.08
    weights_dir: str = "/data/chy/others/MML-Assignment3/results/weights/Qwen_large"
    dataset_len: int = -1


"""
little test
@dataclass
class ConfigS:

    clip_model: str = "clip-vit-base-patch32"
    text_model: str = "gpt2"
    seed: int = 100
    num_workers: int = 2
    train_size: int = 0.84
    val_size: int = 0.13
    epochs: int = 10
    lr: int = 3e-3
    k: float = 0.33
    batch_size_exp: int = 2
    ep_len: int = 4
    num_layers: int = 6
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 40
    dropout: float = 0.1
    weights_dir: str = "/data/chy/others/mml-assignment3/results/weights"
    dataset_len: int = 50
"""