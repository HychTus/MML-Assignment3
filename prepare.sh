#!/bin/bash

# "openai/clip-vit-large-patch14" or "openai/clip-vit-base-patch32"

CLIP_MODEL_S="openai/clip-vit-base-patch32"
CLIP_MODEL_L="openai/clip-vit-large-patch14"
GPT_MODEL_S="openai-community/gpt2"
GPT_MODEL_L="openai-community/gpt2-medium"

# 需要指定到目录
./hfd.sh $CLIP_MODEL_S --local-dir "./models/clip-vit-base-patch32" -x 16
./hfd.sh $CLIP_MODEL_L --local-dir "./models/clip-vit-large-patch14" -x 16
./hfd.sh $GPT_MODEL_S --local-dir "./models/gpt2" -x 16
./hfd.sh $GPT_MODEL_L --local-dir "./models/gpt2-medium" -x 16