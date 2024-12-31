#!/bin/bash

image_path="/data/chy/others/MML-Assignment3/datasets/train2014"
# weight_path="/data/chy/others/MML-Assignment3/results/weights"
result_path="/data/chy/others/MML-Assignment3/results/visual_results/"
num_samples=20

# python mml/evaluate.py -S S -I $image_path \
#                        -R "$result_path" \
#                        -C "$result_path/origin_small/epoch_10.pth" \
#                        -N $num_samples


python mml/evaluate.py -S L -I $image_path \
                       -R "$result_path" \
                       -C "epoch_9.pt" \
                       -N $num_samples

# python mml_Qwen/evaluate.py -S S -I $image_path \
#                             -R "$result_path/Qwen_small" \
#                             -C "$weight_path/Qwen_small/epoch_7.pth" \
#                             -N $num_samples


# python mml_Qwen/evaluate.py -S L -I $image_path \
#                             -R "$result_path" \
#                             -C "epoch_10.pt" \
#                             -N $num_samples

# python mml_QFormer/evaluate.py -S S -I $image_path \
#                                -R "$result_path/QFormer_small" \
#                                -C "$weight_path/QFormer_small/epoch_10.pth" \
#                                -N $num_samples
# 
# python mml_QFormer/evaluate.py -S L -I $image_path \
#                                -R "$result_path/QFormer_large" \
#                                -C "$weight_path/QFormer_large/epoch_10.pth" \
#                                -N $num_samples