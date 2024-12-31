#!/bin/bash

image_path="/data/chy/others/MML_Assignment3/datasets/train2014"
weight_path="/data/chy/others/MML_Assignment3/results/weights"
result_path="/data/chy/others/MML_Assignment3/results/visual_results/"
num_samples=10

python mml/evaluate.py -S S -I $image_path \
                       -R "$result_path/origin_small" \
                       -C "$result_path/origin_small/epoch_10.pth" \
                       -N $num_samples
                       
python mml/evaluate.py -S L -I $image_path \
                       -R "$result_path/origin_large" \
                       -C "$result_path/origin_large/epoch_10.pth" \
                       -N $num_samples

python mml_Qwen/evaluate.py -S S -I $image_path \
                            -R "$result_path/Qwen_small" \
                            -C "$weight_path/Qwen_small/epoch_10.pth" \
                            -N $num_samples


python mml_Qwen/evaluate.py -S L -I $image_path \
                            -R "$result_path/Qwen_large" \
                            -C "$weight_path/Qwen_large/epoch_10.pth" \
                            -N $num_samples

python mml_QFormer/evaluate.py -S S -I $image_path \
                               -R "$result_path/QFormer_small" \
                               -C "$weight_path/QFormer_small/epoch_10.pth" \
                               -N $num_samples

python mml_QFormer/evaluate.py -S L -I $image_path \
                               -R "$result_path/QFormer_large" \
                               -C "$weight_path/QFormer_large/epoch_10.pth" \
                               -N $num_samples