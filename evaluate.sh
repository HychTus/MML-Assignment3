#!/bin/bash

image_path="/data/chy/others/MML-Assignment3/datasets/train2014"
result_path="/data/chy/others/MML-Assignment3/results/visual_results/"
num_samples=10

python mml/evaluate.py -S S -I $image_path \
                       -R "$result_path" \
                       -C "GPT2_SCLIP_best.pt" \
                       -N $num_samples


python mml/evaluate.py -S L -I $image_path \
                       -R "$result_path" \
                       -C "GPT2M_LCLIP_best.pt" \
                       -N $num_samples

python mml_Qwen/evaluate.py -S S -I $image_path \
                            -R "$result_path" \
                            -C "Qwen_SCLIP_best" \
                            -N $num_samples


python mml_Qwen/evaluate.py -S L -I $image_path \
                            -R "$result_path" \
                            -C "Qwen_LCLIP_best.pt" \
                            -N $num_samples

python mml_QFormer/evaluate.py -S S -I $image_path \
                               -R "$result_path" \
                               -C "Qwen_SCLIP_QFormer_best.pt" \
                               -N $num_samples

python mml_QFormer/evaluate.py -S L -I $image_path \
                               -R "$result_path" \
                               -C "Qwen_LCLIP_QFormer_best.pt" \
                               -N $num_samples