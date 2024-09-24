#!/bin/bash

train_csv_file="/data/ephemeral/home/common_data/data/train"
traindata_info_file="/data/ephemeral/home/common_data/data/train.csv" 
save_result_path="/data/ephemeral/home/wandb-test/level1-cv-23/train_result"
pretrained_model_path="/data/ephemeral/home/wandb-test/level1-cv-23/best_convnext_xxlarge.clip_laion2b_soup_ft_in1k.pt"

model_name="convnext_xxlarge.clip_laion2b_soup_ft_in1k"
img_size=256

echo "Starting hyperparameter optimization for model: $model_name, img_size: $img_size"

python wandb_train.py --traindata_dir "$train_csv_file" \
                      --traindata_info_file "$traindata_info_file" \
                      --save_result_path "$save_result_path" \
                      --model_name "$model_name" \
                      --img_size "$img_size" \
                      --pretrained_model_path "$pretrained_model_path"

echo "Hyperparameter optimization completed."