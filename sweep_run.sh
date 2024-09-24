#!/bin/bash

train_csv_file="/data/ephemeral/home/common_data/data/train"
traindata_info_file="/data/ephemeral/home/common_data/data/train.csv" 
save_result_path="/data/ephemeral/home/wandb-test/level1-cv-23/train_result"
pretrained_model_path="/data/ephemeral/home/wandb-test/level1-cv-23/best_convnext_xxlarge.clip_laion2b_soup_ft_in1k.pt"
sweep_config_path="sweep_config.yaml"

model_name="convnext_xxlarge.clip_laion2b_soup_ft_in1k"
img_size=256

echo "Starting hyperparameter optimization for model: $model_name, img_size: $img_size"

# Sweep 생성 및 ID 추출
sweep_output=$(wandb sweep --project model_optimization $sweep_config_path 2>&1)
sweep_id=$(echo "$sweep_output" | grep -oP "(?<=wandb agent )[^ ]+")
sweep_count=5

wandb agent --count $sweep_count $sweep_id


echo "Hyperparameter optimization completed."