#!/bin/bash

lrs=(0.001 0.0001 0.01)
batch_size=(16 32 64)

train_csv_file="/data/ephemeral/home/data/train"
traindata_info_file="/data/ephemeral/home/data/train.csv" 
save_result_path="/data/ephemeral/home/level1/data/train_result"


for lr in "${lrs[@]}"; do
    for bs in "${batch_size[@]}"; do
        echo "Training with lr=$lr and batch_size=$batch_size"
        python train.py --lr "$lr" --batch_size "$batch_size" --traindata_dir "$train_csv_file" --traindata_info_file "$traindata_info_file" --save_result_path "$save_result_path"
  done
done