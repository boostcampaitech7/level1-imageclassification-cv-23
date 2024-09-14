#!/bin/bash

lrs=(0.001 0.0001 0.01)
batch_size=(16 32 64)
epochs=(10 20 30)
schedule_step=(10 20 30)
gamma=(0.1 0.2 0.3)
lr_decay=(2 4 6)
model=(resnet18 resnet34 resnet50 resnet101 resnet152)

train_csv_file="/data/ephemeral/home/data/train"
traindata_info_file="/data/ephemeral/home/data/train.csv" 
save_result_path="/data/ephemeral/home/level1/data/train_result"

for lr in "${lrs[@]}"; do
    for bs in "${batch_size[@]}"; do
        for ep in "${epochs[@]}"; do
            for ss in "${schedule_step[@]}"; do
                for gm in "${gamma[@]}"; do
                    for ld in "${lr_decay[@]}"; do
                        for md in "${model[@]}"; do
                            echo "Training with lr=$lr, batch_size=$bs, epochs=$ep, scheduler_step_size=$ss, scheduler_gamma=$gm, lr_decay=$ld, model_name=$md"
                            python train.py --lr "$lr" --batch_size "$bs" --epochs "$ep" --scheduler_step_size "$ss" --scheduler_gamma "$gm" --lr_decay "$ld" --traindata_dir "$train_csv_file" --traindata_info_file "$traindata_info_file" --save_result_path "$save_result_path" --model_name "$md"
                        done
                    done
                done
            done
        done  
    done
done
