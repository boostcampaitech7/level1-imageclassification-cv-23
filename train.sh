#!/bin/bash

lrs=(0.001 0.0001 0.01)
batch_size=(16 32 64)
epochs=(10 20 30)
gamma=(0.1 0.2 0.3)
lr_decay=(2 4 6)

models_and_img_sizes=( 
    "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k 256" 
    "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384 384" 
    "densenet161.tv_in1k 224" 
    "resnet101 224"
)

train_csv_file="/data/ephemeral/home/data/train"
traindata_info_file="/data/ephemeral/home/data/train.csv" 
save_result_path="/data/ephemeral/home/level1/data/train_result"

for lr in "${lrs[@]}"; do
    for bs in "${batch_size[@]}"; do
        for ep in "${epochs[@]}"; do
            for gm in "${gamma[@]}"; do
                for ld in "${lr_decay[@]}"; do
                    for model_and_img_size in "${models_and_img_sizes[@]}"; do
                        model_name=$(echo $model_and_img_size | cut -d ' ' -f 1)
                        img_size=$(echo $model_and_img_size | cut -d ' ' -f 2)
                        echo "Training with lr=$lr, batch_size=$bs, epochs=$ep, scheduler_gamma=$gm, lr_decay=$ld, model_name=$model_name, img_size=$img_size"
                        python train.py --lr "$lr" --batch_size "$bs" --epochs "$ep" --scheduler_gamma "$gm" --lr_decay "$ld" --traindata_dir "$train_csv_file" --traindata_info_file "$traindata_info_file" --save_result_path "$save_result_path" --model_name "$model_name" --img_size "$img_size"
                    done
                done
            done
        done  
    done
done
