#!/bin/bash

lrs=(0.01)
batch_size=(64)
epochs=(20)
gamma=(0.1)
lr_decay=(4)
L1=0.001
L2=0.001
early_stopping_delta=0.01
early_stopping_patience=3

cross_validation_expression=True # 명시적 변수임. 
AMP=True # 명시적 변수임.

scheduler_type="cosine"
min_lr=0.000001
epochs_per_restart=5
models_and_img_sizes=( 
    "resnext101_32x32d.fb_wsl_ig1b_ft_in1k 224"
)

train_csv_file="/data/ephemeral/home/common_data/data/train"
traindata_info_file="/data/ephemeral/home/common_data/data/train.csv" 
save_result_path="/data/ephemeral/home/workspace/lv1-cv23/data/train_result"

for lr in "${lrs[@]}"; do
    for bs in "${batch_size[@]}"; do
        for ep in "${epochs[@]}"; do
            for gm in "${gamma[@]}"; do
                for ld in "${lr_decay[@]}"; do
                    for model_and_img_size in "${models_and_img_sizes[@]}"; do
                        model_name=$(echo $model_and_img_size | cut -d ' ' -f 1)
                        img_size=$(echo $model_and_img_size | cut -d ' ' -f 2)
                        echo "Training with lr=$lr, batch_size=$bs, epochs=$ep, scheduler_gamma=$gm, lr_decay=$ld, model_name=$model_name, img_size=$img_size, L1=$L1, L2=$L2, cross_validation=$cross_validation_expression, AMP=$AMP"
                        python train.py --lr "$lr" --batch_size "$bs" \
                                        --epochs "$ep" --scheduler_gamma "$gm" \
                                        --lr_decay "$ld" --traindata_dir "$train_csv_file" \
                                        --traindata_info_file "$traindata_info_file" \
                                        --save_result_path "$save_result_path" --model_name "$model_name" \
                                        --img_size "$img_size" \
                                        --L1 "$L1" --L2 "$L2" \
                                        --early_stopping_delta "$early_stopping_delta" \
                                        --early_stopping_patience "$early_stopping_patience" \
                                        --cross_validation \
                                        --scheduler_type "$scheduler_type" \
                                        --min_lr "$min_lr" \
                                        --epochs_per_restart "$epochs_per_restart"
                    done
                done
            done
        done  
    done
done
