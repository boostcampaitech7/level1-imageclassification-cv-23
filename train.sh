#!/bin/bash

lrs=(0.001 0.01)
batch_size=(64)
epochs=(20 30)
gamma=(0.1)
lr_decay=(4)
L1=0.001
L2=0.001
early_stopping_delta=0.01
early_stopping_patience=3

models_and_img_sizes=( 
    "resnet101 224"
)

cross_validation=True
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
                        echo "Training with lr=$lr, batch_size=$bs, epochs=$ep, scheduler_gamma=$gm, lr_decay=$ld, model_name=$model_name, img_size=$img_size, cross_validation=$cross_validation L1=$L1 L2=$L2"
                        python train.py --lr "$lr" --batch_size "$bs" \
                                        --epochs "$ep" --scheduler_gamma "$gm" \
                                        --lr_decay "$ld" --traindata_dir "$train_csv_file" \
                                        --traindata_info_file "$traindata_info_file" \
                                        --save_result_path "$save_result_path" --model_name "$model_name" \
                                        --img_size "$img_size" --cross_validation "$cross_validation" \
                                        --L1 "$L1" --L2 "$L2" \
                                        --early_stopping_delta "$early_stopping_delta" \
                                        --early_stopping_patience "$early_stopping_patience"
                    done
                done
            done
        done  
    done
done
