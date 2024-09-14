#!/bin/bash

lrs=(0.001 0.0001 0.01)
batch_size=(16 32 64)

for lr in "${lrs[@]}"; do
    for bs in "${batch_size[@]}"; do
        echo "Training with lr=$lr and batch_size=$batch_size"
        python train.py --lr "$lr" --batch_size "$batch_size"
  done
done