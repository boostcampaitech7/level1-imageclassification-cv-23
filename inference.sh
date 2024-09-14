batch_size=(16 32 64)

model_name="resnet18"
model_path="/data/ephemeral/home/level1/data/train_result/best_resnet18.pt"
testdata_dir="/data/ephemeral/home/data/test"
testdata_info_file="/data/ephemeral/home/data/test.csv"
save_result_path="/data/ephemeral/home/level1/data/train_result"

for bs in "${batch_size[@]}"; do
    echo "Running inference with batch_size=$bs"

    python inference.py --batch_size "$bs" \
                        --model_name "$model_name" \
                        --model_path "$model_path" \
                        --testdata_dir "$testdata_dir" \
                        --testdata_info_file "$testdata_info_file" \
                        --save_result_path "$save_result_path"
done