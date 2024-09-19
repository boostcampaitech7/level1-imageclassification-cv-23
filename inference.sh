batch_size=(16 32 64)

model_name="resnet101"
img_size='224'

testdata_dir="/data/ephemeral/home/data/test"
testdata_info_file="/data/ephemeral/home/data/test.csv"
save_result_path="/data/ephemeral/home/level1/data/train_result"

cross_validation=True

for bs in "${batch_size[@]}"; do
    echo "Running inference with batch_size=$bs"

    python inference.py --batch_size "$bs" \
                        --model_name "$model_name" \
                        --testdata_dir "$testdata_dir" \
                        --testdata_info_file "$testdata_info_file" \
                        --save_result_path "$save_result_path" \
                        --img_size "$img_size" \
                        --cross_validation "$cross_validation"
done