batch_size=32
worst_n=30

# pt 파일 이름, timm 모델 이름, 이미지 사이즈 (띄어쓰기 없이 콤마로 구분)
model_n_img_size="best_resnetv2_50_fold_1,resnetv2_50,224;
                best_resnetv2_50_fold_2,resnetv2_50,224;
                best_resnetv2_50_fold_1,resnetv2_50,124"
model_n_img_size=$(echo $model_n_img_size | tr -d '[:space:]')

evaldata_dir="/data/ephemeral/home/data/train"
evaldata_info_file="/data/ephemeral/home/data/train.csv"
save_result_path="/data/ephemeral/home/train_result"

for bs in "${batch_size[@]}"; do
    echo "Running evaluate with batch_size=$bs"

    python evaluation.py --batch_size "$bs" \
                        --evaldata_dir "$evaldata_dir" \
                        --evaldata_info_file "$evaldata_info_file" \
                        --save_result_path "$save_result_path" \
                        --model_names "$model_n_img_size" \
                        --worst_n "$worst_n"
done