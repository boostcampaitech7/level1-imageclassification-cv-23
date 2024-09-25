import os
import argparse
import pandas as pd
from utils import test_dataloader, setting_device, parse_model_names, get_model, ensemble_inference
from data import TransformSelector

import numpy as np
from datetime import datetime

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='parameters for inference')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--testdata_dir', type=str, default="./data/test")
    parser.add_argument('--testdata_info_file', type=str, default="./data/test.csv")
    parser.add_argument("--save_result_path", type=str, default="./train_result")
    parser.add_argument('--model_type', type=str, default='timm')

    
    parser.add_argument('--model_names', type=str, default='resnet18,224;resnet34,50;resnet50,100')
    
    return parser.parse_args()

def main(opt):
    model_names = parse_model_names(opt.model_names)
    
    device = setting_device()
    test_info = pd.read_csv(opt.testdata_info_file)
    transform_selector = TransformSelector(transform_type="albumentations")
    num_classes = 500
    ensemble_predictions = []

    for pt_file, model_name, img_size in model_names:
        test_loader = test_dataloader(test_info, opt.testdata_dir, opt.batch_size, transform_selector, img_size=int(img_size), is_inference=True)
        model_path = os.path.join(opt.save_result_path, pt_file + ".pt")
        model = get_model(opt.model_type, model_path, model_name, num_classes)
        model_predictions = ensemble_inference(model=model, device=device, dataloader=test_loader, mode='ensemble')
        ensemble_predictions.append(model_predictions)

    avg_predictions = np.mean(ensemble_predictions, axis=0)
    final_predictions = np.argmax(avg_predictions, axis=1)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"ensemble_{current_time}.csv"
    
    test_info['target'] = final_predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(output_filename, index=False)

if __name__ == '__main__':
    opt = get_args()
    main(opt)
