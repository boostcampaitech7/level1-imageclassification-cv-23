import os
import argparse
import pandas as pd
from utils import test_dataloader, setting_device
from data import TransformSelector
from model import model_selector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='parameters for inference')
    parser.add_argument('--batch_size', type=int, default=64)

    # test data argument parser
    parser.add_argument('--testdata_dir', type=str, default="./data/test")
    parser.add_argument('--testdata_info_file', type=str, default="./data/test.csv")
    parser.add_argument("--save_result_path", type=str, default="./train_result")

    # model argument parser
    parser.add_argument('--model_type', type=str, default='timm')
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument('--model_path', type=str, default="./train_result/best_model.pt")
    parser.add_argument('--img_size', type=str, default=224)

    parser.add_argument('--cross_validation', type=bool, default=False)
    return parser.parse_args()

def get_test_model(model_type, save_result_path, model_name, num_classes):
    model = model_selector(model_type=model_type, num_classes=num_classes, model_name=model_name, pretrained=False)
    model.load_state_dict(
    torch.load(
        save_result_path,
        map_location='cpu'
        )
    )
    return model

def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)
            
            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            
            # 예측 결과 저장
            predictions.extend(logits.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions


def main(opt):
    device = setting_device()

    # 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    test_info = pd.read_csv(opt.testdata_info_file)

    transform_selector = TransformSelector(
    transform_type = "albumentations"
    )

    # 총 class 수.
    num_classes = 500
    save_result_path = opt.save_result_path

    test_loader = test_dataloader(test_info, opt.testdata_dir, opt.batch_size, transform_selector, img_size=int(opt.img_size))

    if opt.cross_validation:
        save_result_path = os.path.join(opt.save_result_path, "cross_validation")
        cross_validation_prediction = []
        for fold in range(5):
            print(f"inference for fold {fold + 1}")
            model_path = os.path.join(save_result_path, f"best_{opt.model_name}_fold_{fold+1}.pt")
            model = get_test_model(opt.model_type, model_path, opt.model_name, num_classes)

            fold_predictions = inference(model=model, device=device, test_loader=test_loader)
            cross_validation_prediction.append(fold_predictions)

        avg_predictions = np.mean(cross_validation_prediction, axis=0)
        final_predictions = np.argmax(avg_predictions, axis=1)

    else:
        model_path = os.path.join(opt.save_result_path, f"best_{opt.model_name}.pt")
        model = get_test_model(opt.model_type, opt.save_result_path, opt.model_name, num_classes)

        predictions = inference(
            model=model, 
            device=device, 
            test_loader=test_loader
            )
        final_predictions = np.argmax(predictions, axis=1)

    test_info['target'] = final_predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output.csv", index=False)

if __name__ == '__main__':
    opt = get_args()
    main(opt)