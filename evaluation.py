import os
import argparse
import pandas as pd
from utils import setting_device
from data import TransformSelector
from model import model_selector
from data import CustomDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='parameters for accuracy per class')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--evaldata_dir', type=str, default="./data/train")
    parser.add_argument('--evaldata_info_file', type=str, default="./data/train.csv")
    parser.add_argument("--save_result_path", type=str, default="./train_result")
    parser.add_argument('--model_type', type=str, default='timm')
    parser.add_argument('--model_names', type=str, default='resnet18,224;resnet34,50;resnet50,100')
    parser.add_argument('--worst_n', type=int, default=10)
    
    return parser.parse_args()

def parse_model_names(model_names_str):
    return [name.split(',') for name in model_names_str.split(';')]

def get_evaluation_model(model_type, model_path, model_name, num_classes):
    model = model_selector(model_type=model_type, num_classes=num_classes, model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def evaluation_dataloader(eval_info, evaldata_dir, batch_size, transform_selector, img_size):
    eval_transform = transform_selector.get_transform(img_size=img_size, is_train=False)

    eval_dataset = CustomDataset(
        root_dir=evaldata_dir,
        info_df=eval_info,
        transform=eval_transform,
        is_inference=False
    )

    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    return eval_loader

def inference(model: nn.Module, device: torch.device, eval_loader: DataLoader):
    model.to(device)
    model.eval()
    
    predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(eval_loader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())
    
    return all_targets, predictions

def get_class_recall(true, pred):

    cm = confusion_matrix(true, pred)

    class_recall = []
    for i in range(len(cm)):
        
        true_positives = cm[i, i]
        total_true_class = cm[i, :].sum()

        if total_true_class == 0:
            recall = 0.0
        else:
            recall = true_positives / total_true_class

        class_recall.append(recall)

    return [round(recall, 4) for recall in class_recall]

def recall_visualization(recall, file_name, save_path):

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))
    plt.bar(x=range(len(recall)), height=recall)
    plt.title(f'{file_name} Recall')

    save_path = os.path.join(save_path, f"{file_name}_recall.png")
    plt.savefig(save_path)
    plt.close()

def worst_recall(recall, n):
    indexed_recall = list(enumerate(recall))

    # recall 값을 기준으로 정렬 (값이 작은 순서대로)
    sorted_indexed_recall = sorted(indexed_recall, key=lambda x: x[1])

    # 가장 낮은 값 n개추출
    lowest_n = sorted_indexed_recall[:n]
    
    for i in lowest_n:
        print(i)

def main(opt):
    model_names = parse_model_names(opt.model_names)
    
    device = setting_device()
    eval_info = pd.read_csv(opt.evaldata_info_file)
    transform_selector = TransformSelector(transform_type="albumentations")
    num_classes = 500

    for pt_file, model_name, img_size in model_names:
        eval_loader = evaluation_dataloader(eval_info, opt.evaldata_dir, opt.batch_size, transform_selector, img_size=int(img_size))
        model_path = os.path.join(opt.save_result_path, pt_file + ".pt")
        model = get_evaluation_model(opt.model_type, model_path, model_name, num_classes)
        all_targets, model_predictions = inference(model=model, device=device, eval_loader=eval_loader)

        class_recall = get_class_recall(true=all_targets, pred=model_predictions)
        recall_visualization(class_recall, file_name=pt_file, save_path=opt.save_result_path)
        print(f"{pt_file} worst prediction class")
        worst_recall(class_recall, n=opt.worst_n)

if __name__ == '__main__':
    opt = get_args()
    main(opt)
