from model import model_selector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_model_names(model_names_str):
    return [name.split(',') for name in model_names_str.split(';')]

def get_model(model_type, model_path, model_name, num_classes):
    model = model_selector(model_type=model_type, num_classes=num_classes, model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def ensemble_inference(model: nn.Module, device: torch.device, dataloader: DataLoader, mode: str):
    model.to(device)
    model.eval()
    predictions, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch[0].to(device)
            logits = F.softmax(model(images), dim=1)
            
            if mode == 'ensemble':
                predictions.extend(logits.cpu().numpy())
            elif mode == 'evaluation':
                targets = batch[1].to(device)
                preds = logits.argmax(dim=1)
                predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

    return predictions, all_targets if mode == 'evaluation' else predictions
