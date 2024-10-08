import wandb
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from torch.cuda.amp import autocast, GradScaler

from data import TransformSelector
from src import Loss
from utils import setting_device, data_split, create_dataloaders, get_scheduler
from model import model_selector

import gc
gc.collect()
torch.cuda.empty_cache()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='hyperparameters for training')
    
    parser.add_argument('--traindata_dir', type=str, default="./data/train")
    parser.add_argument('--traindata_info_file', type=str, default="./data/train.csv")
    parser.add_argument('--save_result_path', type=str, default='./train_result')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--img_size', type=str, default='224')
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--scheduler_gamma', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--L2', type=float)
    
    args = parser.parse_args()
    return args

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        model_name: str,

    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.model_name = model_name
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수
    
    def save_model(self, epoch, loss):
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'{self.model_name}_model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path) # 가중치만 저장
        # torch.save(self.model, current_model_path) # 모델 전체 저장

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort() # loss 기준 오름차순 정렬
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, f'best_{self.model_name}.pt')
            torch.save(self.model.state_dict(), best_model_path)
            # torch.save(self.model, best_model_path)
            print(f"Saved best model for {self.model_name} at epoch {epoch} with loss {loss:.4f}")


    def train_epoch(self) -> float:
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()

            with torch.autograd.detect_anomaly():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        return total_loss / len(self.train_loader), correct_predictions / total_samples * 100

    def validate(self) -> float:
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        return total_loss / len(self.val_loader), correct_predictions / total_samples * 100

    def train(self) -> None:
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step()

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })


        return val_loss  # Return the final validation loss for hyperparameter optimization

def main():
    opt = get_args()
    
    wandb.init(config=opt)
    wandb.config.update(opt)
        
    device = setting_device()
    transform_selector = TransformSelector(transform_type="albumentations")

    train_df, val_df, num_classes = data_split(opt.traindata_info_file)
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, opt.traindata_dir, opt.batch_size, 
        transform_selector, img_size=int(opt.img_size)
    )

    # Load the pretrained model
    model = model_selector(model_type='timm', num_classes=num_classes, model_name=opt.model_name, pretrained=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.L2)
    scheduler = get_scheduler(optimizer, train_loader, opt.lr_decay, opt.scheduler_gamma)
    loss_fn = Loss()

    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=opt.epochs,
        result_path=opt.save_result_path,
        model_name=opt.model_name,
    )

    final_val_loss = trainer.train()
    wandb.log({"final_val_loss": final_val_loss})
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_val_loss

if __name__ == '__main__':
    main()