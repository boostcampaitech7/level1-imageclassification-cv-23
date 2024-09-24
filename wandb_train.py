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
from src import Loss, LossVisualization, EarlyStopping
from utils import setting_device, data_split, create_dataloaders, get_scheduler, L1_regularization
from model import model_selector

import torch, gc


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='hyperparameters for training')
    parser.add_argument('--traindata_dir', type=str, default="./data/train")
    parser.add_argument('--traindata_info_file', type=str, default="./data/train.csv")
    parser.add_argument('--save_result_path', type=str, default='./train_result')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--img_size', type=str, default='224')
    parser.add_argument('--pretrained_model_path', type=str, required=True, help='Path to the pretrained model file')
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
        lambda_L1: float,
        early_stopping_patience: int,
        early_stopping_delta: float
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
        self.lambda_L1 = lambda_L1
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

    def train_epoch(self) -> float:
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        scaler = GradScaler()
        train_predictions = []
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                if self.lambda_L1 > 0.0:
                    loss += L1_regularization(self.model, self.lambda_L1)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            train_preds = F.softmax(outputs, dim=1).argmax(dim=1)
            train_preds = (train_preds == targets)
            train_predictions.extend(train_preds.cpu().detach().numpy())

        return (total_loss / len(self.train_loader)), (sum(train_predictions) / len(train_predictions))

    def validate(self) -> float:
        self.model.eval()
        
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        val_predictions = []
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
                val_preds = F.softmax(outputs, dim=1).argmax(dim=1)
                val_preds = (val_preds == targets)
                val_predictions.extend(val_preds.cpu().detach().numpy())

        return (total_loss / len(self.val_loader)), (sum(val_predictions) / len(val_predictions))

    def train(self) -> None:
        loss_visualizer = LossVisualization(save_dir=self.result_path, hyperparameters={
                                            'model': self.model_name,
                                            'lr': self.optimizer.param_groups[0]['lr'],
                                            'batch_size': self.train_loader.batch_size,
                                            'epochs': self.epochs,
                                            'scheduler_gamma': self.scheduler.gamma if hasattr(self.scheduler, 'gamma') else wandb.config.scheduler_gamma,
                                            'lr_decay': wandb.config.lr_decay,
                                            'lambda_L1': self.lambda_L1,
                                            'L2 weight_decay': wandb.config.L2
                                            })
        early_stopper = EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_delta)
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

            self.scheduler.step()
            loss_visualizer.update(train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            early_stopper(val_loss)
            print(f"EarlyStopping Counter: {early_stopper.counter}/{self.early_stopping_patience}")

            if early_stopper.early_stop:
                print("Early stopping!")
                break

        loss_visualizer.save_plot()
        return val_loss  # Return the final validation loss for hyperparameter optimization

def run_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        opt = get_args()
        device = setting_device()
        transform_selector = TransformSelector(transform_type="albumentations")

        train_df, val_df, num_classes = data_split(opt.traindata_info_file)
        train_loader, val_loader = create_dataloaders(
            train_df, val_df, opt.traindata_dir, config.batch_size, 
            transform_selector, img_size=int(opt.img_size)
        )

        # Load the pretrained model
        model = model_selector(model_type='timm', num_classes=num_classes, model_name=opt.model_name, pretrained=False)
        model.load_state_dict(torch.load(opt.pretrained_model_path))
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.L2)
        scheduler = get_scheduler(optimizer, train_loader, config.lr_decay, config.scheduler_gamma)
        loss_fn = Loss()

        trainer = Trainer(
            model=model, 
            device=device, 
            train_loader=train_loader,
            val_loader=val_loader, 
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn, 
            epochs=config.epochs,
            result_path=opt.save_result_path,
            model_name=opt.model_name,
            lambda_L1=config.L1,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_delta=config.early_stopping_delta
        )

        final_val_loss = trainer.train()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return final_val_loss

def main():
    opt = get_args()
    
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {'min': 1e-5, 'max': 1e-2},
            'batch_size': {'values': [4, 8, 16]},
            'epochs': {'min': 2, 'max': 10},
            'scheduler_gamma': {'min': 0.1, 'max': 0.5},
            'lr_decay': {'min': 1, 'max': 6},
            'L1': {'min': 0.0, 'max': 1e-08},
            'L2': {'min': 0.0, 'max': 1e-05},
            'early_stopping_patience': {'min': 3, 'max': 10},
            'early_stopping_delta': {'min': 0.0, 'max': 0.05}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="model_optimization")
    wandb.agent(sweep_id, function=run_train, count=5)  # Run 10 trials

if __name__ == '__main__':
    main()