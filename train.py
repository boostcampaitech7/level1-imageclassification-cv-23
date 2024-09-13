import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from data import CustomDataset 

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='hyperparameters for training')
    parser.add_argument('--epochs', type = int, default=10)
    parser.add_argument('--batch_size', type = int, default=64)
    parser.add_argument('--lr', type = int, default=0.001)
    parser.add_argument('--scheduler_step_size', type = int, default=30)
    parser.add_argument('--scheduler_gamma', type = int, default=0.1)
    parser.add_argument('--lr_decay', type = int, default=2)
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
        result_path: str
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수

    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.val_loader)

    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step()


if __name__ == '__main__':
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    save_result_path = "./train_result"

    # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    train_info = pd.read_csv(traindata_info_file)

    # 총 class의 수를 측정.
    num_classes = len(train_info['target'].unique())

    # 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target']
    )

    # 학습에 사용할 Dataset을 선언.
    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
    )

    # 학습에 사용할 DataLoader를 선언.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False
    )