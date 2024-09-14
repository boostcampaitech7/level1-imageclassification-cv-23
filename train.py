import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TransformSelector
from src import Loss
from utils import setting_device, data_split, create_dataloaders, get_scheduler
from model import create_model

def get_args() -> argparse.Namespace:
    # hyperparameters argument parser
    parser = argparse.ArgumentParser(description='hyperparameters for training')
    parser.add_argument('--epochs', type = int, default=5)
    parser.add_argument('--batch_size', type = int, default=64)
    parser.add_argument('--lr', type = float, default=0.001)
    parser.add_argument('--scheduler_step_size', type = int, default=30)
    parser.add_argument('--scheduler_gamma', type = float, default=0.1)
    parser.add_argument('--lr_decay', type = int, default=2)

    # utils argument parser
    parser.add_argument('--traindata_dir', type = str, default="./data/train")
    parser.add_argument('--traindata_info_file', type = str, default="./data/train.csv")
    parser.add_argument('--save_result_path', type = str, default='./train_result')

    # model argument parser
    parser.add_argument('--model_name', type = str, default='resnet18')
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
        model_name: str
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
        self.model_name = model_name  # 모델 이름

    def save_model(self, epoch, loss) :
        # 모델 저장 경로 설정
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

def main(opt):
    
    device = setting_device()

    traindata_dir = opt.traindata_dir
    traindata_info_file = opt.traindata_info_file
    save_result_path = opt.save_result_path

    # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    train_df, val_df, num_classes = data_split(traindata_info_file)


    # 학습에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = "albumentations"
    )

    train_loader, val_loader = create_dataloaders(train_df, 
                                                val_df, 
                                                traindata_dir, 
                                                opt.batch_size, 
                                                transform_selector)

    model_name = opt.model_name

    model = create_model(model_type='timm', num_classes=num_classes, model_name=model_name, pretrained=True)
    model.to(device)

    # 학습에 사용할 optimizer를 선언하고, learning rate를 지정
    lr = opt.lr
    optimizer = optim.Adam(
    model.parameters(), 
    lr=lr
    )

    scheduler = get_scheduler(optimizer, train_loader, opt.lr_decay, opt.scheduler_gamma)
    

    # 학습에 사용할 Loss를 선언.
    loss_fn = Loss()

    epochs = opt.epochs
    # 앞서 선언한 필요 class와 변수들을 조합해, 학습을 진행할 Trainer를 선언. 
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=epochs,
        result_path=save_result_path,
        model_name=model_name
    )

    trainer.train()

if __name__ == '__main__':
    opt = get_args()
    main(opt)