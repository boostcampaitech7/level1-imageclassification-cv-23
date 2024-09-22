import torch
import pandas as pd
from torch.utils.data import DataLoader
from data import CustomDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split


def setting_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def data_split(traindata_info_file : str):
    train_info = pd.read_csv(traindata_info_file)

    num_classes = len(train_info['target'].unique())

    train_df, val_df = train_test_split(
    train_info, 
    test_size=0.2,
    stratify=train_info['target']
    )
    return train_df, val_df, num_classes

def create_dataloaders(
    train_df, val_df, traindata_dir, batch_size, transform_selector, img_size
):
    train_transform = transform_selector.get_transform(img_size=img_size, is_train=True)
    val_transform = transform_selector.get_transform(img_size=img_size, is_train=False)

    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader

def test_dataloader(test_info, testdata_dir, batch_size, transform_selector, img_size):
    test_transform = transform_selector.get_transform(img_size = img_size, is_train=False)

    # 추론에 사용할 Dataset을 선언.
    test_dataset = CustomDataset(
        root_dir=testdata_dir,
        info_df=test_info,
        transform=test_transform,
        is_inference=True
    )

    # 추론에 사용할 DataLoader를 선언.
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    return test_loader

def get_scheduler(optimizer, train_loader, epochs_per_lr_decay, scheduler_gamma):
    # 한 epoch당 step 수 계산
    steps_per_epoch = len(train_loader)

    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_step_size, 
        gamma=scheduler_gamma
    )
    return scheduler

def L1_regularization(model, lambda_l1):
    l1_penalty = 0.0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    return lambda_l1 * l1_penalty