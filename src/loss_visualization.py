import os
import matplotlib.pyplot as plt

class LossVisualization:
    def __init__(self, save_dir: str, hyperparameters: dict):
        self.save_dir = save_dir
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.hyperparameters = hyperparameters

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def update(self, train_loss, val_loss, train_acc, val_acc):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

    def save_plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

        # 손실 곡선 (왼쪽 서브플롯)
        axs[0].plot(self.train_loss, label='Train Loss', marker='o', linestyle='-')
        axs[0].plot(self.val_loss, label='Validation Loss', marker='s', linestyle='--')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].legend()

        # 하이퍼파라미터 및 최종 손실 값 텍스트 표시 (오른쪽 서브플롯)
        final_train_loss = self.train_loss[-1] if self.train_loss else 0.0
        final_val_loss = self.val_loss[-1] if self.val_loss else 0.0
        final_train_acc = self.train_acc[-1] if self.train_acc else 0.0
        final_val_acc = self.val_acc[-1] if self.val_acc else 0.0
        hyperparam_text = (
            f'Hyperparameters:\n'
            f'lr: {self.hyperparameters["lr"]}\n'
            f'batch_size: {self.hyperparameters["batch_size"]}\n'
            f'epochs: {self.hyperparameters["epochs"]}\n'
            f'scheduler_gamma: {self.hyperparameters["scheduler_gamma"]}\n'
            f'lr_decay: {self.hyperparameters["lr_decay"]}\n'
            f'lambda L1: {self.hyperparameters["lambda_L1"]}\n'
            f'L2 weight_decay: {self.hyperparameters["L2 weight_decay"]}\n\n'
            f'Final Train Loss: {final_train_loss:.4f}\n'
            f'Final Val Loss: {final_val_loss:.4f}\n'
            f'Final Train Accuracy: {final_train_acc:.4f}\n'
            f'Final Val Accuracy: {final_val_acc:.4f}'
        )

        axs[1].axis('off') 
        axs[1].text(0.5, 0.5, hyperparam_text, fontsize=20, ha='center', va='center', wrap=True, bbox=dict(facecolor='white', alpha=0.5))  # 폰트 크기 및 배경 추가

        plt.tight_layout()
        if self.hyperparameters["fold"] is not None:
            save_path = os.path.join(self.save_dir, f'{self.hyperparameters["model"]}_loss_curve_train_{final_train_loss:.4f}_val_{final_val_loss:.4f}_fold_{self.hyperparameters["fold"]}.png')
        else:
            save_path = os.path.join(self.save_dir, f'{self.hyperparameters["model"]}_loss_curve_train_{final_train_loss:.4f}_val_{final_val_loss:.4f}.png')
        plt.savefig(save_path)
        plt.close()
