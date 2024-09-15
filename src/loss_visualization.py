import os
import matplotlib.pyplot as plt

class LossVisualization:
    def __init__(self, save_dir : str, save_file : str):

        self.save_dir = save_dir
        self.save_file = save_file
        self.train_loss = []
        self.val_loss = []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def update(self, train_loss, val_loss):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    def save_plot(self):
        plt.figure()
        plt.plot(self.train_loss, label='Train Loss', marker='o', linestyle='-')  # 훈련 손실
        plt.plot(self.val_loss, label='Validation Loss', marker='s', linestyle='--')  # 검증 손실
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        save_path = os.path.join(self.save_dir, self.save_file)
        plt.savefig(save_path)
        plt.close()
