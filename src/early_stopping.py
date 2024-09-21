class EarlyStopping:
    def __init__(self, patience:int, min_delta:float):
        self.patience = patience # early stopping을 기다리는 에폭 수
        self.min_delta = min_delta # early stopping을 결정할 최소 변화량
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True