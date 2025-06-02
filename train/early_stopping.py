import os
import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False, delta: float = 0.0) -> None:
        self.patience = patience # num of epochs to wait for validation loss improvement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta # minimum change in the monitored quantity to qualify as an improvement

    def __call__(self, score: float) -> None:
        if self.best_score is not None and score < self.best_score - self.delta:
            self.counter += 1 # it qualifies as a decline of the metric
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # since delta is kept small, a small decrease will not be considered a decline 
            self.best_score = score
            self.counter = 0
