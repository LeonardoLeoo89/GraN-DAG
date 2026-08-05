import argparse
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

from .train import train_loop

class DAGGNNTrainer:
    """
    Object-Oriented native interface for DAG-GNN.
    Acts as a lightweight shell around the functional train.py
    """
    
    def __init__(self, **kwargs):
        self.args = argparse.Namespace(
            epochs=300,
            batch_size=100,
            lr=3e-3,
            encoder_hidden=64,
            decoder_hidden=64,
            k_max_iter=100,
            graph_threshold=0.3,
            tau_A=0.0,
            lambda_A=0.0,
            c_A=1.0,
            h_tol=1e-8,
            use_A_connect_loss=0,
            use_A_positiver_loss=0,
            encoder='mlp',
            decoder='mlp',
            optimizer='Adam',
            lr_decay=200,
            gamma=1.0,
            encoder_dropout=0.0,
            decoder_dropout=0.0,
            factor=True,
            cuda=True,
            x_dims=1,
            z_dims=1
        )
        
        for key, value in kwargs.items():
            if key == 'encoder_type': key = 'encoder'
            if key == 'decoder_type': key = 'decoder'
            if key == 'optimizer_type': key = 'optimizer'
            setattr(self.args, key, value)
            
        self.args.cuda = self.args.cuda and torch.cuda.is_available()
        
    def fit(self, X: np.ndarray, mask_A: np.ndarray = None) -> np.ndarray:
        """
        Runs the DAG-GNN training pipeline natively.
        """
        n_samples, n_vars = X.shape
        
        if X.ndim == 2:
            X = X[..., np.newaxis]
            
        feat_train = torch.FloatTensor(X)
        train_data = TensorDataset(feat_train, feat_train)
        train_loader = DataLoader(train_data, batch_size=self.args.batch_size)
        
        if mask_A is None:
            mask_A = np.ones((n_vars, n_vars)) - np.eye(n_vars)
            
        graph = train_loop(self.args, train_loader, mask_A, n_vars)
        return graph
