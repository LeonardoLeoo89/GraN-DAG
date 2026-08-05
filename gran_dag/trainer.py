import os
import argparse
import numpy as np
import torch

from .models.learnables import LearnableModel_NonLinGauss, LearnableModel_NonLinGaussANM
from .train import pns, train, to_dag
from .data import DataManagerArray


class GraNDAGTrainer:
    """
    Object-Oriented native interface for GraN-DAG.
    Acts as a lightweight shell around the functional train.py
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the GraN-DAG trainer with hyperparameters.
        Default values are based on the original GraN-DAG CLI arguments.
        """
        self.opt = argparse.Namespace(
            i_dataset=1,
            train=True,
            to_dag=True,
            model="NonLinGauss",
            num_layers=2,
            hid_dim=10,
            nonlin="leaky-relu",
            norm_prod="none",
            square_prod=False,
            pns=False,
            pns_thresh=0.75,
            num_neighbors=None,
            cam_pruning=False,
            retrain=False,
            random_seed=42,
            lr=1e-3,
            lr_reinit=None,
            gpu=torch.cuda.is_available(),
            float=False,
            train_samples=0.8,
            test_samples=None,
            normalize_data=False,
            num_train_iter=100000,
            train_batch_size=64,
            mu_init=0.001,
            lambda_init=0.0,
            optimizer="rmsprop",
            edge_clamp_range=0.0001,
            no_w_adjs_log=True,
            stop_crit_win=100,
            omega_lambda=0.0001,
            omega_mu=0.9,
            h_threshold=1e-8,
            plot_freq=1000000,
            jac_thresh=True
        )
        
        # Override with user provided kwargs
        for key, value in kwargs.items():
            setattr(self.opt, key, value)
            
        if self.opt.lr_reinit is None:
            self.opt.lr_reinit = self.opt.lr

    def fit(self, data_array: np.ndarray, adjacency_array: np.ndarray = None):
        """
        Runs the GraN-DAG pipeline natively.
        """
        torch.manual_seed(self.opt.random_seed)
        np.random.seed(self.opt.random_seed)

        if self.opt.gpu:
            if self.opt.float:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            if self.opt.float:
                torch.set_default_tensor_type('torch.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.DoubleTensor')
                
        n_samples, n_vars = data_array.shape
        self.opt.num_vars = n_vars

        if self.opt.model == "NonLinGauss":
            model = LearnableModel_NonLinGauss(self.opt.num_vars, self.opt.num_layers, self.opt.hid_dim, nonlin=self.opt.nonlin,
                                               norm_prod=self.opt.norm_prod, square_prod=self.opt.square_prod)
        elif self.opt.model == "NonLinGaussANM":
            model = LearnableModel_NonLinGaussANM(self.opt.num_vars, self.opt.num_layers, self.opt.hid_dim, nonlin=self.opt.nonlin,
                                                  norm_prod=self.opt.norm_prod, square_prod=self.opt.square_prod)
        else:
            raise ValueError("model has to be in {NonLinGauss, NonLinGaussANM}")

        train_data = DataManagerArray(data_array, adjacency=adjacency_array, train_samples=self.opt.train_samples, test_samples=self.opt.test_samples, train=True,
                                     normalize=self.opt.normalize_data, random_seed=self.opt.random_seed)
        test_data = DataManagerArray(data_array, adjacency=adjacency_array, train_samples=self.opt.train_samples, test_samples=self.opt.test_samples, train=False,
                                    normalize=self.opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                    random_seed=self.opt.random_seed)

        if self.opt.pns:
            num_neighbors = self.opt.num_neighbors if self.opt.num_neighbors is not None else self.opt.num_vars
            model = pns(model, train_data, test_data, num_neighbors, self.opt.pns_thresh)

        if self.opt.train:
            model = train(model, train_data, test_data, self.opt)

        if self.opt.to_dag:
            model = to_dag(model, train_data, test_data, self.opt)

        return model
