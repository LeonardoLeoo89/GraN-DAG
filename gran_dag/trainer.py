import os
import tempfile
import argparse
import numpy as np
import torch

from .models.learnables import LearnableModel_NonLinGauss, LearnableModel_NonLinGaussANM
from .train import pns, train, to_dag, cam_pruning, retrain
from .data import DataManagerArray
from .utils.save import load, dump

def _print_metrics(stage, step, metrics, throttle=None):
    pass # Silent by default to avoid cluttering the wrapper, but can be overridden

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))

class GraNDAGTrainer:
    """
    Object-Oriented native interface for GraN-DAG.
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
            gpu=False,
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
            plot_freq=1000000, # Large number to effectively disable plotting
            jac_thresh=True
        )
        
        # Override with user provided kwargs
        for key, value in kwargs.items():
            setattr(self.opt, key, value)
            
        if self.opt.lr_reinit is None:
            self.opt.lr_reinit = self.opt.lr

    def fit(self, data_array: np.ndarray, adjacency_array: np.ndarray = None, metrics_callback=None, plotting_callback=None):
        """
        Runs the GraN-DAG pipeline natively.
        """
        torch.manual_seed(self.opt.random_seed)
        np.random.seed(self.opt.random_seed)

        if metrics_callback is None:
            metrics_callback = _print_metrics

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

        # Create a temporary directory for GraN-DAG's internal artifact writing
        with tempfile.TemporaryDirectory() as temp_dir:
            self.opt.exp_path = os.path.join(temp_dir, "exp")
            os.makedirs(self.opt.exp_path, exist_ok=True)
            self.opt.data_path = temp_dir # Unused by DataManagerArray but good for consistency

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

            if adjacency_array is not None:
                dump(train_data.adjacency.detach().cpu().numpy(), self.opt.exp_path, 'gt-adjacency')

            if self.opt.pns:
                num_neighbors = self.opt.num_neighbors if self.opt.num_neighbors is not None else self.opt.num_vars
                pns(model, train_data, test_data, num_neighbors, self.opt.pns_thresh, self.opt.exp_path, metrics_callback, plotting_callback)

            if self.opt.train:
                if file_exists(self.opt.exp_path, "pns"):
                    model = load(os.path.join(self.opt.exp_path, "pns"), "model.pkl")
                
                gt_adj = train_data.adjacency.detach().cpu().numpy() if train_data.adjacency is not None else None
                train(model, gt_adj, train_data, test_data, self.opt, metrics_callback, plotting_callback)

            if self.opt.to_dag:
                assert file_exists(self.opt.exp_path, "train"), "The /train folder is required to run to_dag"
                model = load(os.path.join(self.opt.exp_path, "train"), "model.pkl")
                to_dag(model, train_data, test_data, self.opt, metrics_callback, plotting_callback)

            # We return the loaded/trained model directly. 
            # The temporary directory is now automatically deleted.
            return model
