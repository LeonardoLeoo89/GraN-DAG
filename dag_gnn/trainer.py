# ==============================================================================
# ATTRIBUTION: 
# The core PyTorch training logic in this file was adapted from the DAG-GNN 
# repository (https://github.com/fishmoon1234/DAG-GNN). 
# ==============================================================================
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

from dag_gnn.modules import MLPEncoder, MLPDecoder, SEMEncoder, SEMDecoder # type: ignore
from dag_gnn.utils import get_triu_offdiag_indices, get_tril_offdiag_indices, encode_onehot # type: ignore
from dag_gnn.utils import nll_gaussian, kl_gaussian_sem, A_connect_loss, A_positive_loss, matrix_poly # type: ignore

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


class DAGGNNTrainer:
    """
    Clean wrapper for the DAG-GNN PyTorch training loop.
    Extracts the core logic from external/DAG-GNN/src/train.py.
    """
    
    def __init__(self,
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
                 encoder_type='mlp',
                 decoder_type='mlp',
                 optimizer_type='Adam',
                 lr_decay=200,
                 gamma=1.0,
                 encoder_dropout=0.0,
                 decoder_dropout=0.0,
                 factor=True,
                 cuda=False):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.k_max_iter = int(k_max_iter)
        self.graph_threshold = graph_threshold
        self.tau_A = tau_A
        self.lambda_A = lambda_A
        self.c_A = c_A
        self.h_tol = h_tol
        self.use_A_connect_loss = use_A_connect_loss
        self.use_A_positiver_loss = use_A_positiver_loss
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.optimizer_type = optimizer_type
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.factor = factor
        self.cuda = cuda and torch.cuda.is_available()
        
    def fit(self, X: np.ndarray, mask_A: np.ndarray = None) -> np.ndarray:
        """
        Runs the training loop and returns the learned adjacency matrix.
        X should be a 2D numpy array of shape (samples, variables).
        mask_A is an optional binary mask for Preliminary Neighborhood Selection (PNS).
        """
        n_samples, n_vars = X.shape
        x_dims = 1
        z_dims = 1
        
        # DAG-GNN's MLP layer requires inputs of shape (n_samples, n_vars, 1)
        if X.ndim == 2:
            X = X[..., np.newaxis]
            
        feat_train = torch.FloatTensor(X)
        train_data = TensorDataset(feat_train, feat_train)
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        
        off_diag = np.ones([n_vars, n_vars]) - np.eye(n_vars)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        rel_rec = torch.DoubleTensor(rel_rec)
        rel_send = torch.DoubleTensor(rel_send)

        adj_A = np.zeros((n_vars, n_vars))
        
        if mask_A is None:
            mask_A = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        if self.encoder_type == 'mlp':
            encoder = MLPEncoder(n_vars * x_dims, x_dims, self.encoder_hidden,
                                 int(z_dims), adj_A, mask_A,
                                 batch_size=self.batch_size,
                                 do_prob=self.encoder_dropout, factor=self.factor).double()
        elif self.encoder_type == 'sem':
            encoder = SEMEncoder(n_vars * x_dims, self.encoder_hidden,
                                 int(z_dims), adj_A, mask_A,
                                 batch_size=self.batch_size,
                                 do_prob=self.encoder_dropout, factor=self.factor).double()
        else:
            raise ValueError(f"Unknown encoder {self.encoder_type}")

        if self.decoder_type == 'mlp':
            decoder = MLPDecoder(n_vars * x_dims,
                                 z_dims, x_dims, encoder,
                                 data_variable_size=n_vars,
                                 batch_size=self.batch_size,
                                 n_hid=self.decoder_hidden,
                                 do_prob=self.decoder_dropout).double()
        elif self.decoder_type == 'sem':
            decoder = SEMDecoder(n_vars * x_dims,
                                 z_dims, 2, encoder,
                                 data_variable_size=n_vars,
                                 batch_size=self.batch_size,
                                 n_hid=self.decoder_hidden,
                                 do_prob=self.decoder_dropout).double()
        else:
            raise ValueError(f"Unknown decoder {self.decoder_type}")
            
        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr)
        elif self.optimizer_type == 'LBFGS':
            optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr)
        elif self.optimizer_type == 'SGD':
            optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr)
            
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay, gamma=self.gamma)
        
        if self.cuda:
            encoder.cuda()
            decoder.cuda()
            rel_rec = rel_rec.cuda()
            rel_send = rel_send.cuda()

        rel_rec = Variable(rel_rec)
        rel_send = Variable(rel_send)
        
        lambda_A = self.lambda_A
        c_A = self.c_A
        h_A_new = torch.tensor(1.)
        h_A_old = np.inf
        best_ELBO_loss = np.inf
        origin_A = None
        
        def train_step(epoch, lambda_A, c_A, optimizer):
            encoder.train()
            decoder.train()
            
            # scheduler.step() was moved AFTER optimizer.step() to comply with PyTorch 1.1.0+ warnings
            
            opt, lr_val = update_optimizer(optimizer, self.lr, c_A)
            
            kl_train = []
            nll_train = []
            
            for batch_idx, (data_batch, relations) in enumerate(train_loader):
                if self.cuda:
                    data_batch, relations = data_batch.cuda(), relations.cuda()
                data_batch, relations = Variable(data_batch).double(), Variable(relations).double()
                relations = relations.unsqueeze(2)
                
                opt.zero_grad()
                
                enc_x, logits, origin_A_val, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data_batch, rel_rec, rel_send)
                edges = logits
                
                dec_x, output, adj_A_tilt_decoder = decoder(data_batch, edges, n_vars * x_dims, rel_rec, rel_send, origin_A_val, adj_A_tilt_encoder, Wa)
                
                target = data_batch
                preds = output
                variance = 0.
                
                loss_nll = nll_gaussian(preds, target, variance)
                loss_kl = kl_gaussian_sem(logits)
                loss = loss_kl + loss_nll
                
                one_adj_A = origin_A_val
                sparse_loss = self.tau_A * torch.sum(torch.abs(one_adj_A))
                
                if self.use_A_connect_loss:
                    connect_gap = A_connect_loss(one_adj_A, self.graph_threshold, z_gap)
                    loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

                if self.use_A_positiver_loss:
                    positive_gap = A_positive_loss(one_adj_A, z_positive)
                    loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

                h_A_val = _h_A(origin_A_val, n_vars)
                loss += lambda_A * h_A_val + 0.5 * c_A * h_A_val * h_A_val + 100. * torch.trace(origin_A_val*origin_A_val) + sparse_loss
                
                loss.backward()
                opt.step()
                myA.data = stau(myA.data, self.tau_A * lr_val)
                
                nll_train.append(loss_nll.item())
                kl_train.append(loss_kl.item())
                
            scheduler.step()
            return np.mean(kl_train) + np.mean(nll_train), origin_A_val
            
        for step_k in range(self.k_max_iter):
            while c_A < 1e+20:
                for epoch in range(self.epochs):
                    ELBO_loss, origin_A = train_step(epoch, lambda_A, c_A, optimizer)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                
                if ELBO_loss > 2 * best_ELBO_loss:
                    break
                    
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, n_vars)
                
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A *= 10
                else:
                    break
                    
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()
            if h_A_new.item() <= self.h_tol:
                break
                
        graph = origin_A.data.clone().numpy()
        return graph
