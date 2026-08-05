import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import math

from .utils import get_triu_offdiag_indices, get_tril_offdiag_indices, encode_onehot
from .utils import nll_gaussian, kl_gaussian_sem, A_connect_loss, A_positive_loss, matrix_poly
from .modules import MLPEncoder, SEMEncoder, MLPDecoder, SEMDecoder

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

    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


def train_loop(args, train_loader, mask_A, num_nodes):
    # Generate off-diagonal interaction graph
    off_diag = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    adj_A = np.zeros((num_nodes, num_nodes))

    if args.encoder == 'mlp':
        encoder = MLPEncoder(num_nodes * args.x_dims, args.x_dims, args.encoder_hidden,
                             int(args.z_dims), adj_A, mask_A,
                             batch_size=args.batch_size,
                             do_prob=args.encoder_dropout, factor=args.factor).double()
    elif args.encoder == 'sem':
        encoder = SEMEncoder(num_nodes * args.x_dims, args.encoder_hidden,
                             int(args.z_dims), adj_A, mask_A,
                             batch_size=args.batch_size,
                             do_prob=args.encoder_dropout, factor=args.factor).double()

    if args.decoder == 'mlp':
        decoder = MLPDecoder(num_nodes * args.x_dims,
                             args.z_dims, args.x_dims, encoder,
                             data_variable_size=num_nodes,
                             batch_size=args.batch_size,
                             n_hid=args.decoder_hidden,
                             do_prob=args.decoder_dropout).double()
    elif args.decoder == 'sem':
        decoder = SEMDecoder(num_nodes * args.x_dims,
                             args.z_dims, 2, encoder,
                             data_variable_size=num_nodes,
                             batch_size=args.batch_size,
                             n_hid=args.decoder_hidden,
                             do_prob=args.decoder_dropout).double()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    elif args.optimizer == 'LBFGS':
        optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

    if args.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    def train_step(train_loader, lambda_A, c_A, optimizer):
        nll_train = []
        kl_train = []

        encoder.train()
        decoder.train()

        optimizer, lr = update_optimizer(optimizer, args.lr, c_A)

        for batch_idx, (data, relations) in enumerate(train_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data).double(), Variable(relations).double()
            relations = relations.unsqueeze(2)

            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)
            edges = logits

            dec_x, output, adj_A_tilt_decoder = decoder(data, edges, num_nodes * args.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)

            target = data
            preds = output
            variance = 0.

            loss_nll = nll_gaussian(preds, target, variance)
            loss_kl = kl_gaussian_sem(logits)

            loss = loss_kl + loss_nll

            one_adj_A = origin_A
            sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

            if args.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if args.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

            h_A = _h_A(origin_A, num_nodes)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A * origin_A) + sparse_loss

            loss.backward()
            optimizer.step()

            myA.data = stau(myA.data, args.tau_A * lr)

            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            
        scheduler.step()
        return np.mean(kl_train) + np.mean(nll_train), origin_A

    best_ELBO_loss = np.inf
    best_ELBO_graph = np.zeros((num_nodes, num_nodes))
    c_A = args.c_A
    lambda_A = args.lambda_A
    h_A_old = np.inf

    k_max_iter = int(args.k_max_iter)

    for step_k in range(k_max_iter):
        h_A_new = torch.tensor(h_A_old if h_A_old != np.inf else 1.)
        while c_A < 1e+20:
            for epoch in range(args.epochs):
                ELBO_loss, origin_A = train_step(train_loader, lambda_A, c_A, optimizer)
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss

            if ELBO_loss > 2 * best_ELBO_loss:
                break

            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, num_nodes)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A *= 10
            else:
                break

        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= args.h_tol:
            break

    graph = origin_A.data.clone().cpu().numpy()
    return graph
