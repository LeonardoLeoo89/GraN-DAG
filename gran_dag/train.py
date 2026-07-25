import time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
import torch

from .dag_optim import compute_constraint, compute_jacobian_avg, is_acyclic

EPSILON = 1e-8


def pns(model, train_data, test_data, num_neighbors, thresh):
    """Preliminary neighborhood selection"""
    model_adj = model.adjacency.detach().cpu().numpy()
    x_train, _ = train_data.sample(train_data.num_samples)
    x_test, _ = test_data.sample(test_data.num_samples)
    x = np.concatenate([x_train.detach().cpu().numpy(), x_test.detach().cpu().numpy()], 0)

    num_samples = x.shape[0]
    num_nodes = x.shape[1]
    
    for node in range(num_nodes):
        x_other = np.copy(x)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(x_other, x[:, node])
        selected_reg = SelectFromModel(reg, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(np.float64)
        model_adj[:, node] *= mask_selected

    with torch.no_grad():
        model.adjacency.copy_(torch.Tensor(model_adj))
    return model


def train(model, train_data, test_data, opt):
    """
    Applying augmented Lagrangian to solve the continuous constrained problem.
    """
    aug_lagrangian_ma = [0.0] * (opt.num_train_iter + 1)
    aug_lagrangians_val = []
    hs = []
    not_nlls = []

    mu = opt.mu_init
    lamb = opt.lambda_init

    if opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError("optimizer {} is not implemented".format(opt.optimizer))

    for iter in range(opt.num_train_iter):
        model.train()
        x, _ = train_data.sample(opt.train_batch_size)
        weights, biases, extra_params = model.get_parameters(mode="wbx")
        loss = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params))
        model.eval()

        w_adj = model.get_w_adj()
        h = compute_constraint(model, w_adj)

        aug_lagrangian = loss + 0.5 * mu * h ** 2 + lamb * h

        optimizer.zero_grad()
        aug_lagrangian.backward()
        optimizer.step()

        if opt.edge_clamp_range != 0:
            with torch.no_grad():
                to_keep = (w_adj > opt.edge_clamp_range).type(torch.Tensor)
                model.adjacency *= to_keep

        not_nlls.append(0.5 * mu * h.item() ** 2 + lamb * h.item())
        aug_lagrangian_ma[iter + 1] = aug_lagrangian_ma[iter] + 0.01 * (aug_lagrangian.item() - aug_lagrangian_ma[iter])

        if iter % opt.stop_crit_win == 0:
            with torch.no_grad():
                x, _ = test_data.sample(test_data.num_samples)
                loss_val = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params)).item()
                aug_lagrangians_val.append([iter, loss_val + not_nlls[-1]])

        if iter >= 2 * opt.stop_crit_win and iter % (2 * opt.stop_crit_win) == 0:
            t0, t_half, t1 = aug_lagrangians_val[-3][1], aug_lagrangians_val[-2][1], aug_lagrangians_val[-1][1]
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_lambda = -np.inf
            else:
                delta_lambda = (t1 - t0) / opt.stop_crit_win
        else:
            delta_lambda = -np.inf

        if h > opt.h_threshold:
            if abs(delta_lambda) < opt.omega_lambda or delta_lambda > 0:
                lamb += mu * h.item()

                hs.append(h.item())
                if len(hs) >= 2:
                    if hs[-1] > hs[-2] * opt.omega_mu:
                        mu *= 10

                with torch.no_grad():
                    gap_in_not_nll = 0.5 * mu * h.item() ** 2 + lamb * h.item() - not_nlls[-1]
                    aug_lagrangian_ma[iter + 1] += gap_in_not_nll
                    aug_lagrangians_val[-1][1] += gap_in_not_nll

                if opt.optimizer == "rmsprop":
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr_reinit)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr_reinit)
        else:
            with torch.no_grad():
                to_keep = (w_adj > 0).type(torch.Tensor)
                model.adjacency *= to_keep
            return model

    return model


def to_dag(model, train_data, test_data, opt):
    """
    1- If some entries of A_\phi == 0, also mask them
    2- Remove edges (from weaker to stronger) until a DAG is obtained.
    """
    model.eval()

    if opt.jac_thresh:
        A = compute_jacobian_avg(model, train_data, train_data.num_samples).t()
    else:
        A = model.get_w_adj()
    A = A.detach().cpu().numpy()

    with torch.no_grad():
        thresholds = np.unique(A)
        for step, t in enumerate(thresholds):
            to_keep = torch.Tensor(A > t + EPSILON)
            new_adj = model.adjacency * to_keep

            if is_acyclic(new_adj):
                model.adjacency.copy_(new_adj)
                break

    return model
