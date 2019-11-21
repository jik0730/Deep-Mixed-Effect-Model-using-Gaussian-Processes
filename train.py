import os
import math
import copy
import logging

import torch
from pyro.infer import TraceMeanField_ELBO
from tqdm import tqdm
from utils import set_logger
from src.model import DMEGP
from src.means import MLP, MLP_embed, RNN, RNN_embed
from src.data_loader import fetch_dataloaders_PhysioNet
from src.utils import get_mean_params, get_kernel_params


def train_single_epoch(model, dataloader, mean_optim, gp_optim, loss_fn,
                       inner_lr, n_adapt):
    loss = 0.
    for i, (train_X, train_y) in enumerate(dataloader):
        train_X = train_X[0]
        train_y = train_y.view(-1)
        loss_step = model.step(train_X, train_y, mean_optim, gp_optim, loss_fn,
                               inner_lr, n_adapt)
        loss += loss_step.item()
        if i % 100 == 0:
            logging.info("MINIBATCH_loss : {:05.3f}".format(loss_step.item()))
    loss /= (i + 1)
    logging.info("BATCH_loss : {:05.3f}".format(loss))
    return loss


def evaluate(model, dataloader, loss_fn, n_adapt, inner_lr, split):
    rmse, loss = 0., 0.
    for i, (X, y) in enumerate(dataloader):
        X = X.squeeze(0)
        y = y.view(-1)
        # define train data and test data for last prediction
        train_X, train_y = X[:-1, :], y[:-1]
        test_X, test_y = X[-1:, :], y[-1:]
        # model prediction
        pred, pred_var, loss_step = model(train_X, train_y, test_X, loss_fn,
                                          inner_lr, n_adapt)
        rmse += (pred.item() - test_y.item())**2
        loss += loss_step.item()

    rmse /= (i + 1)
    loss /= (i + 1)
    rmse = math.sqrt(rmse)
    logging.info(split.upper() + "_score : {:05.3f}".format(rmse))
    logging.info(split.upper() + "_loss : {:05.3f}".format(loss))
    return rmse, loss


def train_and_evaluate(model, dataloaders, mean_optim, gp_optim, loss_fn,
                       n_epochs, n_adapt, inner_lr):
    dl_train = dataloaders['train']
    dl_val = dataloaders['val']

    best_val_err = float('inf')
    best_state = None
    with tqdm(total=n_epochs) as t:
        for i in range(n_epochs):
            loss = train_single_epoch(model, dl_train, mean_optim, gp_optim,
                                      loss_fn, inner_lr, n_adapt)
            error_val, loss_val = evaluate(model, dl_val, loss_fn, n_adapt,
                                           inner_lr, 'val')
            is_best = error_val <= best_val_err
            if is_best:
                best_val_err = error_val
                best_state = copy.deepcopy(model.state_dict())
                logging.info("Found new best error at {}".format(i))
            t.set_postfix(loss_and_val_err='{:05.3f} and {:05.3f}'.format(
                loss, error_val))
            print('\n')
            t.update()
    return best_state


if __name__ == '__main__':
    """
    Train DMEGP on PHYSIONET dataset.
    """
    # system configuration
    torch.manual_seed(1)

    # logger configuration
    log_dir = 'experiments/physionet'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    set_logger(os.path.join(log_dir, 'train.log'))

    # data configuration
    dataloaders = fetch_dataloaders_PhysioNet(['train', 'val'], 'HR')

    # model configuration
    input_dim = 6
    feature_dim = 16
    hidden_dim = 32
    output_dim = 1
    embed_fn = MLP_embed(input_dim, feature_dim)
    mean_fn = MLP(feature_dim, hidden_dim, output_dim)
    model = DMEGP(input_dim,
                  feature_dim=feature_dim,
                  mean_fn=mean_fn,
                  embed_fn=embed_fn)

    # train configuration
    n_epochs = 10
    lr = 1e-3
    n_adapt = 10
    inner_lr = 1e-2
    l2_penalty = 1e-3
    mean_optim = torch.optim.Adam(get_mean_params(model.gp_model),
                                  lr=lr,
                                  weight_decay=l2_penalty)
    gp_optim = torch.optim.Adam(get_kernel_params(model.gp_model),
                                lr=lr,
                                weight_decay=l2_penalty)
    elbo = TraceMeanField_ELBO()
    loss_fn = elbo.differentiable_loss

    # train and evaluate
    best_state = train_and_evaluate(model, dataloaders, mean_optim, gp_optim,
                                    loss_fn, n_epochs, n_adapt, inner_lr)

    # save a model
    model_name = 'dmegp_physionet.pth'
    state = {
        'input_dim': input_dim,
        'feature_dim': feature_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'n_adapt': n_adapt,
        'inner_lr': inner_lr,
        'state_dict': best_state
    }
    torch.save(state, os.path.join(log_dir, model_name))
