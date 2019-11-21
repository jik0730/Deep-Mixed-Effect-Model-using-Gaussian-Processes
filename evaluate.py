import math

import torch
from tqdm import tqdm
from pyro.infer import TraceMeanField_ELBO
from src.model import DMEGP
from src.means import MLP, MLP_embed, RNN, RNN_embed
from src.data_loader import fetch_dataloaders_PhysioNet


def evaluate(model, dataloader, loss_fn, n_adapt, inner_lr, is_seq=False):
    rmse, loss = 0., 0.
    count = 0
    for X, y in tqdm(dataloader):
        if is_seq:
            # sequential predictions
            X = X.squeeze(0)
            y = y.view(-1)
            for t in range(1, X.size(0)):
                # define train data and test data
                if t == 0:
                    test_X, test_y = X[t:t + 1, :], y[t:t + 1]
                    pred = model.forward_mean(test_X)
                else:
                    train_X, train_y = X[0:t, :], y[0:t]
                    test_X, test_y = X[t:t + 1, :], y[t:t + 1]
                    # model prediction
                    pred, pred_var, loss_step = model(train_X, train_y, test_X,
                                                      loss_fn, inner_lr,
                                                      n_adapt)
                    loss += loss_step.item()
                rmse += (pred.item() - test_y.item())**2
                count += 1
        else:
            # last only predictions
            X = X.squeeze(0)
            y = y.view(-1)
            # define train data and test data
            train_X, train_y = X[:-1, :], y[:-1]
            test_X, test_y = X[-1:, :], y[-1:]
            # model prediction
            pred, pred_var, loss_step = model(train_X, train_y, test_X,
                                              loss_fn, inner_lr, n_adapt)
            rmse += (pred.item() - test_y.item())**2
            loss += loss_step.item()
            count += 1

    rmse /= count
    loss /= count
    rmse = math.sqrt(rmse)
    return rmse, loss


if __name__ == '__main__':
    """
    Evaluate DMEGP on PHYSIONET test dataset.
    """
    # system configuration
    torch.manual_seed(1)

    # data configuration
    dataloaders = fetch_dataloaders_PhysioNet(['val', 'test'], 'HR')
    dataloader_val = dataloaders['val']
    dataloader_test = dataloaders['test']

    # load a model
    model_path = './experiments/physionet/dmegp_physionet.pth'
    cpt = torch.load(model_path)
    embed_fn = MLP_embed(cpt['input_dim'], cpt['feature_dim'])
    mean_fn = MLP(cpt['feature_dim'], cpt['hidden_dim'], cpt['output_dim'])
    model = DMEGP(cpt['input_dim'],
                  feature_dim=cpt['feature_dim'],
                  mean_fn=mean_fn,
                  embed_fn=embed_fn)
    model.load_state_dict(cpt['state_dict'])
    elbo = TraceMeanField_ELBO()
    loss_fn = elbo.differentiable_loss

    # evaluate the model
    rmse_val, loss_val = evaluate(model, dataloader_val, loss_fn,
                                  cpt['n_adapt'], cpt['inner_lr'])
    rmse_te, loss_te = evaluate(model, dataloader_test, loss_fn,
                                cpt['n_adapt'], cpt['inner_lr'])
    print('[Val] RMSE={:05.3f} and loss={:05.3f}'.format(rmse_val, loss_val))
    print('[Test] RMSE={:05.3f} and loss={:05.3f}'.format(rmse_te, loss_te))
