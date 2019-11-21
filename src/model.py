import copy

import torch
import torch.nn as nn
import numpy as np
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models import GPRegression
from pyro.contrib.gp.kernels import Warping
from src.means import Warping_mean
from src.utils import get_mean_params, get_kernel_params


class DMEGP(nn.Module):
    """
    This class represents core model in DMEGP for regression.
    """
    def __init__(self,
                 input_dim,
                 feature_dim=None,
                 mean_fn=None,
                 embed_fn=None):
        super(DMEGP, self).__init__()
        # store params
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        # define mean function and embedding function
        self.mean_fn = mean_fn
        self.embed_fn = embed_fn

        # define kernel function
        if embed_fn == None:
            feature_dim = input_dim
            kernel = RBF(feature_dim, lengthscale=torch.ones(feature_dim))
        else:
            kernel = RBF(feature_dim, lengthscale=torch.ones(feature_dim))
            kernel = Warping(kernel, iwarping_fn=self.embed_fn)
            if mean_fn != None:
                self.mean_fn = Warping_mean(self.mean_fn, self.embed_fn)

        # define gaussian process regression model
        self.gp_model = GPRegression(
            X=torch.ones(1, feature_dim),  # dummy
            y=None,
            kernel=kernel,
            mean_function=self.mean_fn)

    def forward(self, X_sup, Y_sup, X_que, loss_fn, lr, n_adapt):
        """
        Return p(Y_que|X_sup, Y_sup, X_que).
        """
        gp_clone = self.define_new_GP()
        params_clone = copy.deepcopy(self.gp_model.state_dict())
        gp_clone.load_state_dict(params_clone)
        gp_clone.set_data(X_sup, Y_sup)
        optimizer = torch.optim.Adam(get_kernel_params(gp_clone), lr=lr)
        for _ in range(n_adapt):
            optimizer.zero_grad()
            loss = loss_fn(gp_clone.model, gp_clone.guide)
            loss.backward()
            optimizer.step()
        y_loc, y_var = gp_clone(X_que, noiseless=False)
        return y_loc, loss

    def forward_var(self, X_sup, Y_sup, X_que, loss_fn, lr, n_adapt):
        """
        Return p(Y_que|X_sup, Y_sup, X_que).
        """
        gp_clone = self.define_new_GP()
        params_clone = copy.deepcopy(self.gp_model.state_dict())
        gp_clone.load_state_dict(params_clone)
        gp_clone.set_data(X_sup, Y_sup)
        optimizer = torch.optim.Adam(get_kernel_params(gp_clone), lr=lr)
        for _ in range(n_adapt):
            optimizer.zero_grad()
            loss = loss_fn(gp_clone.model, gp_clone.guide)
            loss.backward()
            optimizer.step()
        y_loc, y_var = gp_clone(X_que, noiseless=False)
        return y_loc, y_var

    def forward_mean(self, X_que):
        """
        Return \mu(X_que) that is predictions of global mean function.
        """
        mean_preds = self.gp_model.mean_fn(X_que)
        mean_preds = torch.transpose(mean_preds, -1, -2)
        return mean_preds

    def step(self, X_sup, Y_sup, mean_optim, gp_optim, loss_fn, lr, n_adapt):
        """
        Optimize gp 1 step with single time series data.
        Following new optimization step.
        1. adaptation
        2. alternating w and theta
        """
        self.gp_model.set_data(X_sup, Y_sup)

        mean_optim.zero_grad()
        loss = loss_fn(self.gp_model.model, self.gp_model.guide)
        loss.backward()
        mean_optim.step()

        gp_optim.zero_grad()
        loss = loss_fn(self.gp_model.model, self.gp_model.guide)
        loss.backward()
        gp_optim.step()

        return loss

    def define_new_GP(self):
        # define kernel function
        if self.embed_fn == None:
            feature_dim = self.input_dim
            kernel = RBF(feature_dim, lengthscale=torch.ones(feature_dim))
        else:
            feature_dim = self.feature_dim
            embed_fn = copy.deepcopy(self.embed_fn)
            kernel = RBF(feature_dim, lengthscale=torch.ones(feature_dim))
            kernel = Warping(kernel, iwarping_fn=embed_fn)
            if self.mean_fn != None:
                mean_fn = copy.deepcopy(self.mean_fn)

        # define gaussian process regression model
        gp_model = GPRegression(
            X=torch.ones(1, feature_dim),  # dummy
            y=None,
            kernel=kernel,
            mean_function=mean_fn)
        return gp_model