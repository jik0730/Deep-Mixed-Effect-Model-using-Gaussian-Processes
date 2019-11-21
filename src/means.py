import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        out = self.fc1(X)
        out = torch.sigmoid(out)
        out = self.fc2(out)
        return out


class MLP_embed(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(MLP_embed, self).__init__()
        self.fc1 = nn.Linear(input_dim, feature_dim)

    def forward(self, X):
        out = self.fc1(X)
        out = torch.sigmoid(out)
        return out


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = torch.unsqueeze(X, 0)
        out, _ = self.lstm(X)  # out.size() = (1, seq_length, hidden_dim)
        out = self.fc(out[0, :, :])
        return out


class RNN_embed(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN_embed, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)

    def forward(self, X):
        X = torch.unsqueeze(X, 0)
        out, _ = self.lstm(X)  # out.size() = (1, seq_length, hidden_dim)
        return out[0, :, :]


class Warping_mean(nn.Module):
    def __init__(self, mean_fn, iwarping_fn):
        super(Warping_mean, self).__init__()
        self.mean_fn = mean_fn
        self.iwarping_fn = iwarping_fn

    def forward(self, X):
        out = self.iwarping_fn(X)
        out = self.mean_fn(out)
        return out.transpose(-2, -1)