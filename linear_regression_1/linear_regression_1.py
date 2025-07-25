#!/bin/python3

import torch
import sklearn
import numpy as np
import pandas as pd


def f(x):
    return 2 * x + 5


X_MIN = -10
X_MAX = 10
EPOCHS = 100
SAMPLES = 100


class LR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1, True)

    def forward(self, x):
        return self.fc1(x)


if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    y = f(x) + np.random.normal(0, 7.5, len(x))

    X = torch.FloatTensor(x).reshape(-1, 1)
    Y = torch.FloatTensor(y).reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.3
    )

    train_hst = []
    test_hst = []

    mod = LR()
    obj = torch.nn.MSELoss()
    opt = torch.optim.SGD(mod.parameters(), lr=0.01)
    for _ in range(EPOCHS):
        mod.zero_grad()
        pred = mod.forward(X_train)
        loss = obj(pred, Y_train)
        loss.backward()
        opt.step()
        train_hst.append(float(loss))

        pred = mod.forward(X_test)
        loss = obj(pred, Y_test)
        test_hst.append(float(loss))

    df = pd.DataFrame(train_hst)
    df.to_csv("train_loss.csv")

    df = pd.DataFrame(test_hst)
    df.to_csv("test_loss.csv")

    data = np.ndarray(shape=(SAMPLES, 4))
    data[:, 0] = torch.squeeze(X, 1).detach().numpy()
    data[:, 1] = torch.squeeze(f(X), 1).detach().numpy()
    data[:, 2] = torch.squeeze(Y, 1).detach().numpy()
    data[:, 3] = torch.squeeze(mod.forward(X), 1).detach().numpy()
    df = pd.DataFrame(data, columns=["X", "Y", "Ynoise", "Ypred"])
    df.to_csv("output.csv", index=False)
