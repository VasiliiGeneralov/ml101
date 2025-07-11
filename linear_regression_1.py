#!/bin/python3

import torch
import sklearn
import numpy as np


def f(x):
    return 2 * x + 5


class LR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1, True)

    def forward(self, x):
        return self.fc1(x)


X_MIN = -10
X_MAX = 10
EPOCHS = 100
SAMPLES = 100


if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    y = f(x) + np.random.normal(0, 7.5, len(x))

    X = torch.FloatTensor(x).reshape(-1, 1)
    Y = torch.FloatTensor(y).reshape(-1, 1)

    mod = LR()
    opt = torch.optim.SGD(mod.parameters(), lr=0.01)
    obj = torch.nn.MSELoss()

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)

    train_hst = []
    test_hst = []
    for _ in range(EPOCHS):
        mod.zero_grad()
        pred = mod.forward(X_train)
        loss = obj(pred, Y_train)
        train_hst.append(float(loss))
        loss.backward()
        opt.step()

        pred = mod.forward(X_test)
        loss = obj(pred, Y_test)
        test_hst.append(float(loss))
