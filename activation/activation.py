#!/bin/python3

import torch
import sklearn
import numpy as np
import pandas as pd

X_MIN = -10
X_MAX = 10
SAMPLES = 100

if __name__ == "__main__":
    x = torch.FloatTensor(np.linspace(X_MIN, X_MAX, SAMPLES))
    s = torch.nn.Sigmoid()(x)
    r = torch.nn.ReLU()(x)
    t = torch.nn.Tanh()(x)
    g = torch.nn.GELU()(x)

    data = np.ndarray(shape=(SAMPLES, 5))
    data[:, 0] = x
    data[:, 1] = s
    data[:, 2] = r
    data[:, 3] = t
    data[:, 4] = g

    df = pd.DataFrame(data, columns=["X", "Sigmoid", "ReLU", "tanh", "GELU"])
    df.to_csv("output.csv", index=False)
