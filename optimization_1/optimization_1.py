#!/bin/python3

import torch
import sklearn
import numpy as np
import pandas as pd


X_MIN = -10
X_MAX = 10
Y_MIN = -10
Y_MAX = 10
SAMPLES = 100


if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    y = np.linspace(Y_MIN, Y_MAX, SAMPLES)

    data = np.ndarray(shape=(SAMPLES, 4))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = np.vectorize(lambda x: x**2)(x)
    data[:, 3] = np.vectorize(lambda x, y: (x**2 + y**2) ** (1 / 2))(x, y)
    df = pd.DataFrame(data, columns=["X", "Y", "P2d", "P3d"])
    df.to_csv("output.csv", index=False)
