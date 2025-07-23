#!/bin/python3

import torch
import sklearn
import numpy as np
import pandas as pd


X_MIN = -10
X_MAX = 10
Y_MIN = -10
Y_MAX = 10
SAMPLES = 50


def f2d(x):
    return x**2


def f3d(x, y):
    return x**2 + y**2


if __name__ == "__main__":
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    data = np.ndarray(shape=(SAMPLES, 2))
    data[:, 0] = x
    data[:, 1] = f2d(x)
    df = pd.DataFrame(data)
    df.to_csv("output2d.csv", index=False, header=False)

    x = np.repeat(x, SAMPLES)
    y = np.linspace(Y_MIN, Y_MAX, SAMPLES)
    y = np.tile(y, SAMPLES)
    data = np.ndarray(shape=(SAMPLES * SAMPLES, 3))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = f3d(x, y)
    df = pd.DataFrame(data)
    df.to_csv("output3d.csv", index=False, header=False)
