#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to y"
    assert 0.0 <= test_ratio <= 1.0, \
        "the number of test_ratio must be part of [0.0, 1.0]"

    if seed:
        np.random.seed(seed)

    shuffle_index = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_index[:test_size]
    train_indexes = shuffle_index[test_size:]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    return X_test, X_train, y_test, y_train
