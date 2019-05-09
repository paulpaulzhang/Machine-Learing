#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import numpy as np


class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获得数据集的前n个主成分"""

        def demean(X):
            return X - np.mean(X, axis=0)

        def f(X, w):
            return np.sum(X.dot(w) ** 2) / len(X)

        def df(X, w):
            return X.T.dot(X.dot(w)) * 2 / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_components(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            cur_iters = 0

            while cur_iters < n_iters:
                gradient = df(X, w)
                last_w = w
                w += eta * gradient
                w = direction(w)
                if abs(f(X, last_w) - f(X, w)) < epsilon:
                    break
                cur_iters += 1
            return w

        X_pca = demean(X)
        initial_w = np.random.random(X.shape[1])
        self.components_ = np.empty((self.n_components, X.shape[1]))
        for i in range(self.n_components):
            w = first_components(X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w

            X_pca -= X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert self.components_.shape[1] == X.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射会原来的特征空间"""
        assert self.components_.shape[0] == X.shape[1]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
