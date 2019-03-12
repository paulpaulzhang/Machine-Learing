#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2"

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self

    def transform(self, X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        assert X.shape[1] == self.mean_.shape[0], \
            "the feature number of X must be equal to mean_ and std_"

        return (X - self.mean_) / self.scale_
