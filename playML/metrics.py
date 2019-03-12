#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """根据y_true和y_predict确定当前模型的准确度"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """计算y_true与y_predict之间的 MSE 均方误差"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true与y_predict之间的 RMSE 均方根误差"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return sqrt(np.sum((y_true - y_predict) ** 2) / len(y_true))


def mean_absolute_error(y_true, y_predict):
    """计算y_true与y_predict之间的 MAE 平均绝对误差"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true与y_predict之间的 R^2"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)