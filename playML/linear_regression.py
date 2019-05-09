#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from .metrics import r2_score


class LinearRegression:

    def __init__(self):
        """初始化LinearRegression模型"""
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """通过训练数据集X_train和y_train训练LinearRegression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])

        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """通过训练数据集X_train和y_train使用梯度下降法训练LinearRegression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(X_b, y, theta):
            """计算损失函数"""
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(X_b, y, theta):
            """计算梯度值"""

            return X_b.T.dot((X_b.dot(theta) - y)) * 2 / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon=1e-8):
            """梯度下降法"""
            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(X_b, y, theta)
                last_theta = theta
                theta = theta - gradient * eta
                if abs(J(X_b, y, theta) - J(X_b, y, last_theta)) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t1=5, t0=50):
        """随机梯度下降法"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1, "n_iters must more than 1"

        def dJ_sgd(X_b_i, y_i, theta):
            return X_b_i.T.dot((X_b_i.dot(theta) - y_i)) * 2

        def sgd(X_b, y, initial_theta, n_iters=5, t1=5, t0=50):

            def learning_rate(t):
                return t1 / (t + t0)

            theta = initial_theta
            m = len(X_b)
            for cur_iters in range(n_iters):
                indexes = np.random.permutation(m)
                X_b = X_b[indexes]
                y = y[indexes]
                for i in range(len(X_b)):
                    theta = theta - learning_rate(cur_iters * m + i) * dJ_sgd(X_b[i], y[i], theta)

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t1, t0)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_small_gd(self, X_train, y_train, n_iters=5, k=5, t1=5, t0=50):
        """小批量梯度下降法"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1, "n_iters must more than 1"
        assert 1 <= k <= len(X_train)

        def dJ(X_b_k, y_k, theta, k=5):
            return X_b_k.T.dot(X_b_k.dot(theta) - y_k) * 2 / k

        def small_gd(X_b, y, initial_theta, n_iters=5, k=5, t1=5, t0=50):

            def learning_rate(t):
                return t1 / (t + t0)

            theta = initial_theta
            m = len(X_b)
            for cur_iters in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_shuffle = X_b[indexes]
                y_shuffle = y[indexes]

                for i in range(0, m, k):
                    if i + k < m:
                        theta = theta - learning_rate(cur_iters * m + i) * \
                                dJ(X_b_shuffle[i: i + k], y_shuffle[i:i + k], theta)
                    else:
                        theta = theta - learning_rate(cur_iters * m + i) * \
                                dJ(X_b_shuffle[i:], y_shuffle[i:], theta)
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = small_gd(X_b, y_train, initial_theta, n_iters, k, t1, t0)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """通过待测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.coef_ is not None and self.interception_ is not None, \
            "predict must before fit"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """通过X_test和y_test确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
