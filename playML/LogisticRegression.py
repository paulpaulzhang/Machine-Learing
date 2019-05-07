#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import numpy as np
from .metrics import accuracy_score


class LogisticRegression:
    """initialize logistic regression"""

    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1 + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """according X_train and y_train to train the logistic regression model"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(X_b, y, theta):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) +
                               (1 - y) * np.log(1 - y_hat)) / len(X_b)
            except ValueError:
                return float('inf')

        def dJ(X_b, y, theta):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iters = 0
            while cur_iters < n_iters:
                gradient = dJ(X_b, y, theta)
                last_theta = theta
                theta = theta - gradient * eta

                if abs(J(X_b, y, last_theta) - J(X_b, y, theta)) < epsilon:
                    break

                cur_iters += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_proba(self, X_predict):
        assert self.coef_ is not None and self.interception_ is not None, \
            "predict must before fit"
        assert X_predict.shape[1] == self.coef_.shape[0], \
            "the size of X_test must be equal to the size of y_test"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        assert self.coef_ is not None and self.interception_ is not None, \
            "predict must before fit"
        assert X_predict.shape[1] == self.coef_.shape[0], \
            "the size of X_test must be equal to the size of y_test"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
