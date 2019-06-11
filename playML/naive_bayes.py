#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import jieba as jb
import numpy as np
import pandas as pd


def cut_review(data):
    """将样本进行分词处理"""
    result = []
    for d in data:
        result.append(list(filter(lambda s: s and s.strip(), jb.lcut(d))))
    return result


def create_vocab_list(dataSet):
    """将所有词条集合传入，得到一个所有不重复词条的集合字典"""
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """将词条集合转换为词条向量"""
    returnVec = np.zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


class NBClassifier:
    def __init__(self):
        self.data = None
        self.target = None

    def load_dataset(self, path):
        dataset = pd.read_csv(path, lineterminator='\n')
        data = dataset.values[:, 1]
        self.data = cut_review(data)

        self.target = dataset.values[:, 2].copy()
        self.target[dataset.values[:, 2] == 'Negative'] = 0
        self.target[dataset.values[:, 2] == 'Positive'] = 1

        return self

    def fit(self, trainMatrix, trainCategory):
        # 记录词条向量的个数
        numTrainDocs = len(trainMatrix)
        # 记录单个词条向量的长度即词条字典长度
        numWords = len(trainMatrix[0])
        # pPositive是所有词条向量中是Positive言论的概率
        pPositive = np.sum(trainCategory) / float(numTrainDocs)
        # 初始化Negative/Positive词条分布总和向量
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        # 初始化Negative/Positive言论中词条总个数
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                # 如果为Positive言论，记录所有词条向量累加后的总向量及所有Positive言论中总词条个数
                p1Num += trainMatrix[i]
                p1Denom += np.sum(trainMatrix[i])
            else:
                # 如果为Negative言论，记录所有词条向量累加后的总向量及所有Negative言论中总词条个数
                p0Num += trainMatrix[i]
                p0Denom += np.sum(trainMatrix[i])

        p1Vect = p1Num / p1Denom
        p0Vect = p0Num / p0Denom

        # 对每个元素取对数
        for i in range(len(p1Vect)):
            p1Vect[i] = np.log(p1Vect[i])
            p0Vect[i] = np.log(p0Vect[i])

        return pPositive, p1Vect, p0Vect

    def predict(self, vec2Classify, p0Vec, p1Vec, pPositive):
        p1 = np.sum(vec2Classify * p1Vec) + np.log(pPositive)
        p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pPositive)
        if p1 > p0:
            return 1
        else:
            return 0

    def score(self, y_test, y_predict):
        return np.sum(y_test == y_predict) / len(y_test)
