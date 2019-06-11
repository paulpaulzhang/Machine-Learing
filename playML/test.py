#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import jieba as jb

if __name__ == '__main__':
    trains_data = pd.read_csv('../train.csv', lineterminator='\n')
    review_id = trains_data.values[:, 0]
    review = trains_data.values[:, 1]
    label = trains_data.values[:, 2].copy()

    label[trains_data.values[:, 2] == "Negative"] = 0
    label[trains_data.values[:, 2] == "Positive"] = 1

    seg_list = list(filter(lambda s: s and s.strip(), jb.lcut(review[0])))
    print(seg_list)
