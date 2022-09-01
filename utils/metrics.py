# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 11:09
# @Author  : FAN FAN
# @Site    : 
# @File    : metrics.py
# @Software: PyCharm
import torch
import numpy as np
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from math import sqrt


class Metrics(object):
    def __init__(self, y, y_pred, p):
        super().__init__()
        self.RMSE = self.cal_RMSE(y, y_pred)
        self.AIC = self.cal_AIC(y, y_pred, p)
        self.BIC = self.cal_BIC(y, y_pred, p)
        self.R2 = r2_score(y, y_pred)
        self.MAE = MAE(y, y_pred)
        self.SSE = self.cal_SSE(y, y_pred)
        self.MAPE = MAPE(y, y_pred)
        self.MaxErr = max_error(y, y_pred)
        self.residual = y_pred - y

    def cal_SSE(self, y, y_pred):
        sum_squared = torch.sum((y - y_pred) ** 2)
        return sum_squared.numpy()

    def cal_RMSE(self, y, y_pred):
        mean_squared = MSE(y, y_pred)
        return sqrt(mean_squared)

    def cal_AIC(self, y, y_pred, p):
        n = len(y)
        aic_score = n * np.log(MSE(y, y_pred)) + 2 * p
        return aic_score

    def cal_BIC(self, y, y_pred, p):
        n = len(y)
        bic_score = n * np.log(MSE(y, y_pred)) + p * np.log(n)
        return bic_score

    def cal_MaxErr(self, y, y_pred):
        return torch.max(torch.abs(y - y_pred), dim=0)
