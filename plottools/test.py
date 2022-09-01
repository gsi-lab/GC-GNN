# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 06:53
# @Author  : FAN FAN
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import numpy as np
a = np.random.randn(10, 77)
print(a[0, :].reshape(1, -1).shape)
a_0 = a[0, :].reshape(1, -1)
print((a_0 @ np.linalg.inv(a.T @ a) @ a_0.T))
