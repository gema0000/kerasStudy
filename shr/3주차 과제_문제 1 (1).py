# coding: utf-8

# 0. 사용할 패키지 불러오기
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils

import numpy as np
import tensorflow as tf

tf.set_random_seed(123)  # for reproducibility


#x_train이 11~20, x_test가 21~30 일 경우  최대값은 30, 최소값은 11로 계산해야합니다.


# In[13]:


x_train = np.array([11,12,13,14,15,16,17,18,19,20])
y_train = np.array([40, 50, 60, 70])

x_test = np.array([21,22,23,24,25,26,27,28,29,30])

max_idx_value = 30
min_idx_value = 11

# # 입력값 정규화 시키기
nor = (x_train - min_idx_value)/ (max_idx_value-min_idx_value)

