
import numpy as np
import pandas as pd
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from keras.utils import np_utils

a = np.array([11,12,13,14,15,16,17,18,19,20])

def split_5(seq, window_size):  # 데이터를 5개씩 자르기용.    # 입력이 5이고 5개씩 자르기
    aaa = []
    for i in range(len(a)-window_size +1):                 # 열
        subset = a[i:(i+window_size)]       # 0~5
        aaa.append([item for item in subset])
        # print(aaa)
    print(type(aaa))    
    return np.array(aaa)

dataset = split_5(a, window_size=5)     # 5씩 잘랏으니 (5, 6)가 된다. // window_size+1 만큼씩 잘라진다.
print("===========================")
print(dataset)
print(dataset.shape)


#입력과 출력을 분리시키기  5개와 1개로

x_train = dataset[:,0:4]
y_train = dataset[:,4]

x_train = np.reshape(x_train, (6, 4, 1))


print(x_train.shape)
print(y_train.shape)


# 모델 구성하기
model = Sequential()

model.add(LSTM(32, input_shape=(4,1)))
# model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=2)

loss, acc = model.evaluate(x_train, y_train)


x_test = np.array([[[101],[102],[103],[104]], [[201],[202],[203],[204]]])

y_predict = model.predict(x_test)


print(y_predict)

'''
Epoch 500/500
 - 0s - loss: 1.7484e-04 - acc: 1.0000
6/6 [==============================] - 0s 25ms/step
[[17.670113]
 [14.723088]]
''' 




