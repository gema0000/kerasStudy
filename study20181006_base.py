# coding: utf-8

#1. 문제 정의하기
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM

import numpy as np
# import pandas as pd
# import keras
# import tensorflow as tf

np.random.seed(777)

# 데이터 생성
x = np.array([1,2,3,4,5])   # (5,)
y = np.array([1,2,4,7,5])   # (5,)

print(type(x))
print(x.shape)

# 훈련과 검증 분리
x_train = x[:3]  # 1,2,3
y_train = y[:3]
x_test = x[3:]   # 4,5
y_test = y[3:]

# print(x_train)
print(x_train.shape)
print(x_test.shape)


# 모델 구성하기

model = Sequential()
model.add(Dense(7, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()


model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print(loss, acc)

print(model.predict(x_test))


