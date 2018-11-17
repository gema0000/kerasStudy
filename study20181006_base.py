# coding: utf-8

# 20181117 최종수정

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
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])   # (5,)
y = np.array([2,4,6,8,10,12,14,16,18,20,22,24])   # (5,)

print(type(x))
print(x.shape)








# 훈련과 검증 분리
x_train = x[:7]  # 1,2,3,4,5,6,7
y_train = y[:7]
x_test = x[7:]   # 8, 9, 10, 11, 12
y_test = y[7:]

# print(x_train)
print(x_train.shape)
print(x_test.shape)


# 모델 구성하기

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))     # 7일경우보다 5일경우 predict가 더 잘나옴.
model.add(Dense(3))
model.add(Dense(1, activation='relu'))


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()


model.fit(x_train, y_train, epochs= 500, batch_size=1, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)

print(model.predict(x_test))





'''
Epoch 1000/1000
1/3 [=========>....................] - ETA: 0s - loss: 0.0275 - acc:3/3 [==============================] - 0s 3ms/step - loss: 0.0562 -
acc: 1.0000 - val_loss: 3.0713 - val_acc: 0.0000e+00
2/2 [==============================] - 0s 997us/step
3.0712554454803467 0.0
[[5.338079 ]
 [6.8386216]]
'''
