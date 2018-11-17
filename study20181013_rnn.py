
import numpy as np
import pandas as pd
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from keras.utils import np_utils

# a = np.array([11,12,13,14,15,16,17,18,19,20])
a = np.array(range(1,100))

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
print(dataset.shape) #15, 5

from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify = cancer.target, random_state=66
# )


#입력과 출력을 분리시키기  5개와 1개로

x_data = dataset[:,0:4] 
y_data = dataset[:,4]   

# 표준화
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_data)
x_data2 = scaler.transform(x_data)

print(x_data.shape)     # 95, 4
x_data = np.reshape(x_data2, (95, 4, 1))
print(x_data2.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=66
)

print(x_train)
print(y_train)          # 23, 15, ...

print(x_train.shape)    # 11, 4, 1
print(y_train.shape)    # 11,
print(x_test.shape)     # 4,4,1

print("스케일 조정 후 특성별 최소값:\n {}".format(x_train.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n {}".format(x_train.max(axis=0)))
print("스케일 조정 후 특성별 최소값:\n {}".format(x_test.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n {}".format(x_test.max(axis=0)))


#전처리없이 진행
x_train2 = x_train
x_test2 = x_test


# 모델 구성하기
model = Sequential()

model.add(LSTM(32, input_shape=(4,1)))
# model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=2)
model.fit(x_train2, y_train, epochs= 200, batch_size=1, validation_data=(x_test2, y_test), verbose=2)

loss, acc = model.evaluate(x_test2, y_test)

print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_test2)
print(y_predict)


x_test2 = np.array([[[101],[102],[103],[104]], [[201],[202],[203],[204]]])
y_predict2 = model.predict(x_test2)
print(y_predict2)



'''
Epoch 500/500
 - 0s - loss: 1.7484e-04 - acc: 1.0000
6/6 [==============================] - 0s 25ms/step
[[17.670113]
 [14.723088]]
''' 




