# coding: utf-8

import numpy as np 
import pandas as pd 
import tensorflow as tf 

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils 

# a = np.array([11,12,13,14,15,16,17,18,19,20]) 

a=np.array(range(11,91))
batch_size=1
window_size=5  

#데이터셋 생성 함수
def split_5(seq, window_size):  # 데이터를 5개씩 자르기용.    # 입력이 5이고 5개씩 자르기 
    aaa = [] 
    for i in range(len(a)-window_size +1):                 # 열 
        subset = a[i:(i+window_size)]       # 0~5 
        aaa.append([item for item in subset]) 
        # print(aaa) 
    print(type(aaa))     
    return np.array(aaa) 

# 데이터 셋 생성하기 
dataset = split_5(a, window_size)     # 5씩 잘랏으니 (5, 6)가 된다. // window_size+1 만큼씩 잘라진다. 
# print("===========================") 
# print(dataset) 
# print(dataset.shape)    # 76, 5 

#입력과 출력을 분리시키기  5개와 1개로 

x_train = dataset[:,0:4] 
y_train = dataset[:,4] 

# print(x_train)
# print("+++++++++++++++++++++")
# max_num_value=81
# x_train=x_train/float(max_num_value)      # 중복하여 정규화하여 주석처리
 
x_train = np.reshape(x_train, (len(x_train), window_size-1, batch_size))  # 76,4,1 

 
x_test = x_train * 2 
y_test = y_train * 2 
 
# 정규화 

x_max_value=182
x_min_value=11 

x_train=(x_train-x_min_value)/(x_max_value-x_min_value)
x_test=(x_test-x_min_value)/(x_max_value-x_min_value)

print(x_train)
print('=====================')
print(x_test)  

 
print(x_train.shape)    # (76, 4, 1) 
print(y_train.shape)    # (76, ) 
  
print(x_test.shape)     # (76, 4, 1) 
print(y_test.shape)     # (76, ) 

# 모델 구성하기 
model = Sequential() 
model.add(LSTM(128, batch_input_shape=(1, 4, 1), stateful=True)) 
# model.add(Dropout(0,2))
# model.add(Dense(6, activation='relu')) 
# model.add(Dropout(0.2)) 
# model.add(Dense(5, activation='relu')) 

model.add(Dense(1)) 
 
model.summary() 
  
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) 
 
from keras.callbacks import EarlyStopping 
 
# 5. 모델 학습시키기 
num_epochs = 2000 
 
# history = LossHistory() # 손실 이력 객체 생성 
# history.init() 
 
early_stopping=EarlyStopping(monitor='acc', patience=100, verbose=2, mode='max') 
for epoch_idx in range(num_epochs): 
    print('epochs : ' + str(epoch_idx)) 
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False, 
              validation_data=(x_test, y_test), callbacks=[early_stopping]) 
    model.reset_states() 

loss, acc = model.evaluate(x_train, y_train, batch_size=1) 
print("loss : ", loss) 
print("acc : ", acc) 
model.reset_states() 
# x_test = np.array([[[21],[22],[23],[24]], [[22],[23],[24],[25]], [[23],[24],[25],[26]], [[24],[25],[26],[27]]]) 
 
y_predict = model.predict(x_train, batch_size=1) 
model.reset_states() 
y_predict2 = model.predict(x_test, batch_size=1) 
model.reset_states() 

print(y_predict) 
print("=======================") 
print(y_predict2) 
  

 

 


