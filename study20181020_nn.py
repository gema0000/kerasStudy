###########################
'''
    입력값이 11~90, 출력이 11*2 에서 90*2 의 범위이다 보니,
    다층퍼셉트론으로 계산할 경우 W 의 값이 0, 0, 0, 1 로 계산되어지는듯

'''
###########################


import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils

# a = np.array([11,12,13,14,15,16,17,18,19,20])
a = np.array(range(11,91))
batch_size =1 
window_size = 5

# 데이터 셋 생성 함수
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
print("===========================")
print(dataset)
print(dataset.shape)    # 76, 5


#입력과 출력을 분리시키기  5개와 1개로

x_train = dataset[:,0:4]
y_train = dataset[:,4]

# x_train = np.reshape(x_train, (len(x_train), window_size-1, batch_size))  # 76,4,1

x_test = x_train * 2
y_test = y_train * 2

print(x_train.shape)    # (76, 4, 1)
print(y_train.shape)    # (76, )

print(x_test.shape)     # (76, 4, 1)
print(y_test.shape)     # (76, )


# 모델 구성하기
model = Sequential()

model.add(Dense(128, input_dim=4, activation='relu'))
# model.add(Dropout(0.2)) 
# model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.2))                     # 드랍아웃을 넣으니 acc가 더 확 줄엇다.
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# model.add(LSTM(128, batch_input_shape=(1, 4, 1), stateful=True))
# # model.add(Dropout(0.2))
# # model.add(Dense(6, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(5, activation='relu'))
# model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

# 5. 모델 학습시키기
num_epochs = 2000

# history = LossHistory() # 손실 이력 객체 생성
# history.init()

# early_stopping=EarlyStopping(monitor='acc', patience=100, verbose=2, mode='max')
model.fit(x_train, y_train, epochs=2000, batch_size=10, verbose=2, validation_data=(x_test, y_test))
        #   callbacks=[early_stopping])


# early_stopping=EarlyStopping(monitor='acc', patience=100, verbose=2, mode='max')
# for epoch_idx in range(num_epochs):
#     print('epochs : ' + str(epoch_idx))
#     model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False,
#               validation_data=(x_test, y_test), callbacks=[early_stopping])
#     model.reset_states()


# import matplotlib.pyplot as plt
# plt.plot(history.losses)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()

# early_stopping=EarlyStopping(monitor='acc', patience=50, verbose=2, mode='max')
# # early_stopping=EarlyStopping(monitor='loss', patience=20)
# hist = model.fit(x_train, y_train, epochs=2000, batch_size=1, verbose=2, validation_data=(x_test, y_test),
#                  callbacks=[early_stopping])                             



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


'''


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt

SVG(model_to_dot(model, show_shapes=True, show_layer_names=False).create(prog='dot', format='svg'))



# temp0003 = 0   # 적중률 카운트
# for i in range(len_accuracy_data-1) :    # len(accuracy_data) 는 135개 캐스팅 까먹어서 일단 숫자로 씀
#                         # 135개로 하면 out of size가 걸리므로 134개만 함.
#     temp0004 = accuracy_data[i+1] - accuracy_data[i]
#     #print(accuracy_data[i+1] - accuracy_data[i])
#     #print(temp0004[0], temp0004[1])   # 윗줄이랑 같은 의미 정상작동하는지 확인용
  
#     if temp0004[0] >0 and temp0004[1] >0 :
#         temp0003 = temp0003 + 1   # 적중 카운트 숫자를 센다.
#     elif temp0004[0] <0 and temp0004[1] <0 : 
#         temp0003 = temp0003 + 1   # 적중 카운트 숫자를 센다
         
#     #result_upDown.append(accuracy_data[i+1] - accuracy_data[i])

# print("==================================")
# print("전체데이터 : ", len_accuracy_data-1 )
# print("적  중  율 : ", temp0003 )
# print("예  측  도 : ", temp0003 / (len_accuracy_data-1)) 

'''
