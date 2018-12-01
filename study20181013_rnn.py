import numpy as np
import pandas as pd
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from keras.utils import np_utils

input_size = 151    # 입력 전체 크기 1~100              # 150, 151 이 안먹히는 버그가 있다. 찾아내랏 -> int를 앞에 너면 된다 ㅋㅋ // 아니네 
window_size = 5     # 윈도우 자르는 크기, 5개씩 잘라서, x=4개, y=1개로 다시 자름
# train_len = int(round((input_size-window_size+1)*0.749))    # Train셋 잘린 크기    // 150일경우 (150-5)
# test_len = int(round((input_size-window_size+1)*0.251))     # Test셋 잘린크기

train_len = round((input_size-window_size+1)*0.749)    # Train셋 잘린 크기    // 150일경우 (150-5)  // 이제 101 이 안된다.
test_len = round((input_size-window_size+1)*0.251)     # Test셋 잘린크기


a = np.array(range(1, input_size +1)) 

# 전체크기는 (input_size - window_size +1)행, (window_size)열 
def split_5(seq, window_size):  # 데이터를 5개씩 자르기용.    # 입력이 5이고 5개씩 자르기
    aaa = []
    for i in range(len(seq)-window_size +1):                 # 열
        subset = seq[i:(i+window_size)]       # 0~5
        aaa.append([item for item in subset])
        # print(aaa)
    print(type(aaa))    
    return np.array(aaa)

dataset = split_5(a, window_size)     # 5씩 잘랏으니 (5, 6)가 된다. // window_size+1 만큼씩 잘라진다.
print("===========================")
print(dataset)
print("dataset.shape : ", dataset.shape) # (195, 5)

from sklearn.model_selection import train_test_split

#입력과 출력을 분리시키기  5개와 1개로
x_data = dataset[:,0:4] 
y_data = dataset[:,4]   

print("x_data.shape : ", x_data.shape)     # 195, 4
# x_data = np.reshape(x_data, (95, 4, 1))
# print("x_data.shape : ", x_data.shape)     # 195, 4, 1

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=66
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# print(x_train)
# print(x_train_scaled)

print(x_test_scaled)
print(x_train.shape)            # (146, 4)   
print(x_train_scaled.shape)     # (146, 4)    

# RNN에서 사용할 수 있도록 reshape 시킨다.
x_train_scaled = np.reshape(x_train_scaled, (train_len, 4, 1))    # int((200 - 5)*0.75), 4, 1 // 146  
x_test_scaled = np.reshape(x_test_scaled, (test_len, 4, 1))

print(x_train_scaled.shape)         # (146, 4, 1)
print(x_test_scaled.shape)          # (49, 4, 1)
print(y_train.shape, y_test.shape)  # (146, ), (49, )

print("스케일 조정 후 특성별 최소값:\n {}".format(x_train_scaled.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n {}".format(x_train_scaled.max(axis=0)))
print("스케일 조정 후 특성별 최소값:\n {}".format(x_test_scaled.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n {}".format(x_test_scaled.max(axis=0)))


# #전처리없이 진행
# x_train2 = x_train
# x_test2 = x_test

# 모델 구성하기
model = Sequential()

model.add(LSTM(50, input_shape=(4,1)))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

tb_hist = keras.callbacks.TensorBoard(log_dir='./graph2', # tensorboard --logdir=./graph2 또는 절대경로
                                      histogram_freq=0, write_graph=True, write_images=True) # 텐서보드 사용
from keras.callbacks import EarlyStopping # 조기종료
early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto')  # 개선이 없는 에포가 50회 이상나오면 조기종료
                                       # val_acc 도 사용됨.
# model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=2)
hist = model.fit(x_train_scaled, y_train, epochs= 150, batch_size=1, validation_data=(x_test_scaled, y_test), 
       verbose=2, callbacks=[tb_hist, early_stopping])

# 평가하기
loss_and_metrics = model.evaluate(x_test_scaled, y_test) #, batch_size=10)
print('loss : ' + str(loss_and_metrics[0]))  # loss, acc
print('accuracy : ' + str(loss_and_metrics[1]))  # loss, acc

# 모델 저장하기
from keras.models import load_model
model.save('./save/study20181013_1to1000.h5')  # 실행 루트에 생성됨 // 경로를 다시 바꿔줄것



y_predict = model.predict(x_test_scaled)
#### y_predict값이 잘나왓는지 확인
y_test = np.reshape(y_test, (test_len, 1))        # (49, ) -> (49, 1)
temp= np.hstack((y_test, y_predict))        # (49, 1)
print(temp)

print("===============================")
# x_test2 = np.array([[[11],[12],[13],[14]], [[101],[102],[103],[104]], [[201],[202],[203],[204]], [[301],[302],[303],[304]]])  # 값이 잘 안나옴
temp_b = np.array(range(1001, 1021))
print("temp_b : \n", temp_b)
temp_b2 = split_5(seq = temp_b, window_size = window_size)
x_test3 = temp_b2[:,0:4] 
print("x_test3 : \n", x_test3)
x_test3 = scaler.transform(x_test3)     # 정규화
x_test3 = np.reshape(x_test3, (16, 4, 1))
# print(x_test3)
# print(x_test3.shape)

y_predict2 = model.predict(x_test3)
print(y_predict2)
# print(x_test3)

# 학습과정 살펴보기
import matplotlib.pyplot as plt

fig_ax = loss_ax = plt.subplot()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc = 'lower left')

plt.show()

model.summary()





# print("===============================")
# print(y_test.shape)
# print(y_predict.shape)


# # RMSE 측정 ( 오차범위 계산 )
# from sklearn.metrics import mean_squared_error
# y_predict = model.predict(x_test2)
# y_test2 = np.reshape(y_test, (24,1))      # 왁구에서 오류가 난다.
# lin_mse = mean_squared_error(y_test2, y_predict)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)


'''
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph2', # tensorboard --logdir=./graph2 또는 절대경로
                                      histogram_freq=0, write_graph=True, write_images=True) # 텐서보드 사용
from keras.callbacks import EarlyStopping # 조기종료
early_stopping = EarlyStopping(patience=200)  # 개선이 없는 에포가 200회 이상나오면 조기종료
hist = model.fit(x_train,y_train,  validation_data=(x_test, y_test), verbose=1, 
                    batch_size = 20, epochs=10, callbacks=[tb_hist, early_stopping])   # tb_hist는 텐서보드 사용
# 원래 epochs는 2000 번 이었으나 3로 바꿈  # 너무 시간이 오래걸려서

# 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10)
print('loss : ' + str(loss_and_metrics[0]))  # loss, acc
print('accuracy : ' + str(loss_and_metrics[1]))  # loss, acc

# 모델 저장하기
from keras.models import load_model
model.save('e:/Study/Python2/save/Yul2_save.h5')  # 실행 루트에 생성됨 // 경로를 다시 바꿔줄것

# 학습과정 살펴보기
import matplotlib.pyplot as plt

fig_ax = loss_ax = plt.subplot()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc = 'lower left')

plt.show()

model.summary()

'''
