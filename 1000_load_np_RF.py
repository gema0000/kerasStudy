import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt

# tf.set_random_seed(777)  # for reproducibility

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, BatchNormalization

################# matplotlib 한글 구현 #############################
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
####################################################################

input_size = 200    # 입력 전체 크기 1~1000              # 150, 151 이 안먹히는 버그가 있다. 찾아내랏 -> int를 앞에 너면 된다 ㅋㅋ // 아니네 
window_size = 5     # 윈도우 자르는 크기, 5개씩 잘라서, x=4개, y=1개로 다시 자름

# a = np.array(range(1, input_size +1)) 

# 전체크기는 (input_size - window_size +1)행, (window_size)열 
def split_5(seq, window_size):  # 데이터를 5개씩 자르기용.    # 입력이 5이고 5개씩 자르기
    aaa = []
    for i in range(len(seq)-window_size +1):                 # 열
        subset = seq[i:(i+window_size)]       # 0~5
        aaa.append([item for item in subset])
        # print(aaa)
    print(type(aaa))    
    return np.array(aaa)

# dataset = split_5(a, window_size)     # 5씩 잘랏으니 (5, 6)가 된다. // window_size+1 만큼씩 잘라진다.

dataset = np.load(file='project5_11to91/data/dataset1000.npy')

train_len = round((input_size-window_size+1)*0.749)    # Train셋 잘린 크기    // 200일경우 (200-5+1 = 196) * 0.749 = 147  
test_len = round((input_size-window_size+1)*0.251)     # Test셋 잘린크기

print("===========================")
print(dataset)
print("dataset.shape : ", dataset.shape) # (196, 5)
print(train_len)
print(test_len)



from sklearn.model_selection import train_test_split

#입력과 출력을 분리시키기  5개와 1개로
x_data = dataset[:,0:4] 
y_data = dataset[:,4]   

print("x_data.shape : ", x_data.shape)     # 196, 4



x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, random_state=66
)

# 랜덤포레스트는 전처리를 할 필요가 없다.
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = StandardScaler()
# # scaler = MinMaxScaler()
# scaler.fit(x_train)
x_train_scaled = x_train
x_test_scaled = x_test

# print(x_train)
# print(x_train_scaled)

print(x_test_scaled)
print(x_train_scaled.shape)     # (147, 4) 
print(x_test_scaled.shape)      # (49, 4)   
   

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=30000, random_state=3, n_jobs=-1)   # n_jobs=-1 cpu코어 전체를 사용
                                # n_estimators는 클수록 좋음 // 랜포가 만든 트리저장
model.fit(x_train_scaled, y_train)

print("훈련 세트 정확도 : {:.3f}".format(model.score(x_train_scaled, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(model.score(x_test_scaled, y_test)))


temp_b = np.array(range(101, 121))
print("temp_b : \n", temp_b)
temp_b2 = split_5(seq = temp_b, window_size = window_size)
x_test2 = temp_b2[:,0:4] 
y_test2 = temp_b2[:,4]
print("x_test2 : \n", x_test2)
x_predict = x_test2     # 정규화

y_predict = model.predict(x_predict)
print(y_predict)
print(y_predict.shape)
print(y_test2.shape)


y_test2 = np.reshape(y_test2, (16, 1))
y_predict = np.reshape(y_predict, (16, 1))

temp= np.hstack((y_test2, y_predict))        # (16, 1)
print(temp)

print("특성중요도 : \n{}".format(model.feature_importances_))       # 특성중요도

plt.figure(figsize=(12, 8))
plt.xlabel("특성 중요도")
plt.ylabel("입력변수")
ypos=np.arange(4)       # 4개의 x값
industry=('x1', 'x2', 'x3', 'x4')
plt.barh(ypos, model.feature_importances_, align='center', height=0.5)
plt.yticks(ypos, industry)
plt.show()


# x_test3 = np.reshape(x_test3, (16, 4, 1))
# y_predict = model.predict(x_test_scaled)
# #### y_predict값이 잘나왓는지 확인
# y_test = np.reshape(y_test, (test_len, 1))        # (49, ) -> (49, 1)
# temp= np.hstack((y_test, y_predict))        # (49, 1)
# print(temp)



'''
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




# 모델 구성하기
model = Sequential()

# model.add(LSTM(128, input_shape=(4,1), return_sequences=True, stateful=False, dropout=0.5))
model.add(LSTM(32, batch_input_shape=(1, 4, 1), return_sequences=True, stateful=True)) #, dropout=0.2))
model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(LSTM(100, return_sequences=True, stateful=True)) #, dropout=0.2))
# model.add(BatchNormalization())
model.add(LSTM(32, stateful=True)) #, dropout=0.2))
model.add(Dropout(0.3))
# model.add(BatchNormalization())               # 2번째 batchNormal은 오히려 성능이 안좋다.

# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

tb_hist = keras.callbacks.TensorBoard(log_dir='./graph2', # tensorboard --logdir=./graph2 또는 절대경로
                                      histogram_freq=0, write_graph=True, write_images=True) # 텐서보드 사용
from keras.callbacks import EarlyStopping # 조기종료
early_stopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto')  # 개선이 없는 에포가 50회 이상나오면 조기종료
                                       # val_acc 도 사용됨.

# 상태유지가 아닐경우 fit
# hist = model.fit(x_train_scaled, y_train, epochs= 1000, batch_size=1, validation_data=(x_test_scaled, y_test), 
#        verbose=2, callbacks=[tb_hist, early_stopping])

# 상태유지 fit
for i in range(100):
    print("{}회 돌림".format(i))
    hist = model.fit(x_train_scaled, y_train, epochs=1, batch_size=1, shuffle=False,
                     validation_data=(x_test_scaled, y_test), verbose=2, callbacks=[tb_hist, early_stopping])
    model.reset_states()

# 평가하기
loss_and_metrics = model.evaluate(x_test_scaled, y_test) #, batch_size=10)
print('loss : ' + str(loss_and_metrics[0]))  # loss, acc
print('accuracy : ' + str(loss_and_metrics[1]))  # loss, acc
model.reset_states()

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
