
# coding: utf-8

# # 1주차 학습
# 

# In[1]:


# coding: utf-8

#1. 문제 정의하기
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
# from keras.layers import Conv1D, MaxPooling1D
# from keras.datasets import imdb
# from keras.utils import np_utils
# from keras.optimizers import SGD, RMSprop, Adam
# from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras import backend as K
import numpy as np
# import pandas as pd
# import keras
# import tensorflow as tf


# In[2]:


np.random.seed(777)

# 데이터 생성
x = np.array([1,2,3,4,5])   # (5,)
y = np.array([1,2,3,4,5])   # (5,)


# In[3]:



print(type(x))
print(x.shape)


# In[6]:


# 훈련과 검증 분리
x_train = x[:3]  # 1,2,3
y_train = y[:3]
x_test = x[3:] #4,5
y_test = y[3:] 


# In[7]:


print(x_train.shape) #(행, 열)
print(x_test.shape)


# In[8]:


# 모델 구성하기

model = Sequential()

model.add(Dense(7, input_dim=1, activation='relu')) #7개의 아웃풋을 생성
model.add(Dense(3))  #3개의 아웃풋을 생성
model.add(Dense(1, activation='relu'))  # 최종적으로 1개의 아웃풋을 생성


# In[9]:


# 모델 설정하기
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[10]:


model.summary()


# In[18]:


hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_test, y_test))
# epochs : 총 회전 수 
# batch_size : 한 번 돌릴 때 데이터를 몇 개의 사이즈로 나눠서 돌릴 것인가?


# In[13]:


loss, acc = model.evaluate(x_test, y_test, batch_size=1)


# In[14]:


print(loss, acc)


# In[15]:


print(model.predict(x_test))


# # 과제 
# 
# 
# 1. 위 모델에 matplotlib으로 그래프가 나오게 하기.
# 
# 2. tensorboard 적용
# 
# 3. save 와 load 기능 적용  (h5 파일로 세이브되면 총 3개의 파일로 만드셔야됩니다.)

# ### 1. matplotlib 적용하기

# In[19]:


import matplotlib.pyplot as plt


# In[24]:


# Plot training & validation accuracy values
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[26]:


# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[31]:


#. 텐서보드 적용하기
import keras

tb_hist = keras.callbacks.TensorBoard(log_dir='C:/Users/shl2143/PycharmProjects', histogram_freq=0, write_graph=True, write_images=True) # 텐서보드 사용
hist =model.fit(x_train, y_train, epochs=1000, batch_size=1,
                validation_data=(x_test, y_test), callbacks=[tb_hist])  # 콜백은 텐서보드 사용

#7. 모델 평가하기
scores = model.evaluate(x_test, y_test)
print("+++++++++++++++++++++++++++++++++++++")
print(scores)
print("+++++++++++++++++++++++++++++++++++++")
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# print(x_test_data[0])
# print(y_test_data)

# 히스토리 기능 사용하기
print(hist.history['loss'])
print(hist.history['acc'])
print(hist.history['val_loss'])
print(hist.history['val_acc'])

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label ='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc ='lower left')

plt.show()


