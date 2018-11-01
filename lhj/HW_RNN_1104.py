
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import tensorflow as tf 

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils 
from sklearn.model_selection import train_test_split


# In[14]:


'''
4. 101 - 200 까지 10개씩 잘라서 LSTM 모델 만들기
test값 : 155~ 164 를 넣어서 값을 뽑아낸다. predict
'''
'''
#seed 값 설정  --> 필요한지 확인 
seed=0
numpy.random.seed(seed) 
tf.set_random_seed(seed) 
'''

a=np.array(range(101,201))
#batch_size=1
#window_size=10  

#데이터셋 생성 함수
def split_10(seq, window_size):  # 데이터 10개씩 자르기 
    aaa = [] 
    for i in range(len(a)-window_size +1):                 # 열 
        subset = a[i:(i+window_size)]       
        aaa.append([item for item in subset]) 
        # print(aaa) 
    print(type(aaa))     
    return np.array(aaa)

dataset = split_10(a, window_size=10)   # 10 개씩 잘라서,(91,10)된다.  
print("=================================================================") 
print(dataset) 
print(dataset.shape)    

#입력(x), 출력(y) 분리시키기 

x_train= dataset[:,0:9] 
y_train= dataset[:,9] 


x_train = np.reshape(x_train, (91, 9, 1))
#x_train = np.reshape(x_train, (len(x_train), window_size-1, batch_size))  


print(x_train.shape)
print(y_train.shape)

# 모델 구성하기 
model = Sequential() 
#model.add(LSTM(32, batch_input_shape=(1, 9, 1), stateful=True)) 
model.add(LSTM(32, input_shape=(9, 1))) 
#model.add(LSTM(32, input_shape=x_train.shape[:]))
#model.add(Dense(5,activation='relu'))
model.add(Dense(5,activation='relu')) 
model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1) 

loss,acc=model.evaluate(x_train,y_train)


x_test=np.array([155,156,157,158,159,160,161,162,163])
y_predict = model.predict(x_test) 

print(y_predict)

