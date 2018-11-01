
# coding: utf-8

# In[1]:


from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM


# In[2]:


from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K

import numpy as np
import pandas as pd
import keras
import tensorflow as tf


# In[3]:


np.random.seed(777)


# In[4]:


x = np.array([1,2,3,4,5])   # (5,)
y = np.array([1,2,3,4,5])   # (5,)


# In[5]:


print(type(x))
print(x.shape)


# In[6]:


x_train = x[:3]  # 1,2,3
y_train = y[:3]
x_test = x[3:]   # 4,5
y_test = y[3:]


# In[7]:


print(x_train.shape)
print(x_test.shape)


# In[8]:


model = Sequential()
model.add(Dense(7, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))


# In[9]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

#model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print(loss, acc)

print(model.predict(x_test))

history=model.fit(x_train, y_train, epochs=1, batch_size=1, validation_data=(x_test, y_test),verbose=1)


# In[10]:


import matplotlib.pyplot as plt 
import numpy


# In[11]:


y_vloss=history.history['val_loss']
y_acc=history.history['acc']


# In[12]:



x_len=np.arange(len(y_acc))
plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
plt.plot(x_len,y_acc,"o",c="blue",markersize=3)
plt.show()


# In[13]:


from keras.models import load_model
model.save('./save/reg_model.h5')


# In[14]:


import tensorflow as tf
sess = tf.Session()
import scipy.optimize as optimizer


# In[15]:


acc_summ=tf.summary.scalar("acc",acc)
loss_summ=tf.summary.scalar("loss",loss)
#merge all summaries 
summary=tf.summary.merge_all()
writer=tf.summary.FileWriter('C:/Users/hjlee/Anaconda3/Lib\site-packages/tensorflow/contrib/tensorboard/logs',sess.graph) 
#C:\Users\hjlee\Documents\Python Study_1011
#C:\Users\hjlee\Anaconda3\Lib\site-packages\tensorflow\contrib\tensorboard
#writer.add_graph(sess.graph)

#s, _=sess.run([summary,optimizer],feed_dict=feed_dict)
#writer.add_summary(s, global_step=global_step)
#global_step += 1 

