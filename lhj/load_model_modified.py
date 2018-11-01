
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import keras

# In[3]:
# In[4]:

x = np.array([1,2,3,4,5])   # (5,)
y = np.array([1,2,3,4,5])   # (5,)

# In[5]:

# In[6]:

x_train = x[:3]  # 1,2,3
y_train = y[:3]
x_test = x[3:]   # 4,5
y_test = y[3:]


from keras.models import load_model
model=load_model('reg_model.h5')

print(model.predict(x_test))