import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np
import datetime

num_samples =100
height = 71
width = 71
num_classes = 100

start = datetime.datetime.now()
with tf.device('/gpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3), classes=num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Generate dummy data.
    x = np.random.random((num_samples, height, width, 3))
    y = np.random.random((num_samples, num_classes))

    model.fit(x, y, epochs=3, batch_size=16) 

    model.save('my_model.h5')

end = datetime.datetime.now()
time_delta = end - start
print('GPU 처리시간 : ', time_delta)


start = datetime.datetime.now()
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3), classes=num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Generate dummy data.
    x = np.random.random((num_samples, height, width, 3))
    y = np.random.random((num_samples, num_classes))

    model.fit(x, y, epochs=3, batch_size=16) 

    model.save('my_model.h5')

end = datetime.datetime.now()
time_delta = end - start
print('CPU 처리시간 : ', time_delta)

# GPU 와 CPU의 처리시간 비교  # 콜랩을 GPU로 셋팅하고 돌려보자.

'''
내 GPU 1060Ti        : 00:32.72
내 CPU 7700HQ        : 04:48.69
Colab GPU            : 00:59.36
Colab CPU Xeon 2.20G : 02:07.94  
Colab TPU            : 01:27.57     // TPU 테스트는 CPU처럼 해서 나중에 다시 정확한 테스트 요함.
'''
