# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import layers

# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# define model
model = Sequential()
model.add(layers.Bidirectional(LSTM(50, activation='relu', input_shape=(3, 1)))) #, return_sequences=True))  # (None, 50)
# model.add(LSTM(50, activation='relu', return_sequences=True))  # (None, 50)
# model.add(LSTM(50, activation='relu'))  # (None, 50)
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))                                         # (None, 1)   
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=2)
# demonstrate prediction
x_input = array([141, 151, 161])
x2_input = array([25, 35, 45])
x_input = x_input.reshape((1, 3, 1))
x2_input = x2_input.reshape((1, 3, 1))
yhat = model.predict(x_input, verbose=0)
yhat2 = model.predict(x2_input, verbose=0)
print(yhat)
print(yhat2)

model.summary()