import numpy as np
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

epochs = 30

model = Sequential()
model.add(Dense(518, activation="relu"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="mean_squared_error", optimizer='adam')

