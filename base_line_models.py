import numpy as np
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

epochs = 30


def shallow_nn(input_dim:float):
	"""
	Create shallow neural network benchmark for drug IC50 prediction

	Parameters:
	-----------
	input_dim: model input dimension

	Returns:
	--------
	the built model
	"""

	model = Sequential()
	model.add(Dense(500, input_dim=input_dim, activation= "relu",kernel_regularizer=regularizers.L2(1e-4)))
	model.add(Dense(100, activation= "relu",kernel_regularizer=regularizers.L2(1e-4)))
	model.add(Dense(50, activation= "relu",kernel_regularizer=regularizers.L2(1e-4)))
	model.add(Dense(1))

	model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

	return model


def shallow_position_wise_nn():
	"""
	Testing baseline for single position-wise feed forward nn
	"""
	X_input = Input((130, 56))
	Y_input = Input((5842, 1))

	dense_1 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_3 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_5 = tf.keras.layers.Dense(1)


	flattern = tf.keras.layers.Flatten()

	concatenation_layer = concatenation_layer()

	X = dense_1(X_input)
	Y = dense_2(Y_input)

	X = flattern(X)
	Y = flattern(Y)

	Y = concatenation_layer(X,Y)

	Y = dense_3(Y)
	Y = dense_4(Y)
	Y = dense_5(Y)

	model = Model(inputs=(X_input, Y_input), outputs=Y)

	model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

	return model






