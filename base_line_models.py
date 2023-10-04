import numpy as np
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from drug_transformer import *

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
	Abalation study on testing baseline for single position-wise feed forward nn
	"""
	X_input = Input((130, 56))
	Y_input = Input((5842, 1))

	dense_1 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_3 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_5 = tf.keras.layers.Dense(1)


	flattern = tf.keras.layers.Flatten()

	#concatenation_layer = concatenation_layer()

	X = dense_1(X_input)
	Y = dense_2(Y_input)

	X = flattern(X)
	Y = flattern(Y)

	Y = tf.concat([X,Y],axis=1)

	Y = dense_3(Y)
	Y = dense_4(Y)
	Y = dense_5(Y)

	model = Model(inputs=(X_input, Y_input), outputs=Y)

	model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

	return model

def base_drug_transformer():
	"""
	Abalation study on basic configuration transformer
	"""
	X_input = Input((130, 56))
	Y_input = Input((5842, 1))
	enc_valid_lens = Input(())

	masked_softmax_ = masked_softmax()
	dotproductattention1 = dotproductattention(50)

	att_embedding = attention_embedding()
	r_connection = residual_connection()

	dense_1 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_3 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_5 = tf.keras.layers.Dense(1)


	flattern = tf.keras.layers.Flatten()

	#concatenation_layer = concatenation_layer()

	#X = dense_1(X_input)

	score, value = dotproductattention1(X_input,X_input,X_input, enc_valid_lens)
	att_score = masked_softmax_(score, enc_valid_lens)
	att_embedding_ = att_embedding(att_score, value)
	#X = r_connection(value, att_embedding_)

	Y = dense_2(Y_input)

	X = flattern(att_embedding_)
	Y = flattern(Y)

	Y = tf.concat([X,Y],axis=1)

	Y = dense_3(Y)
	Y = dense_4(Y)
	Y = dense_5(Y)

	model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=Y)

	model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

	return model


def att_score_output(input_model):
	"""
	Generate intermediate attention score for examination
	"""
	att_layer = input_model.get_layer('masked_softmax')
	att_output = Model(inputs=input_model.input, outputs = att_layer.output)

	return att_output


def double_shallow_position_wise_nn():
	"""
	Abalation study on testing double head position wise nn
	"""
	X_input = Input((130, 56))
	Y_input = Input((5842, 1))

	dense_1 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_1_1 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_2_2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_3 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_5 = tf.keras.layers.Dense(1)


	flattern = tf.keras.layers.Flatten()

	#concatenation_layer = concatenation_layer()

	X = dense_1(X_input)
	X2 = dense_1_1(X_input)
	Y = dense_2(Y_input)
	Y2 = dense_2_2(Y_input)

	X = flattern(X)
	X2 = flattern(X)
	Y = flattern(Y)
	Y2 = flattern(Y2)

	X = tf.concat([X,X2],axis=1)
	Y = tf.concat([Y,Y2],axis=1)
	Y = tf.concat([X,Y],axis=1)

	Y = dense_3(Y)
	Y = dense_4(Y)
	Y = dense_5(Y)

	model = Model(inputs=(X_input, Y_input), outputs=Y)

	model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

	return model










