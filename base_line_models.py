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
	Y_input = Input((5843, 1))

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
	Y_input = Input((5843, 1))
	enc_valid_lens = Input(())

	masked_softmax_ = masked_softmax()
	dotproductattention1 = dotproductattention(50)

	dotproductattention_deco = dotproductattention(50)

	att_embedding = attention_embedding()
	r_connection = residual_connection()

	dense_1 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_3 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

	dense_5 = tf.keras.layers.Dense(1)

	kernel_key = tf.keras.layers.Dense(50, activation='sigmoid', 
		kernel_regularizer=regularizers.L2(1e-4))

	kernel_query = tf.keras.layers.Dense(50, activation='sigmoid', 
		kernel_regularizer=regularizers.L2(1e-4))

	pos_encoding = positionalencoding(50,130)

	#kernel_value = tf.keras.layers.Dense(output_dim, activation='relu', 
	#	kernel_regularizer=regularizers.L2(1e-4))


	flattern = tf.keras.layers.Flatten()

	#concatenation_layer = concatenation_layer()

	X = dense_1(X_input)

	X = pos_encoding(X)

	#X_query = kernel_query(X)
	#X_key = kernel_key(X)

	#d = X.shape[-1]

	#scores = tf.matmul(X_query, X_key, transpose_b=True)/tf.math.sqrt(
	#	tf.cast(d, dtype=tf.float32))

	score, value = dotproductattention1(X,X,X, enc_valid_lens)
	att_score = masked_softmax_(score, enc_valid_lens)
	att_embedding_ = att_embedding(att_score, value)
	X = r_connection(value, att_embedding_)

	Y = dense_2(Y_input)

	X = flattern(X)
	Y = flattern(Y)

	Y = tf.concat([X,Y],axis=1)

	Y = dense_3(Y)
	Y = dense_4(Y)
	Y = dense_5(Y)

	model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=Y)

	model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

	return model

def return_drug_gene(CCLE_name_test, drug_name_list_test, gene_expression_test, drug_one_hot_encoding_test, drug_smile_length_test, ic50_list_test, drug_smile_list_test):
    lung_index = []
    for i in range(len(CCLE_name_test)):
        #print(np.array(CCLE_name_test)[i][-4:])
        if np.array(CCLE_name_test)[i][-4:] == "LUNG":
            lung_index.append(i)
    drug_name_lung = [drug_name_list_test[i] for i in lung_index]
    CCLE_name_test_lung = [CCLE_name_test[i] for i in lung_index]
    drug_one_hot_encoding_test_lung = [drug_one_hot_encoding_test[i] for i in lung_index]
    gene_expression_test_lung = [gene_expression_test[i] for i in lung_index]
    drug_smile_length_test_lung = [drug_smile_length_test[i] for i in lung_index]
    ic50_list_test_lung = [ic50_list_test[i] for i in lung_index]
    drug_smile_list_test_lung = [drug_smile_list_test[i] for i in lung_index]
    return np.array(drug_one_hot_encoding_test_lung),np.array(gene_expression_test_lung), np.array(drug_smile_length_test_lung), drug_name_lung, CCLE_name_test_lung, ic50_list_test_lung, drug_smile_list_test_lung


def compute_feature_importance(model, gene_expression, drug_one_hot_encoding):
	"""
	Compute feature importance based on model parameters
	with linear regression parameters

	Parameters:
	-----------
	model: the input Midi-transformer model
	gene_expression: the batch gene expression data
	drug_one_hot_encoding: the drug one hot encoding input

	Returns:
	--------
	the feature importance score
	"""
	linear_regression_parameters = model.layers[-1].get_wrights()[0][:,0]
	linear_regression_weights = tf.expand_dims(linear_regression_weights,axis=0)
	att_layer = model.get_layer('tf.math.l2_normalize')
	att_output = Model(inputs=model.input, outputs = att_layer.output)

	score = np.abs(tf.math.multiply(feature_value,linear_regression_weights))

	return score 

def return_gene_drug_target(index, model, gene_names, drug_lung, gene_lung, drug_lung_length,CCLE_name_lung, drug_name_lung,drug_smile_lung, ic50_lung,top_gene=100):
    """
    return the gene drug targeting cross attention matrix
    """
    feature_select_score_model = att_score_self_enco(model,"feature_selection_layer")
    feature_select_score = feature_select_score_model.predict((drug_lung, gene_lung, np.array(drug_lung_length)))
    
    cross_att_model = att_score_self_enco(model, "decoder_cross_block")
    cross_att_model1 = att_score_self_enco(model, "decoder_cross_block_1")
    cross_att_model2 = att_score_self_enco(model, "decoder_cross_block_2")
    cross_att_score = cross_att_model.predict((drug_lung, gene_lung, np.array(drug_lung_length)))
    cross_att_score1 = cross_att_model1.predict((drug_lung, gene_lung, np.array(drug_lung_length)))
    cross_att_score2 = cross_att_model2.predict((drug_lung, gene_lung, np.array(drug_lung_length)))
    
    top_genes_score, top_genes_index = tf.math.top_k(feature_select_score[index][:,0], k=top_gene)
    drug_scores = np.array([cross_att_score[1][index][i] for i in top_genes_index])
    drug_scores2 = np.array([cross_att_score1[1][index][i] for i in top_genes_index])
    drug_scores3 = np.array([cross_att_score2[1][index][i] for i in top_genes_index])
    top_gene_names = np.array([gene_names[i] for i in top_genes_index])
    
    CCLE_name = CCLE_name_lung[index]
    drug_name = drug_name_lung[index]
    drug_smile = drug_smile_lung[index]
    ic50_value = ic50_lung[index]
    
    return drug_scores, drug_scores2, drug_scores3, top_gene_names, drug_smile, CCLE_name, drug_name, ic50_value, feature_select_score

def return_gene_drug_target_all(model, gene_names, drug_lung, gene_lung, drug_lung_length,CCLE_name_lung, drug_name_lung,drug_smile_lung, ic50_lung,top_gene=300):
	"""
	return the gene drug targeting cross attention matrix
	"""
	#feature_select_score_model = att_score_self_enco(model,"feature_selection_layer")
	#feature_select_score = feature_select_score_model.predict((drug_lung, gene_lung, np.array(drug_lung_length)))

	linear_regression_parameters = model.layers[-1].get_weights()[0][:,0]
	linear_regression_weights = tf.expand_dims(linear_regression_weights,axis=0)

	feature_select_score_model = att_score_self_enco(model,"tf.math.l2_normalize")
	feature_select_score = feature_select_score_model.predict((drug_lung, gene_lung, np.array(drug_lung_length)))

	feature_select_score = np.abs(tf.math.multiply(feature_select_score,linear_regression_weights))

	cross_att_model = att_score_self_enco(model, "decoder_cross_block")
	cross_att_model1 = att_score_self_enco(model, "decoder_cross_block_1")
	cross_att_model2 = att_score_self_enco(model, "decoder_cross_block_2")
	cross_att_score = cross_att_model.predict((drug_lung, gene_lung, np.array(drug_lung_length)))
	cross_att_score1 = cross_att_model1.predict((drug_lung, gene_lung, np.array(drug_lung_length)))
	cross_att_score2 = cross_att_model2.predict((drug_lung, gene_lung, np.array(drug_lung_length)))

	target_drug_name_list = []
	target_top_gene_name_list = []
	target_cell_line_name_list = []
	target_top_gene_score_list = []

	test_length = len(CCLE_name_lung)
	for index in range(test_length):
		print(index)
		top_genes_score, top_genes_index = tf.math.top_k(feature_select_score[index][:,0], k=top_gene)
		drug_scores = np.array([cross_att_score[1][index][i] for i in top_genes_index])
		drug_scores2 = np.array([cross_att_score1[1][index][i] for i in top_genes_index])
		drug_scores3 = np.array([cross_att_score2[1][index][i] for i in top_genes_index])
		top_gene_names = np.array([gene_names[i] for i in top_genes_index])
		target_drug_name_list.append(drug_name_lung[index])
		target_top_gene_name_list.append(top_gene_names)
		target_cell_line_name_list.append(CCLE_name_lung[index])
		target_top_gene_score_list.append(top_genes_score)


	return target_drug_name_list, target_cell_line_name_list, target_top_gene_name_list, target_top_gene_score_list

    

def model_save(input_model, name):
	"""
	save current model, name with a tf at last
	"""
	tf.keras.saving.save_model(input_model,name)

def model_load(name):
	"""
	load model
	"""
	return tf.keras.saving.load_model(name)

def att_score_self_enco(input_model, name):
	"""
	Generate intermediate attention score for examination
	"""
	att_layer = input_model.get_layer(name)
	att_output = Model(inputs=input_model.input, outputs = att_layer.output)

	return att_output

def att_score_self_doce(input_model, index):
	#att_layer = input_model.get_layer(name)
	att_layer = input_model.layers[index]
	att_output = Model(inputs=input_model.input, outputs = att_layer.output)

	return att_output


def double_shallow_position_wise_nn():
	"""
	Abalation study on testing double head position wise nn
	"""
	X_input = Input((130, 56))
	Y_input = Input((5843, 1))

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










