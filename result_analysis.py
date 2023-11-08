import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score

from utils.process_data import *
from base_line_models import *
from drug_transformer import *
import scipy.stats
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

"""
Perform all result analysis, including statistic results, gene targeting results,
and explainations
"""

gene_expression_filtered = pd.read_csv(gene_expression_filtered_path)
drug_names = ['17-AAG','NVP-AEW541','AZD0530','AZD6244','Erlotinib','Irinotecan',
 'L-685458','lapatinib','LBW242','nilotinib','nutlin-3','Paclitaxel','Panobinostat',
 'PD-0325901','PD-0332991','Crizotinib','PHA-665752','PLX-4720','RAF265','sorafenib',
 'NVP-TAE684','dovitinib','topotecan','vandetanib']

 train_sample_num, test_sample_num = train_test_split(drug_cellline_features_ic50_normalized_df)

gene_expression_filtered.set_index('CCLE_ID',inplace=True)

gene_expression, drug_one_hot_encoding, ic50_list, drug_name_list,drug_smile_length, CCLE_name, drug_smile_list = process_chunck_data_transformer(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	train_sample_num)

gene_expression = tf.reshape(gene_expression,[gene_expression.shape[0],gene_expression.shape[1],1])
drug_one_hot_encoding = tf.reshape(drug_one_hot_encoding,[drug_one_hot_encoding.shape[0],130,56])

gene_expression_test, drug_one_hot_encoding_test, ic50_list_test, drug_name_list_test,drug_smile_length_test, CCLE_name_test, drug_smile_list_test = process_chunck_data_transformer(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	test_sample_num)

gene_expression_test = tf.reshape(gene_expression_test,[gene_expression_test.shape[0],gene_expression_test.shape[1],1])
drug_one_hot_encoding_test = tf.reshape(drug_one_hot_encoding_test,[drug_one_hot_encoding_test.shape[0],130,56])

model = model_load("3_heads_explain.tf/")
model.summary()

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


   drug_lung, gene_lung, drug_lung_length, drug_name_lung, CCLE_name_lung, ic50_lung, drug_smile_lung = return_drug_gene(CCLE_name_test, drug_name_list_test, gene_expression_test, drug_one_hot_encoding_test, drug_smile_length_test, ic50_list_test,drug_smile_list_test)

