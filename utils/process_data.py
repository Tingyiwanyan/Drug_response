import numpy as np
import pandas as pd
import pyreadr
import random
import tensorflow as tf
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
import matplotlib.pyplot as plt
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
from rdkit.Chem import MolFromSmiles, rdmolops
from rdkit.Chem import AllChem, AddHs
from utils.smile_rel_dist_interpreter import *
import networkx as nx
import numpy as np
import seaborn as sns
import rdkit.Chem.rdchem as rdrd

from pubchempy import get_compounds, Compound


P = np.zeros((1, 100, 60))
XX = np.arange(100, dtype=np.float32).reshape(
    -1,1)/np.power(1000, np.arange(
        0, 60, 2, dtype=np.float32) / 60)
P[:, :, 0::2] = np.sin(XX)
P[:, :, 1::2] = np.cos(XX)
#P[0][0] = np.zeros((60))
#shape_X = tf.shape(X)
#X = tf.math.l2_normalize(X, axis=-1)
P = tf.cast(tf.math.l2_normalize(P[:, :100,:], axis=-1), 
    dtype=tf.float32)
edge_type_dict = np.zeros((5,5))
gene_expression_bin_dict = np.zeros((4,4))
for i in range(5):
    edge_type_dict[i,i] = 1
edge_type_dict = tf.cast(edge_type_dict,dtype=tf.float32)
for i in range(4):
    gene_expression_bin_dict[i,i] = 1
gene_expression_bin_dict = tf.cast(gene_expression_bin_dict,dtype=tf.float32)

std_threshold = 0.8
zero_threshold = 300

gene_expression_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/lung_and_all_processed_data/CCLE/RNAseq.rds"
cell_line_drug_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.rds"
drug_index_match_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.annot.csv"

feature_frame_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug_cellline_features.csv"

feature_clean_frame_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug_celline_features_clean.csv"

feature_ic50_normalized_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug_celline_ic50_normalized_correction.csv"

gene_expression_filtered_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/gene_expression_filtered.csv"

gene_expression_selected_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/gene_expression_selected.csv"


CCLE_drug_smiles = ['CC1CC(C(C(C=C(C(C(C=CC=C(C(=O)NC2=CC(=O)C(=C(C1)C2=O)NCC=C)C)OC)OC(=O)N)C)C)O)OC',
 'C1CN(C1)CC2CC(C2)N3C=C(C4=C(N=CN=C43)N)C5=CC(=CC=C5)OCC6=CC=CC=C6',
 'CN1CCN(CC1)CCOC2=CC3=C(C(=C2)OC4CCOCC4)C(=NC=N3)NC5=C(C=CC6=C5OCO6)Cl',
 'CN1C=NC2=C1C=C(C(=C2F)NC3=C(C=C(C=C3)Br)Cl)C(=O)NOCCO',
 'COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC',
 'CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)OC(=O)N6CCC(CC6)N7CCCCC7',
 'CC(C)CC(C(=O)NC(CC1=CC=CC=C1)C(=O)N)NC(=O)C(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OC(C)(C)C)O',
 'CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl',
 'CC(C(=O)NC(C1CCCCC1)C(=O)N2CCC3C2CN(CC3)CCC4=CC=CC=C4)NC',
 'CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)NC4=NC=CC(=N4)C5=CN=CC=C5',
 'CC(C)OC1=C(C=CC(=C1)OC)C2=NC(C(N2C(=O)N3CCNC(=O)C3)C4=CC=C(C=C4)Cl)C5=CC=C(C=C5)Cl',
 'CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C',
 'CC1=C(C2=CC=CC=C2N1)CCNCC3=CC=C(C=C3)C=CC(=O)NO',
 'C1=CC(=C(C=C1I)F)NC2=C(C=CC(=C2F)F)C(=O)NOCC(CO)O',
 'CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C',
 'CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=C(N=CC(=C2)C3=CN(N=C3)C4CCNCC4)N',
 'CC1=C(NC(=C1C(=O)N2CCCC2CN3CCCC3)C)C=C4C5=C(C=CC(=C5)S(=O)(=O)CC6=C(C=CC=C6Cl)Cl)NC4=O',
 'CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=C2C=C(C=N3)Cl)F',
 'CN1C2=C(C=C(C=C2)OC3=CC(=NC=C3)C4=NC=C(N4)C(F)(F)F)N=C1NC5=CC=C(C=C5)C(F)(F)F',
 'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',
 'CC(C)S(=O)(=O)C1=CC=CC=C1NC2=NC(=NC=C2Cl)NC3=C(C=C(C=C3)N4CCC(CC4)N5CCN(CC5)C)OC',
 'CN1CCN(CC1)C2=CC3=C(C=C2)N=C(N3)C4=C(C5=C(C=CC=C5F)NC4=O)N',
 'CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=C(C=CC(=C5CN(C)C)O)N=C4C3=C2)O',
 'CN1CCC(CC1)COC2=C(C=C3C(=C2)N=CN=C3NC4=C(C=C(C=C4)Br)F)OC']

vocabulary_drug = ['F', 'S', 'N', 'O', 'I', 'L', 'B', 'C']
vocabulary_gene_mutation = [0, 1]
vocabulary_gene_expression = [1, 2, 3, 4]

drug_names = ['17-AAG','NVP-AEW541','AZD0530','AZD6244','Erlotinib','Irinotecan',
 'L-685458','lapatinib','LBW242','nilotinib','nutlin-3','Paclitaxel','Panobinostat',
 'PD-0325901','PD-0332991','Crizotinib','PHA-665752','PLX-4720','RAF265','sorafenib',
 'NVP-TAE684','dovitinib','topotecan','vandetanib']

gene_expression = pyreadr.read_r(gene_expression_path)[None]
cell_line_drug = pyreadr.read_r(cell_line_drug_path)[None]
drug_index_match = pd.read_csv(drug_index_match_path, encoding='windows-1254')

#drug_cellline_features_df = pd.read_csv(feature_frame_path)

#drug_cellline_features_clean_df = pd.read_csv(feature_clean_frame_path)

#drug_cellline_features_ic50_normalized_df = pd.read_csv(feature_ic50_normalized_path)

#gene_expression_selected = pd.read_csv(gene_expression_selected_path)

#gene_expression_filtered = pd.read_csv(gene_expression_filtered_path)

"""
One hot encoding smile drug molecule sequence, reference:
https://towardsdatascience.com/basic-molecular-representation-for-machine-learning-b6be52e9ff76
"""
SMILES_CHARS = [' ',
                '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']


smi2index = dict( (c,i) for i,c in enumerate( SMILES_CHARS ) )
index2smi = dict( (i,c) for i,c in enumerate( SMILES_CHARS ) )

def smiles_encoder( smiles, maxlen=130 )-> list:
    X = np.zeros( ( maxlen, len( SMILES_CHARS ) ) )
    for i, c in enumerate( smiles ):
        X[i, smi2index[c] ] = 1

    X = np.concatenate(X,0)
    X = list(X)
    return X

def smiles_decoder( X ):
    smi = ''
    X = X.argmax( axis=-1 )
    for i in X:
        smi += index2smi[ i ]
    return smi

def select_row_gene_expression(gene_expression_filtered: pd.DataFrame, CCLE_name: str)->list:
	"""
	return a row of list of gene expression data from filtered dataframe
	"""
	
	return list(gene_expression_filtered.loc[CCLE_name][1:])

def selecting_filtered_gene_expression(gene_expression: pd.DataFrame, CCLE_names:list)->pd.DataFrame:
	"""
	Selecting the gene_expression data from CCLE_names

	Parameters:
	-----------
	gene_expression: gene expression data frame
	CCLE_names: list of CCLE names to be selected

	Returns:
	--------
	selected gene expression dataframe
	"""

	return gene_expression[gene_expression['CCLE_ID'].isin(CCLE_names)]


def filtering_raw_gene_expression(gene_expression: pd.DataFrame)->pd.DataFrame:
	"""
	Compute the variance of each gene expression, and also return 
	the zero amount of gene expression

	Parameters:
	-----------
	gene_expression: cell line gene expression input

	Returns:
	--------
	dataframe with gene expression variance and zero amount
	"""
	std_list = []
	zeros_list = []
	filtered_list = []
	gene_names = gene_expression.columns[1:]
	index = 0
	for i in gene_names:
		#print(index)
		std = np.nanstd(gene_expression[i])
		std_list.append(std)
		zeros_num = list(gene_expression[i]).count(0)
		zeros_list.append(zeros_num)
		if std < std_threshold or zeros_num > zero_threshold:
			#gene_expression = gene_expression.drop([i],axis=1)
			filtered_list.append(i)
			#print("im here in condition")
		#print(index)
		index+= 1
	gene_expression = gene_expression.drop(filtered_list,axis=1)

	return gene_expression

def quantile_normalization(input_array, quantile_num=4):
    """
    Perform quantile normalization
    
    Parameters:
    -----------
    input_array: one dimensional input array
    quntile_num: the quantile numer to divide the input array
    
    Rerutns:
    --------
    quantile normalized array
    """
    output_array = np.zeros(shape=input_array.shape)
    quantile_percent = 1/quantile_num
    index_quantile_groups = []
    avarage_values = []
    quantile_values = []
    for i in range(quantile_num):
        quantile_percent_ = (i+1)*quantile_percent
        if quantile_percent_ > 1:
            quantile_percent_ = 1
        quantile_value = np.quantile(input_array, quantile_percent_)
        quantile_values.append(quantile_value)
        
    for i in range(quantile_num):
        single_quantile_group = []
        single_index = []
        for j in range(input_array.shape[0]):
            if i == 0:
                if input_array[j] <= quantile_values[i]:
                    single_quantile_group.append(input_array[j])
                    single_index.append(j)
            else:
                if input_array[j] > quantile_values[i-1] and input_array[j] <= quantile_values[i]:
                    single_quantile_group.append(input_array[j])
                    single_index.append(j)
        #whole_quantile_groups.append(single_quantile_group)
        mean_ = np.median(single_quantile_group)
        avarage_values.append(mean_)
        index_quantile_groups.append(single_index)
    
    for i in range(quantile_num):
        single_index_ = index_quantile_groups[i]
        replace_value = avarage_values[i]
        for j in single_index_:
            output_array[j] = replace_value 
        
    return output_array#, avarage_values, index_quantile_groups, quantile_values

def normalize_min_max(inputs: list, max=None, min=None)->list:
	"""
	normalize ic50 values through the list

	Parameters:
	-----------
	ic50_inputs: list of raw ic50 input

	Returns:
	--------
	normalized ic50 values
	"""

	#max = np.max(inputs)
	#min = np.min(inputs)

	#q_output = quantile_normalization(np.array(inputs), 20)

	q_output = inputs
	if max == None:
		max = np.max(q_output)
	if min == None:
		min = np.min(q_output)

	#normalized_ic50 = [(x-min)/(max-min) for x in inputs]
	normalized_ic50 = [(x-min)/(max-min) for x in q_output]

	return normalized_ic50

def normalize_min_max_array(inputs: np.array)-> np.array:
	"""
	normalize array data on gene expression for learning

	Parameters:
	-----------
	inputs: array of data

	Returns:
	--------
	min max normalized array of data
	"""
	#q_output = quantile_normalization(inputs)
	inputs_list = list(inputs)
	normalized_list = list(map(normalize_min_max, inputs_list))

	return np.array(normalized_list)


def process_ic50(ic50_input: str)->float:
	"""
	generate float value ic50

	Parameters:
	-----------
	ic50_input: dataframe input 

	Returns:
	--------
	ic50 float values
	"""

	ic50_input = float(ic50_input.split(' ')[3].split('\n')[0])

	return ic50_input


def normalize_ic50_drug(drug_cellline_features_clean_df: pd.DataFrame)-> pd.DataFrame:
	"""
	return the dataframe with drug-based normalized ic50 values
	"""
	drug_names = list(drug_cellline_features_clean_df['drug_name'])
	ic50 = list(drug_cellline_features_clean_df['IC50_value'])
	comparing_drug_names = list(cell_line_drug.columns)[1:]
	drug_ic50_means = []
	drug_ic50_stds = []
	for i in comparing_drug_names:
		index = np.where(np.array(drug_names)==i)[0]
		drug_ic50 = [ic50[x] for x in index]
		drug_ic50 = list(map(process_ic50, drug_ic50))
		print(drug_ic50)
		ic50_mean = np.nanmean(drug_ic50)
		ic50_std = np.nanstd(drug_ic50)
		drug_ic50_means.append(ic50_mean)
		drug_ic50_stds.append(ic50_std)

	ic50_normalized_df = pd.DataFrame(list(zip(comparing_drug_names, drug_ic50_means, drug_ic50_stds
		)),columns=['drug_name','drug_ic50_mean','drug_ic50_std'])

	return ic50_normalized_df

def z_normalization(drug_ic50_df: pd.DataFrame, drug_name: str, ic50_value: float)-> float:
	"""
	z normaliztion to drug ic50 values

	Parameters:
	-----------
	drug_ic50_df: drug to mean and std projection
	drug_name: drug name
	ic50_value: single ic50 drug value

	Returns:
	--------
	normalized drug ic50 value
	"""
	mean = float(drug_ic50_df.loc[drug_name]['drug_ic50_mean'])
	std = float(drug_ic50_df.loc[drug_name]['drug_ic50_std'])

	#print(ic50_value)
	#print(mean)
	#print(std)

	return (ic50_value-mean)/std

def generate_df_origin_ic50(drug_cellline_features_clean_df: pd.DataFrame):
	"""
	generate filtered clean dataframe with drug specific z-normalized values

	Parameters:
	-----------
	drug_cellline_features_clean_df: the pre-filtered clean drug cellline dataframe
	drug_ic50_df: the calculated drug ic50 mean and std dataframe

	Returns:
	-------
	dataframe with filtered data and z-normalized ic50 values
	"""
	cell_line_name_list = []
	drug_name_list = []
	gene_expression_data_list = []
	drug_compound_smile_list = []
	drug_one_hot_encoding_list = []
	IC50_list = []
	
	#drug_ic50_df.set_index("drug_name", inplace =True)
	#drug_cellline_features_clean_df.set_index()
	#for row in drug_cellline_features_df['drug_name']:
	for i in range(len(drug_cellline_features_clean_df)):
		gene_expression = drug_cellline_features_clean_df['gene_expression_data'][i]
		drug_compound = drug_cellline_features_clean_df['drug_compound_smile'][i]
		ic50_value = drug_cellline_features_clean_df['IC50_value'][i]
		ic50_value = process_ic50(ic50_value)
		drug_name = drug_cellline_features_clean_df['drug_name'][i]
		#try: 
		if np.isnan(ic50_value) == True:
			print("im here in is nan")
			print(ic50_value)
			continue
		#else:
		#	ic50_value = z_normalization(drug_ic50_df, drug_name, ic50_value)
		#except:
			#ic50_value = z_normalization(drug_ic50_df, drug_name, ic50_value)
		#try:
		#print(i)
		drug_compound = smiles_encoder(drug_compound)
		cell_line_name_list.append(drug_cellline_features_clean_df['cell_line_name'][i])
		drug_name_list.append(drug_name)
		gene_expression_data_list.append(gene_expression)
		drug_compound_smile_list.append(drug_cellline_features_clean_df['drug_compound_smile'][i])
		drug_one_hot_encoding_list.append(drug_compound)
		IC50_list.append(ic50_value)
			#drug_cellline_features_df['gene_expression_data'][i] = gene_expression
		#except:
			#continue


	#drug_cellline_features_df.loc[:,"drug_one_hot_encoding"] = drug_one_hot_encoding

	df_cell_line_drug_feature_ic50_normalied = pd.DataFrame(list(zip(cell_line_name_list, drug_name_list, gene_expression_data_list,\
		drug_compound_smile_list, drug_one_hot_encoding_list, IC50_list)),columns=['cell_line_name','drug_name','gene_expression_data',\
		'drug_compound_smile','drug_one_hot_encoding','IC50_value'])

	return df_cell_line_drug_feature_ic50_normalied

def generate_df_normalized_ic50(drug_cellline_features_clean_df: pd.DataFrame, drug_ic50_df: pd.DataFrame):
	"""
	generate filtered clean dataframe with drug specific z-normalized values

	Parameters:
	-----------
	drug_cellline_features_clean_df: the pre-filtered clean drug cellline dataframe
	drug_ic50_df: the calculated drug ic50 mean and std dataframe

	Returns:
	-------
	dataframe with filtered data and z-normalized ic50 values
	"""
	cell_line_name_list = []
	drug_name_list = []
	gene_expression_data_list = []
	drug_compound_smile_list = []
	drug_one_hot_encoding_list = []
	IC50_list = []
	
	drug_ic50_df.set_index("drug_name", inplace =True) 
	#drug_cellline_features_clean_df.set_index()
	#for row in drug_cellline_features_df['drug_name']:
	for i in range(len(drug_cellline_features_clean_df)):
		gene_expression = drug_cellline_features_clean_df['gene_expression_data'][i]
		drug_compound = drug_cellline_features_clean_df['drug_compound_smile'][i]
		ic50_value = drug_cellline_features_clean_df['IC50_value'][i]
		ic50_value = process_ic50(ic50_value)
		drug_name = drug_cellline_features_clean_df['drug_name'][i]
		#try: 
		if np.isnan(ic50_value) == True:
			print("im here in is nan")
			print(ic50_value)
			continue
		else:
			ic50_value = z_normalization(drug_ic50_df, drug_name, ic50_value)
		#except:
			#ic50_value = z_normalization(drug_ic50_df, drug_name, ic50_value)
		#try:
		#print(i)
		drug_compound = smiles_encoder(drug_compound)
		cell_line_name_list.append(drug_cellline_features_clean_df['cell_line_name'][i])
		drug_name_list.append(drug_name)
		gene_expression_data_list.append(gene_expression)
		drug_compound_smile_list.append(drug_cellline_features_clean_df['drug_compound_smile'][i])
		drug_one_hot_encoding_list.append(drug_compound)
		IC50_list.append(ic50_value)
			#drug_cellline_features_df['gene_expression_data'][i] = gene_expression
		#except:
			#continue


	#drug_cellline_features_df.loc[:,"drug_one_hot_encoding"] = drug_one_hot_encoding

	df_cell_line_drug_feature_ic50_normalied = pd.DataFrame(list(zip(cell_line_name_list, drug_name_list, gene_expression_data_list,\
		drug_compound_smile_list, drug_one_hot_encoding_list, IC50_list)),columns=['cell_line_name','drug_name','gene_expression_data',\
		'drug_compound_smile','drug_one_hot_encoding','IC50_value'])

	return df_cell_line_drug_feature_ic50_normalied

def generate_data_frame(drug_cellline_features_df: pd.DataFrame):
	"""
	generate data frame for training and testing

	Parameters:
	-----------
	drug_cellline_features_df: raw feature dataframe 

	Returns:
	--------
	filtered feature frame with cleaned data
	"""
	cell_line_name_list = []
	drug_name_list = []
	gene_expression_data_list = []
	drug_compound_smile_list = []
	drug_one_hot_encoding_list = []
	IC50_list = []

	for i in range(len(drug_cellline_features_df)):
		gene_expression = drug_cellline_features_df['gene_expression_data'][i]
		drug_compound = drug_cellline_features_df['drug_compound_smile'][i]

		#try:
		print(i)
		gene_expression = process_gene_expression(gene_expression)
		if gene_expression == []:
			continue
		drug_compound = smiles_encoder(drug_compound)
		cell_line_name_list.append(drug_cellline_features_df['cell_line_name'][i])
		drug_name_list.append(drug_cellline_features_df['drug_name'][i])
		gene_expression_data_list.append(gene_expression)
		drug_compound_smile_list.append(drug_cellline_features_df['drug_compound_smile'][i])
		drug_one_hot_encoding_list.append(drug_compound)
		IC50_list.append(drug_cellline_features_df['IC50_value'][i])
			#drug_cellline_features_df['gene_expression_data'][i] = gene_expression
		#except:
			#continue


	#drug_cellline_features_df.loc[:,"drug_one_hot_encoding"] = drug_one_hot_encoding

	df_cell_line_drug_feature = pd.DataFrame(list(zip(cell_line_name_list, drug_name_list, gene_expression_data_list,\
		drug_compound_smile_list, drug_one_hot_encoding_list, IC50_list)),columns=['cell_line_name','drug_name','gene_expression_data',\
		'drug_compound_smile','drug_one_hot_encoding','IC50_value'])

	return df_cell_line_drug_feature

	#gene_expression_df = list(map(process_gene_expression, gene_expressions))


#def one_hot_encoding_smile(drug_smile: str):


def convert_to_list(text_data: str)-> list:
	"""
	convert text data to list structure

	Parameters:
	-----------
	text_data: str input for list converting

	Returns:
	--------
	converted list
	"""
	text_data = text_data.replace('[','')
	text_data = text_data.replace(']','')
	text_data = text_data.split(",")
	text_data = [float(x) for x in text_data]

	return text_data

def genereate_data_feature(gene_expressions:list, drug_one_hot_encodings: list, 
	ic50s: list):
	"""
	generate model acceptable data features

	Parameters:
	-----------
	gene_expressions: list of gene expression data
	drug_one_hot_encodings: list of drug one hot encodings
	ic50s: list of ic50 values

	Returns:
	--------
	the converted data features for train and test
	"""
	gene_expressions_list = []
	drug_one_hot_encoding_list = []
	ic50_list = []

	#gene_expression_list = list(map(convert_to_list, gene_expressions))
	drug_one_hot_encoding_list = list(map(convert_to_list, drug_one_hot_encodings))
	#ic50_list = list(map(process_ic50, ic50s))
	#ic50_list = list(map())

	return gene_expressions, drug_one_hot_encoding_list, ic50s

def process_chunck_data(drug_cellline_features_clean_df: pd.DataFrame, gene_expression_filtered: pd.DataFrame, index_array:list =None):
	"""
	extract from the clean feature dataframe to generate chunk of training 
	or testing data

	Parameters:
	-----------
	drug_cellline_features_clean_df: drug cellline featrure dataframe
	gene_expression_filtered: filtered gene expression data
	index_array: array of index for selecting

	Returns:
	--------
	np array of training or testing data
	"""
	CCLE_names = [drug_cellline_features_clean_df['cell_line_name'][i] for i in index_array]
	#gene_expression_list = [list(drug_cellline_features_clean_df['gene_expression_data'])[i] for i in index_array]
	gene_expression_list = [gene_expression_filtered.loc[i].values for i in CCLE_names]
	drug_one_hot_encoding_list = [list(drug_cellline_features_clean_df['drug_one_hot_encoding'])[i] for i in index_array]
	ic50_list = [list(drug_cellline_features_clean_df['IC50_value'])[i] for i in index_array]
	drug_name_list = [list(drug_cellline_features_clean_df['drug_name'])[i] for i in index_array]

	gene_expression_list, drug_one_hot_encoding_list, ic50_list = \
	genereate_data_feature(gene_expression_list, drug_one_hot_encoding_list, ic50_list)

	#ic50_list = normalize_ic50(ic50_list)
	gene_expression_array = np.array(gene_expression_list)
	drug_one_hot_encoding_array = np.array(drug_one_hot_encoding_list)
	gene_expression_array = normalize_min_max_array(gene_expression_array)
	cell_line_drug_feature = np.concatenate((gene_expression_array,drug_one_hot_encoding_array),1)
	#cell_line_drug_feature = normalize_min_max_array(cell_line_drug_feature)

	return cell_line_drug_feature, ic50_list, drug_name_list


def get_gene_mutation_input(gene_name_update, mutation, CCLE_name):
    gene_name_decode = np.array([i.decode() for i in gene_name_update])
    mutation_dict = {}
    gene_mutation_input = []
    #index_ = 0
    for drug_name in CCLE_name:
        #print(index_)
        #print(drug_name)
        if drug_name in mutation_dict.keys():
            #print("im here")
            gene_mutation_input.append(mutation_dict[drug_name]['mutation_vector'])
        else:
            mutation_gene_list = []
            mutation_gene_index = []
            mutation_gene_list_data = []
            mutation_vector = np.zeros(len(gene_name_update))
            mutation_vector_ = np.ones(len(gene_name_update))
            mutation_vector = np.stack([mutation_vector, mutation_vector_],axis=1)
            for i in mutation.loc[drug_name].keys():
                if not mutation.loc[drug_name][i] == '':
                    mutation_gene_list.append(i)
            for j in mutation_gene_list:
                try:
                    index = np.where(gene_name_decode == j)[0][0]
                    mutation_gene_index.append(index)
                    mutation_gene_list_data.append(j)
                except:
                    continue
            for i in mutation_gene_index:
                mutation_vector[int(i),0] = 1
                mutation_vector[int(i),1] = 0
            mutation_dict[drug_name] = {}
            mutation_dict[drug_name]['mutation_vector'] = mutation_vector
            mutation_dict[drug_name]['mutation_gene_list'] = mutation_gene_list_data
            mutation_dict[drug_name]['mutation_index'] = mutation_gene_index
            gene_mutation_input.append(mutation_vector)
        #index_ += 1
    #gene_mutation_input = tf.stack(gene_mutation_input)
    return mutation_dict, gene_mutation_input


def process_chunck_data_transformer(drug_cellline_features_clean_df: pd.DataFrame, gene_expression_filtered: pd.DataFrame, index_array:list =None):
	"""
	extract from the clean feature dataframe to generate chunk of training 
	or testing data for transformer input

	Parameters:
	-----------
	drug_cellline_features_clean_df: drug cellline featrure dataframe
	gene_expression_filtered: filtered gene expression data
	index_array: array of index for selecting

	Returns:
	--------
	np array of training or testing data
	corresponding array of valid length
	"""
	drug_smile_length = []
	CCLE_names = [drug_cellline_features_clean_df['cell_line_name'][i] for i in index_array]
	#gene_expression_list = [list(drug_cellline_features_clean_df['gene_expression_data'])[i] for i in index_array]
	gene_expression_list = [gene_expression_filtered.loc[i].values for i in CCLE_names]
	drug_one_hot_encoding_list = [list(drug_cellline_features_clean_df['drug_one_hot_encoding'])[i] for i in index_array]
	drug_smile_list = [list(drug_cellline_features_clean_df['drug_compound_smile'])[i] for i in index_array]
	ic50_list = [list(drug_cellline_features_clean_df['IC50_value'])[i] for i in index_array]
	drug_name_list = [list(drug_cellline_features_clean_df['drug_name'])[i] for i in index_array]

	for i in range(len(drug_smile_list)):
		length = len(drug_smile_list[i])
		drug_smile_length.append(length)

	gene_expression_list, drug_one_hot_encoding_list, ic50_list = \
	genereate_data_feature(gene_expression_list, drug_one_hot_encoding_list, ic50_list)

	#ic50_list = normalize_ic50(ic50_list)
	gene_expression_array = np.array(gene_expression_list)
	drug_one_hot_encoding_array = np.array(drug_one_hot_encoding_list)
	gene_expression_array = normalize_min_max_array(gene_expression_array)
	cell_line_drug_feature = np.concatenate((gene_expression_array,drug_one_hot_encoding_array),1)
	#cell_line_drug_feature = normalize_min_max_array(cell_line_drug_feature)

	return gene_expression_array, drug_one_hot_encoding_array, ic50_list, drug_name_list,drug_smile_length, CCLE_names, drug_smile_list


def train_test_split(drug_cellline_features_clean_df: pd.DataFrame, train_percent:float=0.8):
	"""
	perform training and testing dataset split
	"""
	total_num = len(drug_cellline_features_clean_df)
	num_list = list(np.array(range(total_num)))
	train_num = int(np.floor(total_num*train_percent))
	random.seed(50)
	train_sample_num = random.sample(num_list,train_num)

	[num_list.remove(i) for i in train_sample_num]
	train_sample_num.sort()

	return train_sample_num, num_list

def train_test_split_cell_line(drug_cellline_features_clean_df: pd.DataFrame, train_percent:float=0.8):
	"""
	perform training and testing dataset split
	"""
	train_list = []
	distinct_list = []
	for i in drug_cellline_features_clean_df['cell_line_name']:
	    if not i in distinct_list:
	        distinct_list.append(i)
	total_num_cell_line = len(distinct_list)
	num_list = list(np.array(range(total_num_cell_line)))
	train_num = int(np.floor(total_num_cell_line*train_percent))
	random.seed(50)
	train_sample_num = random.sample(num_list,train_num)

	[num_list.remove(i) for i in train_sample_num]
	train_sample_num.sort()
	train_cell_line = [distinct_list[i] for i in train_sample_num]

	total_num = len(drug_cellline_features_clean_df)
	num_list_whole = list(np.array(range(total_num)))

	k = list(drug_cellline_features_clean_df['cell_line_name'])
	for i in train_cell_line:
		single_train_num = list(np.where(np.array(k)==i)[0])
		train_list += single_train_num

	[num_list.remove(i) for i in train_list]
	train_list.sort()

	return train_list, num_list


def process_gene_expression(gene_expression: str)-> list:
	"""
	Process sting-wise gene expression data

	Parameters:
	-----------
	raw_input: string gene expression data

	Returns:
	--------
	gene expression data in list form
	"""
	gene_expression = gene_expression.replace('[','')
	gene_expression = gene_expression.replace(']','')
	gene_expression = gene_expression.split(",")
	gene_expression = gene_expression[1:]
	gene_expression = [float(x) for x in gene_expression]

	#gene_expression = gene_expression[1:]

	return gene_expression


def get_cell_line_feature(cell_line: str, drug_name: str):
	"""
	Generate single cell_line features, including gene expression
	and drug smile molecule sqeuence features.
	
	Parameters:
	-----------
	cell_line: string of cell line name
	drug_name: drug name

	Returns:
	--------
	cell line gene expression together with drug smile sequence
	"""
	#try:
	gene_exp = gene_expression.loc[gene_expression['CCLE_ID'] == cell_line].values
	d_cid = drug_index_match.loc[drug_index_match['unique_Compound_Name'] == drug_name]['PubChemID'].values[0]
	print(d_cid)

	comp = Compound.from_cid(str(d_cid))
	csmile = comp.canonical_smiles

	return gene_exp, csmile
	#except:
		#return None


def generate_feature_frame(cell_line_drug: pd.DataFrame):
	"""
	Generate the dataframe containing: cell_line_name, drug_name,
	gene_expression_data, drug_compound_smile, and IC50 values

	Parameter:
	----------
	cell_line_drug: dataframe of cell_line& drug IC50 values

	Return:
	-------
	the data frame for training and testing
	"""
	drug_names = cell_line_drug.columns[1:].to_list()
	print(drug_names)
	cell_line_names = cell_line_drug['Cell_line_Name'].to_list()
	print(cell_line_names)
	cell_line_name_list = []
	drug_name_list = []
	gene_expression_data_list = []
	drug_compound_smile_list = []
	IC50_list = []

	for i in range(len(cell_line_names)):
		for j in range(len(drug_names)):		

	#for i in range(3):
		#for j in range(10):
			drug_name = drug_names[j]
			cell_line_name = cell_line_names[i]
			features = get_cell_line_feature(cell_line_name, drug_name)
			#print(features)
			ic50_value = cell_line_drug.loc[cell_line_drug['Cell_line_Name'] == cell_line_name][drug_name]
			print(drug_name)
			print(cell_line_name)
			print(ic50_value)

			#print(features[0])
			cell_line_name_list.append(cell_line_name)
			drug_name_list.append(drug_name)
			gene_expression_data_list.append(features[0])
			drug_compound_smile_list.append(features[1])
			IC50_list.append(ic50_value)
			print("im here")
			print(i)

	df_cell_line_drug = pd.DataFrame(list(zip(cell_line_name_list, drug_name_list, gene_expression_data_list,\
		drug_compound_smile_list, IC50_list)),columns=['cell_line_name','drug_name','gene_expression_data',\
		'drug_compound_smile','IC50_value'])

	for i in range(len(df_cell_line_drug.index)):
		try:
			df_cell_line_drug['gene_expression_data'][i] = list(df_cell_line_drug['gene_expression_data'][i][0])
			print("converting gene expression")
			print(i)
		except:
			pass

	return df_cell_line_drug


def cross_validate_10(drug_cellline_features_clean_df: pd.DataFrame, train_percent:float=0.9):
    """
    perform training and testing dataset split
    """
    distinct_list = []
    for i in drug_cellline_features_clean_df['cell_line_name']:
        if not i in distinct_list:
            distinct_list.append(i)
    total_num_cell_line = len(distinct_list)
    train_num = int(np.floor(total_num_cell_line*train_percent))
    total_num = len(drug_cellline_features_clean_df)
    #if cross_val == True:
    random.seed(50)
    train_pair = []
    test_pair = []
    for kk in range(10):
        train_list = []
        num_list = list(np.array(range(total_num_cell_line)))
        train_sample_num = random.sample(num_list,train_num)

        [num_list.remove(i) for i in train_sample_num]
        train_sample_num.sort()
        train_cell_line = [distinct_list[i] for i in train_sample_num]

        num_list_whole = list(np.array(range(total_num)))

        k = list(drug_cellline_features_clean_df['cell_line_name'])
        #print(train_cell_line)
        for i in train_cell_line:
            single_train_num = list(np.where(np.array(k)==i)[0])
            #print(single_train_num)
            train_list += single_train_num


        [num_list_whole.remove(i) for i in train_list]
        train_list.sort()
        
        train_pair.append(train_list)
        test_pair.append(num_list_whole)

    return train_pair, test_pair

def extract_atoms_bonds(weight_min_max,smile):
    mol = Chem.MolFromSmiles(smile)
    resolution = (weight_min_max.max()-weight_min_max.min())/40
    resolution_color = 1/40
    highlight_atoms = []
    weight_atoms_indices = list(np.argsort(-weight_min_max.diagonal())[0:6])
    weight_atoms_indices = [int(kk) for kk in weight_atoms_indices]
    colors = {}
    value_color_list = []
    for h in weight_atoms_indices:
        value_color = ((weight_min_max.diagonal()[h]-weight_min_max.min())/resolution)*resolution_color
        #colors[h] = ( 1, 1-value_color, 1-value_color)
        value_color_list.append(1-value_color)
    max_value_color = np.array(value_color_list).max()
    min_value_color = np.array(value_color_list).min()
    range_value_color = max_value_color - min_value_color
    
    for h in weight_atoms_indices:
        value_color = 1-((weight_min_max.diagonal()[h]-weight_min_max.min())/resolution)*resolution_color
        new_value_color = ((value_color - min_value_color)/range_value_color)*(0.7)
        colors[h] = ( 1, new_value_color, new_value_color)
    
    highlight_bond = []
    weight_bond = []
    colors_bond = {}
    bond_idx_ = []
    value_color_list_bond = []
    for bond_idx, bond in enumerate(mol.GetBonds()):
        bond_i, bond_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mid_weight = weight_min_max[bond_i,bond_j]#(weight_min_max[bond_i] + weight_min_max[bond_j]) / 2
        weight_bond.append(mid_weight)
        bond_idx_.append(bond_idx)
    highlight_indices = list(np.argsort(-np.array(weight_bond)))[0:10]
    for bond_idx, bond in enumerate(mol.GetBonds()):
        bond_i, bond_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #mid_weight = weight_min_max[bond_i,bond_j]#(weight_min_max[bond_i] + weight_min_max[bond_j]) / 2
        mid_weight = weight_bond[bond_idx]
        #weight_bond.append(mid_weight)
        #if bond_i in weight_atoms_indices:
        if bond_idx in highlight_indices:
            highlight_bond.append(bond_idx)
            value_color_ = ((mid_weight-weight_min_max.min())/resolution)*resolution_color
            #value_color_list_bond.append(1-value_color_)
            colors_bond[bond_idx] = (1, 1-value_color_, 1-value_color_)
    return weight_atoms_indices, highlight_bond, colors, colors_bond


	










