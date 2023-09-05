import numpy as np
import pandas as pd
import pyreadr
import random
from pubchempy import get_compounds, Compound



std_threshold = 2.0
zero_threshold = 200

gene_expression_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/lung_and_all_processed_data/CCLE/RNAseq.rds"
cell_line_drug_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.rds"
drug_index_match_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.annot.csv"

feature_frame_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug_cellline_features.csv"

feature_clean_frame_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug_celline_features_clean.csv"

feature_ic50_normalized_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug_celline_ic50_normalized.csv"



gene_expression = pyreadr.read_r(gene_expression_path)[None]
cell_line_drug = pyreadr.read_r(cell_line_drug_path)[None]
drug_index_match = pd.read_csv(drug_index_match_path, encoding='windows-1254')

drug_cellline_features_df = pd.read_csv(feature_frame_path)

drug_cellline_features_clean_df = pd.read_csv(feature_clean_frame_path)

drug_cellline_features_ic50_normalized_df = pd.read_csv(feature_ic50_normalized_path)

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
	gene_names = gene_expression.columns[1:]
	index = 0
	for i in gene_names:
		print(index)
		std = np.nanstd(gene_expression[i])
		std_list.append(std)
		zeros_num = list(gene_expression[i]).count(0)
		zeros_list.append(zeros_num)
		if std < std_threshold or zeros_num > zero_threshold:
			gene_expression = gene_expression.drop([i],axis=1)
			print("im here in condition")
		print(index)
		index+= 1

	return gene_expression

def normalize_min_max(inputs: list)->list:
	"""
	normalize ic50 values through the list

	Parameters:
	-----------
	ic50_inputs: list of raw ic50 input

	Returns:
	--------
	normalized ic50 values
	"""
	max = np.max(inputs)
	min = np.min(inputs)

	normalized_ic50 = [(x-min)/(max-min) for x in inputs]

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

def genereate_data_feature(gene_expressions: list, drug_one_hot_encodings: list, 
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

	gene_expression_list = list(map(convert_to_list, gene_expressions))
	drug_one_hot_encoding_list = list(map(convert_to_list, drug_one_hot_encodings))
	#ic50_list = list(map(process_ic50, ic50s))
	#ic50_list = list(map())

	return gene_expression_list, drug_one_hot_encoding_list, ic50s

def process_chunck_data(drug_cellline_features_clean_df: pd.DataFrame, index_array:list =None):
	"""
	extract from the clean feature dataframe to generate chunk of training 
	or testing data

	Parameters:
	-----------
	drug_cellline_features_clean_df: drug cellline featrure dataframe
	index_array: array of index for selecting

	Returns:
	--------
	np array of training or testing data
	"""
	gene_expression_list = [list(drug_cellline_features_clean_df['gene_expression_data'])[i] for i in index_array]
	drug_one_hot_encoding_list = [list(drug_cellline_features_clean_df['drug_one_hot_encoding'])[i] for i in index_array]
	ic50_list = [list(drug_cellline_features_clean_df['IC50_value'])[i] for i in index_array]
	drug_name_list = [list(drug_cellline_features_clean_df['drug_name'])[i] for i in index_array]

	gene_expression_list, drug_one_hot_encoding_list, ic50_list = \
	genereate_data_feature(gene_expression_list, drug_one_hot_encoding_list, ic50_list)

	#ic50_list = normalize_ic50(ic50_list)
	gene_expression_array = np.array(gene_expression_list)
	drug_one_hot_encoding_array = np.array(drug_one_hot_encoding_list)

	cell_line_drug_feature = np.concatenate((gene_expression_array,drug_one_hot_encoding_array),1)
	cell_line_drug_feature = normalize_min_max_array(cell_line_drug_feature)

	return cell_line_drug_feature, ic50_list, drug_name_list


def train_test_split(drug_cellline_features_clean_df: pd.DataFrame, train_percent:float=0.8):
	"""
	perform training and testing dataset split
	"""
	total_num = len(drug_cellline_features_clean_df)
	num_list = list(np.array(range(total_num)))
	train_num = int(np.floor(total_num*train_percent))
	train_sample_num = random.sample(num_list,train_num)

	[num_list.remove(i) for i in train_sample_num]
	train_sample_num.sort()

	return train_sample_num, num_list


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





	










