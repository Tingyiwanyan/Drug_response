import numpy as np
import pandas as pd
import pyreadr
from pubchempy import get_compounds, Compound


gene_expression_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/lung_and_all_processed_data/CCLE/RNAseq.rds"
cell_line_drug_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.rds"
drug_index_match_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.annot.csv"

feature_frame_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug_cellline_features.csv"



gene_expression = pyreadr.read_r(gene_expression_path)[None]
cell_line_drug = pyreadr.read_r(cell_line_drug_path)[None]
drug_index_match = pd.read_csv(drug_index_match_path, encoding='windows-1254')

drug_cellline_features_df = pd.read_csv(feature_frame_path)


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

	#for i in range(len(cell_line_names)):
		#for j in range(len(drug_names)):
			

	for i in range(3):
		for j in range(10):
			drug_name = drug_names[j]
			cell_line_name = cell_line_names[i]
			features = get_cell_line_feature(cell_line_name, drug_name)
			#print(features)
			ic50_value = cell_line_drug.loc[cell_line_drug['Cell_line_Name'] == cell_line_name][drug_name]
			print(drug_name)
			print(cell_line_name)
			print(ic50_value)

			print(features[0])
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
		print("converting gene expression")
		print(i)
		try:
			df_cell_line_drug['gene_expression_data'][i] = list(df_cell_line_drug['gene_expression_data'][i][0])
		except:
			pass

	return df_cell_line_drug





	










