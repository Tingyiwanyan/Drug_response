import numpy as np
import pandas as pd
import pyreadr
from pubchempy import get_compounds, Compound


gene_expression_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/lung_and_all_processed_data/CCLE/RNAseq.rds"
cell_line_drug_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.rds"
drug_index_match_path = "/project/DPDS/Xiao_lab/shared/lcai/Ling-Tingyi/drug_consistency/drug-CCLE.annot.csv"



gene_expression = pyreadr.read_r(gene_expression_path)[None]
cell_line_drug = pyreadr.read_r(cell_line_drug_path)[None]
drug_index_match = pd.read_csv(drug_index_match_path, encoding='windows-1254')


def get_cell_line_feature(cell_line: str, drug_name: str):
	"""
	generate single cell_line features, including gene expression
	and drug smile molecule sqeuence features.
	
	Parameters:
	-----------
	cell_line: string of cell line name
	drug_name: drug name

	Returns:
	--------
	cell line gene expression together with drug smile sequence
	"""
	try:
		gene_exp = gene_expression.loc[gene_expression['CCLE_ID'] == cell_line].values[0][1:]
		d_cid = drug_index_match.loc[drug_index_match['unique_Compound_Name'] == drug_name]['PubChemID'].values[0]

		comp = Compound.from_cid(d_cid)
		csmile = comp.canonical_smiles
	
		return gene_exp, csmile
	except:
		return None

	










