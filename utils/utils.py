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
gene_mutation_dict = np.zeros((2,2))
for i in range(5):
    edge_type_dict[i,i] = 1
edge_type_dict = tf.cast(edge_type_dict,dtype=tf.float32)
for i in range(4):
    gene_expression_bin_dict[i,i] = 1
gene_expression_bin_dict = tf.cast(gene_expression_bin_dict,dtype=tf.float32)
for i in range(2):
    gene_mutation_dict[i,i] = 1
gene_mutation_dict = tf.cast(gene_mutation_dict, dtype=tf.float32)

with open('gene_embedding_important.npy', 'rb') as f:
	gene_embeddings = np.load(f)

vocabulary_drug = ['F', 'S', 'N', 'O', 'I', 'L', 'B', 'C']
vocabulary_gene_mutation = [0, 1]

drug_names = ['17-AAG','NVP-AEW541','AZD0530','AZD6244','Erlotinib','Irinotecan',
 'L-685458','lapatinib','LBW242','nilotinib','nutlin-3','Paclitaxel','Panobinostat',
 'PD-0325901','PD-0332991','Crizotinib','PHA-665752','PLX-4720','RAF265','sorafenib',
 'NVP-TAE684','dovitinib','topotecan','vandetanib']



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
    #filtered_index_list = [] 
    std_threshold = 1
    zero_threshold = 250
    gene_names = gene_expression.columns
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
            #filtered_index_list.append(index)
            #print("im here in condition")
        #print(index)
    index+= 1
    gene_expression = gene_expression.drop(filtered_list,axis=1)
    
    return gene_expression

def smile_cl_converter(smile):
    new_smile = ''
    for i in range(len(smile)):
        if smile[i] == 'C':
            if not i == len(smile) - 1:
                if smile[i+1] == 'l':
                    new_smile += 'L'
                else:
                    new_smile += smile[i]
            else:
                new_smile += smile[i]
        elif smile[i] == 'l' and smile[i-1] == 'C':
            continue

        elif smile[i] == 'B':
            if not i == len(smile) - 1:
                if smile[i+1] == 'r':
                    new_smile += 'B'
                else:
                    return None
                    #new_smile += smile[i]
        elif smile[i] == 'r' and smile[i-1] == 'B':
            continue
            
        else:
            #new_smile.append(smile[i])
            new_smile+=smile[i]
    return new_smile

def generate_interpret_smile(smile):
    new_smile = smile_cl_converter(smile)
    length = len(new_smile)
    rel_distance_whole = []
    interpret_smile_whole = []
    projection_whole = []

    for i in range(length):
        symbol, dist = symbol_converter(new_smile[i])
        if symbol == None:
            continue
        elif symbol not in vocabulary_drug:
            return None
        else:
            rel_distance, interpret_smile, projection = smile_rel_dis_interpreter(new_smile, i)
            #length_seq = len(rel_distance)
            rel_distance_whole.append(rel_distance)
            interpret_smile_whole.append(interpret_smile)
            projection_whole.append(projection)

    rel_distance_whole = np.stack(rel_distance_whole)
    projection_whole = np.stack(projection_whole)
    interpret_smile_whole = np.stack(interpret_smile_whole)

    return interpret_smile_whole

def extract_input_data_midi(batch_drug_name, batch_smile_seq, batch_cell_line_name, batch_drug_response, continuous_gene_exp, batch_gene_prior=None):
	"""
	Return the actual input data for midi model
	"""
	rel_distance_batch = [generate_rel_dist_matrix(x) for x in batch_smile_seq]
	drug_rel_position_chunk = []
	drug_smile_length_chunk = []
	drug_atom_one_hot_chunk = []
	gene_mutation_chunk = []
	edge_type_matrix_chunk = []
	gene_expression_chunk = []
	gene_selection_chunk = []
	for rel_distance_ in rel_distance_batch: 
	    shape = rel_distance_.shape[0]
	    drug_rel_position = tf.cast(tf.gather(P[0], tf.cast(rel_distance_,tf.int32), axis=0), tf.float32)
	    concat_left = tf.cast(tf.zeros((smile_length-shape,shape,60)), tf.float32)
	    concat_right = tf.cast(tf.zeros((smile_length,smile_length-shape,60)), tf.float32)
	    drug_rel_position = tf.concat((drug_rel_position,concat_left),axis=0)
	    drug_rel_position = tf.concat((drug_rel_position,concat_right),axis=1)
	    drug_rel_position_chunk.append(drug_rel_position)
	drug_rel_position_chunk = tf.stack(drug_rel_position_chunk)
    
	for smile_seq_origin in batch_smile_seq:
		interpret_smile = generate_interpret_smile(smile_seq_origin)
		input_drug_atom_names = tf.constant(list(interpret_smile))
		input_drug_atom_index = string_lookup(input_drug_atom_names)-1
		input_drug_atom_one_hot = layer_one_hot(input_drug_atom_index)
		shape_drug_miss = input_drug_atom_one_hot.shape[0]
		concat_right = tf.zeros((smile_length-shape_drug_miss,8))
		input_drug_atom_one_hot = tf.concat((input_drug_atom_one_hot,concat_right),axis=0)
		drug_smile_length_chunk.append(shape_drug_miss)
		drug_atom_one_hot_chunk.append(input_drug_atom_one_hot)
	drug_smile_length_chunk = np.array(drug_smile_length_chunk)
	drug_atom_one_hot_chunk = tf.stack(drug_atom_one_hot_chunk)

	for smile_seq in batch_smile_seq:
	    edge_type_matrix = get_drug_edge_type(smile_seq)
	    shape = edge_type_matrix.shape[0]
	    edge_type_matrix = tf.gather(edge_type_dict,tf.cast(edge_type_matrix,tf.int16),axis=0)
	    #drug_rel_position = tf.cast(tf.gather(P[0], tf.cast(rel_distance_,tf.int32), axis=0), tf.float32)
	    concat_left = tf.zeros((smile_length-shape,shape,5))
	    concat_right = tf.zeros((smile_length,smile_length-shape,5))
	    edge_type_matrix = tf.concat((edge_type_matrix,concat_left),axis=0)
	    edge_type_matrix = tf.concat((edge_type_matrix,concat_right),axis=1)
	    edge_type_matrix_chunk.append(edge_type_matrix)
	edge_type_matrix_chunk = tf.stack(edge_type_matrix_chunk)

	for cell_line_ in batch_cell_line_name:
	    gene_expression_singlecelline = continuous_gene_exp.loc[cell_line_]
	    gene_expression_chunk.append(gene_expression_singlecelline)
	    gene_mutation_singlecelline = mutation_whole.loc[cell_line_]
	    gene_mutation_chunk.append(gene_mutation_singlecelline)

	if not batch_gene_prior == None:
	    gene_prior_chunk = tf.stack(batch_gene_prior)
	else:
	    gene_prior_chunk = 0
	gene_expression_chunk = tf.stack(gene_expression_chunk)
	gene_expression_bin_chunk = tf.gather(gene_expression_bin_dict,tf.cast(gene_expression_chunk,tf.int16),axis=0)
	gene_mutation_chunk = tf.stack(gene_mutation_chunk)
	gene_mutation_bin_chunk = tf.gather(gene_mutation_dict,tf.cast(gene_mutation_chunk,tf.int16),axis=0)

	return drug_atom_one_hot_chunk, drug_rel_position_chunk, edge_type_matrix_chunk, \
	drug_smile_length_chunk, gene_expression_chunk, gene_mutation_bin_chunk, gene_prior_chunk

def extract_atoms_bonds(weight_min_max,smile):
    mol = Chem.MolFromSmiles(smile)
    resolution = (weight_min_max.max()-weight_min_max.min())/40
    resolution_color = 1/40
    highlight_atoms = []
    weight_atoms_indices = list(np.argsort(-weight_min_max.diagonal())[0:5])
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
    """
    for h in weight_atoms_indices:
        value_color = 1-((weight_min_max.diagonal()[h]-weight_min_max.min())/resolution)*resolution_color
        new_value_color = ((value_color - min_value_color)/range_value_color)*(0.7)
        #colors[h] = ( 1, new_value_color, new_value_color)
        colors[h] = ( 1, 0, 0)
    """
    highlight_bond = []
    highlight_bond_atoms = []
    weight_bond = []
    colors_bond = {}
    bond_idx_ = []
    value_color_list_bond = []
    for bond_idx, bond in enumerate(mol.GetBonds()):
        bond_i, bond_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mid_weight = weight_min_max[bond_i,bond_j]#(weight_min_max[bond_i] + weight_min_max[bond_j]) / 2
        weight_bond.append(mid_weight)
        bond_idx_.append(bond_idx)
    highlight_indices = list(np.argsort(-np.array(weight_bond)))[0:7]#[0:5]

    for bond_idx, bond in enumerate(mol.GetBonds()):
        if bond_idx in highlight_indices:
            bond_i, bond_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_i_weight = weight_min_max.diagonal()[bond_i]
            bond_j_weight = weight_min_max.diagonal()[bond_j]
            highlight_bond_atoms.append(bond_i)
            highlight_bond_atoms.append(bond_j)
            """
            if bond_i_weight > bond_j_weight:
                highlight_bond_atoms.append(bond_i)
            else:
                highlight_bond_atoms.append(bond_j)
            """
    weight_atoms_indices = weight_atoms_indices + highlight_bond_atoms

    for h in weight_atoms_indices:
        value_color = 1-((weight_min_max.diagonal()[h]-weight_min_max.min())/resolution)*resolution_color
        new_value_color = ((value_color - min_value_color)/range_value_color)*(0.7)
        #colors[h] = ( 1, new_value_color, new_value_color)
        colors[h] = ( 1, 0, 0)
    
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

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol