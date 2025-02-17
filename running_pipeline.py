from utils.process_data import *
from utils.smile_rel_dist_interpreter import *
from base_line_models import *
from drug_transformer import *
import scipy.stats
from sklearn.mixture import BayesianGaussianMixture
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from collections import Counter
#import keras_nlp
from tensorflow.keras import initializers
import json
tf.keras.utils.set_random_seed(812)
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
from random import seed
from random import sample

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.tree import DecisionTreeRegressor
import selfies as sf
import numpy as np
import Geneformer.geneformer as ge
import gseapy as gp
import sys 


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

def extract_input_data_midi(batch_drug_name, batch_smile_seq, batch_interpret_smile, batch_cell_line_name, batch_drug_response, batch_gene_prior=None):
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
    
    for interpret_smile in batch_interpret_smile:
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
        gene_expression_singlecelline = continuous_gene_df_filter.loc[cell_line_]
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



if __name__ == '__main__':
	"""
	Load model
	"""
	if len(sys.argv) > 1:
	    drug_smile_input = sys.argv[1]
	    print(f"Smile Sequence input is, {drug_smile_input}")
	else:
	    print("No Smile Input")
	    sys.exit(0)

	ensemble_id = pyreadr.read_r('/home/tingyi/Ling-Tingyi/LCCL_input/RNA-CCLE_RNAseq.annot.rds')[None]
	gene_expression = gene_expression.set_index("CCLE_ID")
	kk = pd.read_csv('/home/tingyi/Ling-Tingyi/LCCL_input/sample_info.csv')
	total_pathway = pd.read_excel('GSEA.graph_midi_.xlsx')
	prior_knowledge_drug_gene = pd.read_table('canonical_smiles.Tingyi_gene.pccompound.gene_id.interaction_score.txt',sep="\t",on_bad_lines='skip',header=None)


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

	k = drug_transformer_(gene_embeddings)


	"""
	Load gene identification embeddings
	"""
	gene_filtered_var = filtering_raw_gene_expression(gene_expression)
	drug_ic50_value_whole = []
	drug_name_whole = []
	gene_expression_value_avail = []
	gene_expression_value_whole = []
	cell_line_name_avail = []
	cell_line_name_whole = []
	cell_line_name = cell_line_drug["Cell_line_Name"]
	for i in list(cell_line_name):
	    try:
	        gene_expression_value_avail.append(gene_expression.loc[i])
	        cell_line_name_avail.append(i)
	    except:
	        continue

	mutation = pyreadr.read_r('/home/tingyi/Ling-Tingyi/lung_and_all_processed_data/CCLE/driver_mutations_all.rds')[None]
	mutation.set_index("CCLE_ID", inplace =True)
	mutation_avail_filter = mutation.loc[cell_line_name_avail].replace('',0)
	mutation_avail_filter[mutation_avail_filter != 0] = 1


	"""
	Drug tokenize information
	"""
	
	drug_prior = pd.read_csv('drug.pccompound.TARGET.TARGET_PATHWAY 1.txt',header=None,sep="\t",on_bad_lines='skip')
	k = [i.split(',') for i in list(drug_prior[2])]
	target_genes = []
	for i in k:
	    target_genes += i
	target_genes = [i.strip() for i in target_genes]
	pathway_gene = pd.read_csv("pathway.Tingyi_gene.tsv",sep='\t')
	drug_gene_interaction = pd.read_csv("interactions_laetst.tsv",sep='\t')
	drug_target_df = pd.read_csv("gene.max_interaction_score.anti_neoplastic_or_immunotherapy.known_target_added.txt", sep="\t",header=None)
	gene_important = np.unique(list(pathway_gene['ALDH2'])+list(drug_target_df[0])+
	                           list(gene_filtered_var.columns)+list(mutation_avail_filter.columns)+target_genes)

	pathway_gene = pd.read_csv("pathway.Tingyi_gene.tsv",sep='\t',header=None)
	pathway_names = np.unique(list(pathway_gene[0]))
	pathway_gene.set_index(0, inplace=True)
	target_path_way = pd.read_csv('drug.pathway_with_target_gene.NES_rank.txt',sep='\t',header=None)
	target_drug_name = list(target_path_way[0])
	target_drug_pathway = list(target_path_way[1])
	gene_set = {}
	for i in pathway_names:
	    pathway_gene_set = list(pathway_gene.loc[i][1])
	    gene_set[i] = pathway_gene_set

	#import geneformer as ge
	import pickle
	gene_names = []
	ensemble_ids = []
	ensemble_id = pyreadr.read_r('/home/tingyi/Ling-Tingyi/LCCL_input/RNA-CCLE_RNAseq.annot.rds')[None]
	ensemble_id.set_index('EntrezSymbol',inplace =True)
	for i in gene_important:
	    try:
	        ens = ensemble_id.loc[i]['ENSG_id']
	        gene_names.append(i)
	        ensemble_ids.append(ens)
	    except:
	        continue
	        
	ge.emb_extractor.pu
	ge.tokenizer.TOKEN_DICTIONARY_FILE
	with open(ge.tokenizer.TOKEN_DICTIONARY_FILE, "rb") as f:
	    gene_token_dict = pickle.load(f)
	ensemble_id = pyreadr.read_r('/home/tingyi/Ling-Tingyi/LCCL_input/RNA-CCLE_RNAseq.annot.rds')[None]
	ensemble_id.set_index('ENSG_id',inplace =True)
	gene_name_avail_geneformer = []
	token_avail_geneformer = []
	ensemble_avail_geneformer = []
	for i in ensemble_ids:
	    try:
	        token_id = gene_token_dict[i]
	        token_avail_geneformer.append(token_id)
	        ensemble_avail_geneformer.append(i)
	        gene_name_avail_geneformer.append(ensemble_id.loc[i]['EntrezSymbol'])
	    except:
	        continue

	gene_expression_filter = gene_expression[gene_name_avail_geneformer]
	cell_line_name = cell_line_drug['Cell_line_Name']
	cell_line_drug.set_index("Cell_line_Name", inplace =True) 

	gene_expression_whole_avail = gene_expression.loc[cell_line_name_avail]
	disc_gene_total = []
	continuous_gene_total = []
	for name in cell_line_name_avail:
	    #print(name)
	    max_value = np.max(np.array(gene_expression.loc[name]))
	    #min_value = np.min(np.array(gene_expression.loc[name]))
	    continuous_gene = normalize_min_max(gene_expression_whole_avail.loc[name])
	    continuous_gene_total.append(continuous_gene)
	    bin_value = max_value/5
	    #bin_value = max_value/4
	    #Dis = tf.keras.layers.Discretization(bin_boundaries=[0, bin_value,2*bin_value, 3*bin_value ],epsilon=0.001)
	    Dis = tf.keras.layers.Discretization(bin_boundaries=[bin_value,2*bin_value,3*bin_value],epsilon=0.001)
	    #Dis.adapt(np.array(gene_expression_whole_avail))
	    disc_gene_ = Dis(np.array(gene_expression_whole_avail.loc[name]))
	    disc_gene_total.append(disc_gene_)
	disc_gene_total = tf.stack(disc_gene_total)
	disc_gene_df = pd.DataFrame(disc_gene_total, index=cell_line_name_avail)
	disc_gene_df.columns = list(gene_expression.columns)
	disc_gene_df_filter = disc_gene_df[gene_name_avail_geneformer]

	continuous_gene_total = tf.stack(continuous_gene_total)
	continuous_gene_df = pd.DataFrame(continuous_gene_total, index=cell_line_name_avail)
	continuous_gene_df.columns = list(gene_expression.columns)
	continuous_gene_df_filter = continuous_gene_df[gene_name_avail_geneformer]

	avail_mutation_list = disc_gene_df_filter.columns.intersection(mutation_avail_filter.columns)
	mutation_avail_filter = mutation_avail_filter[avail_mutation_list]
	mutation_whole = np.zeros(np.array(disc_gene_df_filter).shape)
	mutation_whole = pd.DataFrame(mutation_whole, index=cell_line_name_avail)
	mutation_whole.columns = list(disc_gene_df_filter.columns)
	mutation_whole[mutation_avail_filter.columns] = mutation_avail_filter

	prior_drug_information_total = pd.read_csv('prior_drug_gene_target_info.csv')

	interpret_drug_smile_input = generate_interpret_smile(drug_smile_input)[0]

	k = drug_transformer_(gene_embeddings)#, relative_pos_enc_lookup=relative_pos_embedding)
	model_midi = k.model_construction_midi(if_mutation=True)
	model_midi.load_weights('midi_55_epochs_prior_3000_pairs_with_drug_regularizer_softmax_temperature_9_training.h5')
	model_midi.summary()

	df_drug_smile = pd.DataFrame(list(zip(drug_names,CCLE_drug_smiles)),
	                             columns=['drug_name','drug_smiles'])
	df_drug_smile.set_index('drug_name',inplace =True)
	drug_prior = pd.read_csv('drug.pccompound.TARGET.TARGET_PATHWAY 1.txt',header=None,sep="\t",on_bad_lines='skip')
	drug_prior.set_index(0,inplace=True)
	string_lookup = tf.keras.layers.StringLookup(vocabulary=vocabulary_drug)
	layer_one_hot = tf.keras.layers.CategoryEncoding(num_tokens=8, output_mode="one_hot")
	smile_length = 100
	gene_name_lists = []
	top_gene_score_whole = []
	top_gene_index_whole = []
	feature_select_score_drug_whole = []
	top_pathway_rank = []
	ranked_pathway = []
	total_gsea = []
	total_top_gene_rank = []
	feature_select_score_model = att_score_self_enco(model_midi,7)
	feature_select_score_model1 = att_score_self_enco(model_midi,31)
	#feature_select_score_model2 = att_score_self_enco(model_midi,24)
	#feature_select_score_model3 = att_score_self_enco(model_midi,25)
	drug_feature_select_score = []
	df_data = pd.read_csv('train_data_midi.csv')
	df_data.set_index("drug_name", inplace=True)
	

	drug_name = drug_names[21]
	#print(drug_name)
	df_drug_train_data = df_data.loc[drug_name]
	#batch_smile_seq = df_drug_train_data['smile_seq'] 
	batch_smile_seq = [drug_smile_input]
	#batch_interpret_smile = df_drug_train_data['interpret_smile']
	batch_interpret_smile = [interpret_drug_smile_input]
	batch_cell_line_name = [df_drug_train_data['cell_line_name'][0]]
	batch_drug_response = [df_drug_train_data['drug_response'][0]]
	batch_drug_name = [drug_name for i in range(len(batch_smile_seq))]
	drug_atom_one_hot_chunk, drug_rel_position_chunk, edge_type_matrix_chunk,\
	drug_smile_length_chunk, gene_expression_bin_chunk, gene_mutation_bin_chunk, gene_prior_chunk = \
	extract_input_data_midi(batch_drug_name, batch_smile_seq, \
	                        batch_interpret_smile, batch_cell_line_name, batch_drug_response)
	    
	batch_shape = drug_atom_one_hot_chunk.shape[0]
	mask = tf.range(start=0, limit=100, dtype=tf.float32)
	mask = tf.broadcast_to(tf.expand_dims(mask,axis=0),shape=[batch_shape,100])
	mask = tf.reshape(mask, shape=(batch_shape*100))
	mask = mask < tf.cast(tf.repeat(drug_smile_length_chunk,repeats=100),tf.float32)
	mask = tf.where(mask,1,0)
	mask = tf.reshape(mask, shape=(batch_shape,100))
	mask = tf.expand_dims(mask, axis=-1)

	feature_select_score_drug = feature_select_score_model.predict((drug_atom_one_hot_chunk, gene_expression_bin_chunk, \
	                                                                drug_smile_length_chunk, drug_rel_position_chunk, \
	                                                                edge_type_matrix_chunk, gene_mutation_bin_chunk, mask))[1]

	feature_select_score_drug_whole.append(feature_select_score_drug[0])
	#drug_feature_select_score.append(feature_select_score_drug)

	#feature_select_score = drug_feature_select_score[4]

	plt.figure()
	g = sns.heatmap(feature_select_score_drug[0][0:drug_smile_length_chunk[0],0:drug_smile_length_chunk[0]], cmap="Blues")
	score = list(np.array(tf.reduce_mean(tf.reduce_mean(feature_select_score_drug, axis=0)[0:drug_smile_length_chunk[0], \
	                                     0:drug_smile_length_chunk[0]],axis=0)))
	score_att = feature_select_score_drug[0:drug_smile_length_chunk[0],0:drug_smile_length_chunk[0]]
	print(score)
	sns.set(rc={"figure.figsize":(10,10)})
	x_labels = list(batch_interpret_smile[0])
	y_labels = list(batch_interpret_smile[0])
	#atoms_drug.append(x_labels)
	#y_labels.reverse()
	g.set_xticks(range(len(x_labels)), labels=x_labels)
	#g.set_yticks(range(1), labels=[' '])
	g.set_yticks(range(len(y_labels)), labels=y_labels)
	g.tick_params(axis='x', rotation=0)
	#g.set(title =drug_name)
	#g.plot()
	figure = g.get_figure()
	figure.savefig('Output/heatmap.png', dpi=300)

	weight_atoms_indices, highlight_bond, colors, colors_bond = extract_atoms_bonds(feature_select_score_drug[0], batch_smile_seq[0])
	#weight_atoms_indices, highlight_bond, colors, colors_bond = extract_atoms_bonds(feature_select_score_drug_whole[0], batch_smile_seq[0])
	mol = Chem.MolFromSmiles(batch_smile_seq[0])
	mol = mol_with_atom_index(mol)
	d2d = rdMolDraw2D.MolDraw2DCairo(500,300)
	option = d2d.drawOptions()
	option.legendFontSize = 18
	option.bondLineWidth = 1.5
	option.highlightBondWidthMultiplier = 20
	option.updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
	d2d.DrawMolecule(mol,highlightAtoms=weight_atoms_indices,highlightAtomColors=colors)#, highlightBonds=highlight_bond, highlightBondColors=colors_bond)
	#d2d.DrawMolecule(mol,highlightAtoms=weight_atoms_indices, highlightBonds=highlight_bond, highlightBondColors=colors_bond)
	d2d.FinishDrawing()
	d2d.WriteDrawingText("Output/molecule.png")

    