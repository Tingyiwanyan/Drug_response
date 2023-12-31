import numpy as np
from utils.process_data import *
from base_line_models import *
from drug_transformer import *
import scipy.stats
from sklearn import linear_model


#ic50_df = normalize_ic50_drug(drug_cellline_features_clean_df)
#df_cell_line_drug_ic50_normalized = generate_df_normalized_ic50(drug_cellline_features_clean_df, ic50_df)


reg = linear_model.Ridge()
train_sample_num, test_sample_num = train_test_split(drug_cellline_features_ic50_normalized_df)

gene_expression_filtered.set_index('CCLE_ID',inplace=True)

"""
cell_line_drug_feature, ic50_list, drug_name_list = process_chunck_data(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	train_sample_num)
cell_line_drug_feature_test, ic50_list_test, drug_name_list_test = process_chunck_data(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	test_sample_num)
"""


gene_expression, drug_one_hot_encoding, ic50_list, drug_name_list,drug_smile_length, CCLE_name,drug_smile_list = process_chunck_data_transformer(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	train_sample_num)

gene_expression = tf.reshape(gene_expression,[gene_expression.shape[0],gene_expression.shape[1],1])
drug_one_hot_encoding = tf.reshape(drug_one_hot_encoding,[drug_one_hot_encoding.shape[0],130,56])

gene_expression_test, drug_one_hot_encoding_test, ic50_list_test, drug_name_list_test,drug_smile_length_test, CCLE_name_test, drug_smile_list_test = process_chunck_data_transformer(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	test_sample_num)

gene_expression_test = tf.reshape(gene_expression_test,[gene_expression_test.shape[0],gene_expression_test.shape[1],1])
drug_one_hot_encoding_test = tf.reshape(drug_one_hot_encoding_test,[drug_one_hot_encoding_test.shape[0],130,56])


#train_dataset = tf.data.Dataset.from_tensor_slices(
#            (gene_expression, drug_one_hot_encoding, drug_smile_length, np.array(ic50_list)))



#testing_gene_expression = np.ones((2000, 5842, 1))
#testing_drug_one_hot = np.ones((2000,130,56))
#testing_drug_smile_length = 50*np.ones((2000))
#testing_ic50 = np.ones((2000))
#inital_model = tf.keras.saving.load_model('single_head.tf')

"""
gene_expression_vocab = list(gene_expression_filtered.columns)

string_lookup = tf.keras.layers.StringLookup(vocabulary=gene_expression_vocab)
layer_one_hot = tf.keras.layers.CategoryEncoding(num_tokens=5843, output_mode="one_hot")
input_gene_expression_names = tf.constant(gene_expression_vocab)
input_gene_expression_index = string_lookup(input_gene_expression_names)-1

input_gene_expression_one_hot = layer_one_hot(input_gene_expression_index)

gene_expression_input = tf.broadcast_to(tf.expand_dims(input_gene_expression_one_hot, axis=0),shape=(gene_expression.shape[0],5843,5843))
gene_expression_input_test = tf.broadcast_to(tf.expand_dims(input_gene_expression_one_hot, axis=0),shape=(gene_expression_test.shape[0],5843,5843))
"""
k = drug_transformer_(list(gene_expression_filtered.columns))
model = k.model_construction()
model.summary()
#k.model_compile()

#model = base_drug_transformer()

#att_output_model = att_score_output(model)
#att_output_ = att_output_model.predict((drug_one_hot_encoding_test[0:10], gene_expression_test[0:10], np.array(drug_smile_length_test)[0:10]))
for i in range(20):
	history = model.fit((drug_one_hot_encoding, gene_expression, np.array(drug_smile_length)),np.array(ic50_list),batch_size=32, validation_split=0.2, epochs=5)
	ic50_predict = model.predict((drug_one_hot_encoding_test, gene_expression_test, np.array(drug_smile_length_test)))
	print("%i th correlation is: %f" %((i+1)*5, scipy.stats.pearsonr(np.array(ic50_list_test),ic50_predict[:,0])[0]))


#history = k.model.fit((testing_drug_one_hot, testing_gene_expression, testing_drug_smile_length),testing_ic50,batch_size=5, validation_split=0.2, epochs=1)
#model = shallow_nn(cell_line_drug_feature.shape[1])

#hitory = model.fit(cell_line_drug_feature, np.array(ic50_list), validation_split=0.2, epochs=30)
#cell_line_drug_feature, ic50_list = process_chunck_data(drug_cellline_features_ic50_normalized_df,0,5000)

#cell_line_drug_feature_test, ic50_list_test = process_chunck_data(drug_cellline_features_ic50_normalized_df,10000,10899)

#reg.fit(cell_line_drug_feature, ic50_list)

#prediction_ic50 = reg.predict(cell_line_drug_feature_test)

#lst = [prediction_ic50, ic50_list_test]

#df = pd.DataFrame()




