import numpy as np
from utils.process_data import *
from base_line_models import *
from drug_transformer import *
#from sklearn import linear_model


#ic50_df = normalize_ic50_drug(drug_cellline_features_clean_df)
#df_cell_line_drug_ic50_normalized = generate_df_normalized_ic50(drug_cellline_features_clean_df, ic50_df)

reg = linear_model.Ridge()
train_sample_num, test_sample_num = train_test_split(drug_cellline_features_ic50_normalized_df)

gene_expression_filtered.set_index('CCLE_ID',inplace=True)

cell_line_drug_feature, ic50_list, drug_name_list = process_chunck_data(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	train_sample_num)
cell_line_drug_feature_test, ic50_list_test, drug_name_list_test = process_chunck_data(drug_cellline_features_ic50_normalized_df, gene_expression_filtered,
	test_sample_num)

#model = shallow_nn(cell_line_drug_feature.shape[1])

#hitory = model.fit(cell_line_drug_feature, ic50_list, validation_split=0.2, epochs=30)
#cell_line_drug_feature, ic50_list = process_chunck_data(drug_cellline_features_ic50_normalized_df,0,5000)

#cell_line_drug_feature_test, ic50_list_test = process_chunck_data(drug_cellline_features_ic50_normalized_df,10000,10899)

#reg.fit(cell_line_drug_feature, ic50_list)

#prediction_ic50 = reg.predict(cell_line_drug_feature_test)

#lst = [prediction_ic50, ic50_list_test]

#df = pd.DataFrame()




