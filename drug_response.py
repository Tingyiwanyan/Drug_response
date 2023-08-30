import numpy as np
from utils.process_data import *
from sklearn import linear_model


#ic50_df = normalize_ic50_drug(drug_cellline_features_clean_df)
#df_cell_line_drug_ic50_normalized = generate_df_normalized_ic50(drug_cellline_features_clean_df, ic50_df)

reg = linear_model.Ridge()
cell_line_drug_feature, ic50_list = process_chunck_data(drug_cellline_features_ic50_normalized_df,0,10000)

cell_line_drug_feature_test, ic50_list_test = process_chunck_data(drug_cellline_features_ic50_normalized_df,10000,10899)

reg.fit(cell_line_drug_feature, ic50_list)

prediction_ic50 = reg.predict(cell_line_drug_feature_test)

lst = [prediction_ic50, ic50_list_test]



