import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score

df = pd.read_csv('ridge_regression_train.csv')
df_test = pd.read_csv('ridge_regression_test_random.csv')

def get_drug_categories(drug_names: list, test_drug_names: list)->list:
	"""
	return a list of number representing the drug name categories
	"""
	category_list = []
	for i in range(len(test_drug_names)):
		cat_num = np.where(np.array(drug_names)== test_drug_names[i])[0][0]
		category_list.append(cat_num)

	return category_list


drug_names = ['17-AAG',
 'NVP-AEW541',
 'AZD0530',
 'AZD6244',
 'Erlotinib',
 'Irinotecan',
 'L-685458',
 'lapatinib',
 'LBW242',
 'nilotinib',
 'nutlin-3',
 'Paclitaxel',
 'Panobinostat',
 'PD-0325901',
 'PD-0332991',
 'Crizotinib',
 'PHA-665752',
 'PLX-4720',
 'RAF265',
 'sorafenib',
 'NVP-TAE684',
 'dovitinib',
 'topotecan',
 'vandetanib']

df_test.set_index("drug_name", inplace=True)
df_drug_spec = df_test.loc['AZD6244']

ic50_predict_az = list(df_drug_spec['ic50_predict'])

ic50_true_az = list(df_drug_spec['ic50_true'])

az_r2 = r2_score(ic50_true_az,ic50_predict_az)

df_drug_spec = df_test.loc['NVP-AEW541']

ic50_predict_nv = list(df_drug_spec['ic50_predict'])

ic50_true_nv = list(df_drug_spec['ic50_true'])

nv_r2 = r2_score(ic50_true_nv, ic50_predict_nv)

df_drug_spec = df_test.loc['17-AAG']

ic50_predict_17 = list(df_drug_spec['ic50_predict'])

ic50_true_17 = list(df_drug_spec['ic50_true'])

r2_17=r2_score(ic50_true_17, ic50_predict_17)

df_drug_spec = df_test.loc['AZD0530']

ic50_predict_azd = list(df_drug_spec['ic50_predict'])

ic50_true_azd = list(df_drug_spec['ic50_true'])

r2_azd = r2_score(ic50_true_azd, ic50_predict_azd)
#plt.scatter(ic50_predict, ic50_true, s=3)
#plt.title('Testing drug effect prediciton (Ridge_regression)')
#plt.xlabel("ic50_prediction")
#plt.ylabel("ic50_true")

fig, axs = plt.subplots(2,2)
axs[0,0].scatter(ic50_predict_az, ic50_true_az, s=5)
axs[0,0].set_title(' AZD6244, r2_score=%f' %az_r2)
axs[0,1].scatter(ic50_predict_nv, ic50_true_nv, s=5)
axs[0,1].set_title(' NVP-AEW541, r2_score=%f' %nv_r2)
axs[1,0].scatter(ic50_predict_17, ic50_true_17, s=5)
axs[1,0].set_title(' 17-AAG, r2_score=%f' %r2_17)
axs[1,1].scatter(ic50_predict_azd, ic50_true_azd, s=5)
axs[1,1].set_title(' AZD0530, r2_score=%f' %r2_azd)
#plt.show()