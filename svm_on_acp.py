# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 16:27:08 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen

Ce script montre un exemple d'utilisation de quelques fonctions
pour l'étude de cas 2.
"""
import importlib

# %%
# On importe tout et on les recharges à chaque fois pour éviter les problèmes de cache
# Il est inutile de relancer le kernel, il suffit de relancer la cellule
import utils.data_loading
import utils.visualization
import utils.data_processing
import analysis.methods
importlib.reload(utils.data_loading)
importlib.reload(utils.visualization)
importlib.reload(utils.data_processing)
importlib.reload(analysis.methods)

from analysis.methods import svm_skf, xgboost_skf, svm_skf_gridsearch
from utils.global_variables import PATH
from utils.data_processing import (
    get_data_big_corr_matrix,
    pca_on_patients,
    patient_info_by_frequency
)
from utils.visualization import plot_scatter_mean_std, complete_classification_report
from utils.data_loading import load_data, dict_to_df
# %%
# Chargement des données
data_dict = load_data(PATH)
df_data = dict_to_df(data_dict)

#%% ACP
frequencies = [
    'ALPHA',
    #'BETA',
    #'DELTA',
    'THETA'
    ]

cum_var_threshold = 0.95

df_freq = patient_info_by_frequency(data_dict, frequencies= frequencies)
# %%
acp_patient = pca_on_patients(df_freq, cum_var_threshold=cum_var_threshold)



#equilibrer classes
import pandas as pd
ad_sci_df = acp_patient[acp_patient['State'].isin(['AD', 'SCI'])]

# Sélectionner 28 échantillons pour 'MCI'
mci_df = acp_patient[acp_patient['State'] == 'MCI'].sample(n=28, random_state=1)

# Concaténer les deux DataFrames
balanced_df = pd.concat([ad_sci_df, mci_df])

# %%

print("\n-----------------------------------------------------------------------")
print("Bande de fréquence utilisée(s): ", *frequencies)
print("Seuil de variance cumulée de l'ACP: ", cum_var_threshold)
print("-----------------------------------------------------------------------")


y_pred_svm, accuracy_svm = svm_skf_gridsearch(balanced_df, verbose=True)
y_true = balanced_df["State"].values.tolist()

print("Utilisation d'un SVM sur toutes les données :")
complete_classification_report(y_true, y_pred_svm, method_title= "SVM-ACP")



y_pred_xgboost, accuracy_xgboost = xgboost_skf(balanced_df, verbose=True)
print("Utilisation d'un XGBoost sur toutes les données :")
complete_classification_report(y_true, y_pred_xgboost,  method_title= "XGB-ACP")


