# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 16:27:08 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen

Ce script montre un exemple d'utilisation de quelques fonctions
pour l'étude de cas 2.
"""
import importlib
from sklearn.metrics import confusion_matrix, classification_report

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

from analysis.methods import svm_skf, xgboost_skf
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

# %% 
# Analyse des données
# On commence par afficher les données brutes
plot_scatter_mean_std(data_dict, group_states=False)

# %% Exploration des données
# On récupère les données pour les matrices de corrélation
corr_matrices = get_data_big_corr_matrix(data_dict)

# %%
# Classification avec un SVM sur toutes les données
y_pred_svm, accuracy_svm = svm_skf(df_data, mode="multi", verbose=True)
y_true = df_data["State"].values.tolist()

print("Utilisation d'un SVM sur toutes les données :")
complete_classification_report(y_true, y_pred_svm)

# %% 
# ACP sur toutes les données, pour réduire le nombre de variables équivalentes
# à des électrodes
new_dict = patient_info_by_frequency(data_dict)
df_acp = pca_on_patients(new_dict)

# %%
# Classification avec un SVM sur les données après ACP
y_pred_svm_acp, accuracy_svm_acp = svm_skf(df_acp, mode="multi", verbose=True)
y_true = df_acp["State"].values.tolist()

print("Utilisation d'un SVM après ACP :")
complete_classification_report(y_true, y_pred_svm_acp)

# %%
# Classification avec XG Boost sur les données après ACP
y_pred_xgb_acp, accuracy_xgb_acp = xgboost_skf(df_acp, verbose=True)
y_true = df_acp["State"].values.tolist()

print("Utilisation de XGBoost après ACP :")
complete_classification_report(y_true, y_pred_xgb_acp)
