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

from analysis.methods import xgb_skf_gridsearch
from utils.global_variables import PATH
from utils.data_processing import (
    pca_on_patients,
    patient_info_by_frequency
)
from utils.visualization import complete_classification_report
from utils.data_loading import load_data, dict_to_df
# %%
# Chargement des données
data_dict = load_data(PATH)
df_data = dict_to_df(data_dict)

#%% ACP
frequencies = [
    "ALPHA",
    "BETA",
    # "DELTA",
    "THETA"
    ]
df_freq = patient_info_by_frequency(data_dict, frequencies= frequencies)
# %%
acp_patient = pca_on_patients(df_freq, cum_var_threshold=0.95)



#equilibrer classes
import pandas as pd
ad_sci_df = acp_patient[acp_patient["State"].isin(["AD", "SCI"])]

# Sélectionner 28 échantillons pour "MCI"
mci_df = acp_patient[acp_patient["State"] == "MCI"].sample(n=28, random_state=1)

# Concaténer les deux DataFrames
balanced_df = pd.concat([ad_sci_df, mci_df])

# %%
y_pred_svm, accuracy_svm, best_model, corresp_y = xgb_skf_gridsearch(balanced_df, verbose=True, k=3)
y_true = balanced_df["State"].values.tolist()

print("Utilisation d'un XGB sur toutes les données :")
complete_classification_report(y_true, y_pred_svm)

# %%
y_pred_svm = best_model.predict(acp_patient.drop(columns=["State", "Patient"]))
y_true = acp_patient["State"].values.tolist()

# inverser le dictionnaire
corresp_y = {v: k for k, v in corresp_y.items()}

# corresp_y = {i: state for i, state in enumerate(df_data["State"].unique())}
print(f"corresp_y : {corresp_y}")
y_pred_svm = [corresp_y[state] for state in y_pred_svm]

print("Utilisation d'un XGB sur toutes les données :")
complete_classification_report(y_true, y_pred_svm)

# %%
y_pred_svm = best_model.predict(acp_patient.drop(columns=["State", "Patient"]))
y_true = acp_patient["State"].values.tolist()

corresp_y2 = {0: 'AD', 2: 'MCI', 1: 'SCI'}
y_pred_svm = [corresp_y2[state] for state in y_pred_svm]

print("Utilisation d'un SVM sur toutes les données :")
complete_classification_report(y_true, y_pred_svm)
# %%
