# %%
import os

from Etude_de_cas_2_script import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = './EpEn Data_sans diag_norm_90 sujets/EpEn Data_sans diag_norm_90 sujets'

greek = ['ALPHA', 'BETA', 'DELTA', 'THETA'] #bp-frequence  

state = {'AD':28, 'MCI':40, 'SCI':22} #etat du trouble (keys) : nombre de patient (values)

NB_ELEC = 30 #nombre d'électrode (constante)

use_total = False


data_dict_ = load_data(path, greek, state)

df_dict = dict_to_df(data_dict_)

dict_pca = create_frequency_matrices_with_patient_labels(data_dict_, greek, state)

latent_4freq = proj_acp_4freq(dict_pca)

# %%
output1, mean_accuracy1 = xgboost_analysis(latent_4freq, verbose = True, mode = "pca")
if use_total:
    # Total should not be used as it is expected to give bad results and is unsuitable for this dataset
    output2, mean_accuracy2 = xgboost_analysis(df_dict, verbose = True, mode="total")

# %%
# Confusion matrix of output and output2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred1 = output1
if use_total:
    y_pred2 = output2

corresp_y = {state: i for i, state in enumerate(df_dict["State"].unique())}
y_test1 = [corresp_y[state] for state in latent_4freq.index]
if use_total:
    y_test2 = [corresp_y[state] for state in df_dict["State"]]

# %%
cm1 = confusion_matrix(y_test1, y_pred1)
if use_total:
    cm2 = confusion_matrix(y_test2, y_pred2)

if use_total:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(cm1, annot=True, ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion matrix for PCA")
    ax[0].set_xticklabels(df_dict["State"].unique())
    ax[0].set_yticklabels(df_dict["State"].unique())
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    sns.heatmap(cm2, annot=True, ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion matrix for total")
    ax[1].set_xticklabels(df_dict["State"].unique())
    ax[1].set_yticklabels(df_dict["State"].unique())
    ax[1].set_xlabel("Predicted")
else:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(cm1, annot=True, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix for PCA")
    ax.set_xticklabels(df_dict["State"].unique())
    ax.set_yticklabels(df_dict["State"].unique())
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.show()

# %%
# Do it again without using stratified kfold
output1_wo_strat, mean_accuracy1_wo_strat = xgboost_analysis(latent_4freq, verbose = True, mode = "pca", use_stratified_kfold=False)
if use_total:
    output2_wo_strat, mean_accuracy2_wo_strat = xgboost_analysis(df_dict, verbose = True, mode="total", use_stratified_kfold=False)

y_pred1_wo_strat = output1_wo_strat
if use_total:
    y_pred2_wo_strat = output2_wo_strat
y_test1_wo_strat = [corresp_y[state] for state in latent_4freq.index]
if use_total:
    y_test2_wo_strat = [corresp_y[state] for state in df_dict["State"]]

# %%
cm1_wo_strat = confusion_matrix(y_test1_wo_strat, y_pred1_wo_strat)
if use_total:
    cm2_wo_strat = confusion_matrix(y_test2_wo_strat, y_pred2_wo_strat)

if use_total:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(cm1_wo_strat, annot=True, ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion matrix for PCA")
    ax[0].set_xticklabels(df_dict["State"].unique())
    ax[0].set_yticklabels(df_dict["State"].unique())
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    sns.heatmap(cm2_wo_strat, annot=True, ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion matrix for total")
    ax[1].set_xticklabels(df_dict["State"].unique())
    ax[1].set_yticklabels(df_dict["State"].unique())
    ax[1].set_xlabel("Predicted")
else:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(cm1_wo_strat, annot=True, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix for PCA")
    ax.set_xticklabels(df_dict["State"].unique())
    ax.set_yticklabels(df_dict["State"].unique())
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.show()
