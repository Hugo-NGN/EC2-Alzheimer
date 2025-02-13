import importlib
import utils.data_loading
import utils.visualization
import utils.data_processing
import analysis.methods
importlib.reload(utils.data_loading)
importlib.reload(utils.visualization)
importlib.reload(utils.data_processing)
importlib.reload(analysis.methods)
from utils.data_loading import load_data, dict_to_df
from utils.global_variables import PATH
from utils.data_processing import lda_for_var_selection
import pandas as pd
from utils.visualization import plot_scatter_mean_std, complete_classification_report
from analysis.methods import svm_skf, xgboost_skf, svm_skf_gridsearch

data_dict = load_data(PATH)
df_data = dict_to_df(data_dict)

# Préparation des données
X = df_data.drop(columns=["State", "Patient"])  # Variables explicatives
y = df_data["State"].values  # Classes cibles



print(X.head())

# Sélection des variables avec LDA
selected_indices, selected_features, selected_features_names = lda_for_var_selection(X, y, variance_threshold=0.01)

# Résultats
print("Indices des variables sélectionnées :", selected_indices)
if selected_features_names is not None:
    print("Noms des variables sélectionnées :", selected_features_names)






df_selected = pd.DataFrame(selected_features, columns=selected_features_names)
df_selected["State"] = df_data["State"].values
df_selected["Patient"] = df_data["Patient"]



do_balance = True
if do_balance:
    ad_sci_df = df_selected[df_selected['State'].isin(['AD', 'SCI'])]

    # Sélectionner 28 échantillons pour 'MCI'
    mci_df = df_selected[df_selected['State'] == 'MCI'].sample(n=28, random_state=1)

    # Concaténer les deux DataFrames
    df_selected = pd.concat([ad_sci_df, mci_df])



cols = df_selected.columns.tolist()
cols.insert(0, cols.pop(cols.index('State')))
cols.insert(1, cols.pop(cols.index('Patient')))
df_selected = df_selected[cols]

y_pred_svm, accuracy_svm = svm_skf_gridsearch(df_selected, verbose=True)
y_true = df_selected["State"].values.tolist()

print("Utilisation d'un SVM sur toutes les données :")
complete_classification_report(y_true, y_pred_svm)

y_pred_xgboost, accuracy_xgboost = xgboost_skf(df_selected, verbose=True)
print("Utilisation d'un XGBoost sur toutes les données :")
complete_classification_report(y_true, y_pred_xgboost)
