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
selected_indices, selected_features, selected_features_names = lda_for_var_selection(X, y, variance_threshold=0.10)

# Résultats
print("Indices des variables sélectionnées :", selected_indices)
if selected_features_names is not None:
    print("Noms des variables sélectionnées :", selected_features_names)

df_selected = pd.DataFrame(selected_features, columns=selected_features_names)
df_selected["State"] = df_data["State"].values
df_selected["Patient"] = [1 for index in df_selected.index]

y_pred_svm, accuracy_svm = svm_skf_gridsearch(df_selected, verbose=True)
y_true = df_data["State"].values.tolist()

print("Utilisation d'un SVM sur toutes les données :")
complete_classification_report(y_true, y_pred_svm)

y_pred_xgboost, accuracy_xgboost = xgboost_skf(df_selected, verbose=True)
print("Utilisation d'un XGBoost sur toutes les données :")
complete_classification_report(y_true, y_pred_xgboost)
