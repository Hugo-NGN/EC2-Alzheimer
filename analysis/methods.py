# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 18:09:09 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen
"""
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.svm import SVC
from typing import Literal
from xgboost import XGBClassifier


def svm_skf(data, mode : Literal["multi","1v1v1"]="multi", verbose=False,
            stratified=True, p_kernel="linear"):
    """
    Applique une classification SVM avec (Stratified)KFold.

    Arguments:
        data (pd.DataFrame): Les données sur lesquelles effectuer la classification.
            Les données sont obtenues par `load_data` puis `dict_to_df`.
        mode (str, optionnel): Le mode de classification à utiliser.
            Peut être "multi" (classification sur les 3 états) ou "1v1v1" (classification binaire).
            Par défaut, "multi".
        verbose (bool, optionnel): Si True, afficher les résultats.
            Par défaut, False.
        stratified (bool, optionnel): Si True, utiliser StratifiedKFold.
            Par défaut, True.
        p_kernel (str, optionnel): Le noyau du SVM.
            Par défaut, "linear".

    Retourne:
        tuple:
            np.array: Les prédictions de la classification.
            float: La moyenne de l'accuracy sur les plis de validation.
    """
    df_data = data.copy()

    corresp_y = {state: i for i, state in enumerate(df_data["State"].unique())}
    df_data["State"] = df_data["State"].map(corresp_y)

    if mode == "multi":
        model = SVC(kernel=p_kernel)
    elif mode == "1v1v1":
        model = SVC(kernel=p_kernel, decision_function_shape="ovo")

    if stratified:
        skf = StratifiedKFold(n_splits=5)
    else:
        skf = KFold(n_splits=5)
    
    accuracies = []
    output = np.zeros(len(df_data))

    for train_index, test_index in skf.split(df_data.iloc[:, 2:], df_data["State"]):
        X_train, X_test = df_data.iloc[train_index,2:], df_data.iloc[test_index,2:]
        y_train, y_test = df_data.iloc[train_index]["State"], df_data.iloc[test_index]["State"]

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)

    output = [list(corresp_y.keys())[list(corresp_y.values()).index(i)] for i in output]

    if verbose:
        print("Précision moyenne avec StratifiedKFold (mode : "
              f"{mode.capitalize()}) : {mean_accuracy:.2f}")
        
    return output, mean_accuracy
    


def xgboost_skf(data: pd.DataFrame, verbose=False, k=5,
                use_stratified_kfold=True):
    """
    Applique une classification XGBoost avec (Stratified)KFold.

    Arguments:
        data (pd.DataFrame): Les données sur lesquelles effectuer la classification.
            Les données sont obtenues par `load_data` puis `dict_to_df`.
        verbose (bool, optionnel): Si True, afficher les résultats.
            Par défaut, False.
        k (int, optionnel): Le nombre de plis pour la validation croisée.
            Par défaut, 5.
        use_stratified_kfold (bool, optionnel): Si True, utiliser StratifiedKFold.
            Par défaut, True.

    Retourne:
        tuple:
            np.array: Les prédictions de la classification.
            float: La moyenne de l'accuracy sur les plis de validation.
    """
    df_data = data.copy()

    # Map labels to integers
    corresp_y = {state: i for i, state in enumerate(df_data["State"].unique())}
    df_data["State"] = df_data["State"].map(corresp_y)
    
    cols = df_data.columns.tolist()
    cols.insert(0, cols.pop(cols.index('State')))
    df_data = df_data[cols]

    model = XGBClassifier()
    if use_stratified_kfold:
        skf = StratifiedKFold(n_splits=k)
    else:
        # Use K-Fold cross-validation
        skf = KFold(n_splits=k)
    accuracies = []
    output = np.zeros(len(df_data))

    for train_index, test_index in skf.split(df_data.iloc[:, 2:], df_data["State"]):
        X_train, X_test = df_data.iloc[train_index, 2:], df_data.iloc[test_index, 2:]
        y_train, y_test = df_data.iloc[train_index]["State"], df_data.iloc[test_index]["State"]

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)

    output = [list(corresp_y.keys())[list(corresp_y.values()).index(i)] for i in output]

    if verbose:
        print(f"Précision moyenne avec StratifiedKFold : {mean_accuracy:.2f}")

    return output, mean_accuracy





def svm_skf_gridsearch(data, verbose=False,stratified=True):
    """
    Applique une classification SVM avec (Stratified)KFold et GridSearchCV pour optimiser les hyperparamètres.

    Arguments:
        data (pd.DataFrame): Les données sur lesquelles effectuer la classification.
            Les données sont obtenues par `load_data` puis `dict_to_df`.
        mode (str, optionnel): Le mode de classification à utiliser.
            Peut être "multi" (classification sur les 3 états) ou "1v1v1" (classification binaire).
            Par défaut, "multi".
        verbose (bool, optionnel): Si True, afficher les résultats.
            Par défaut, False.
        stratified (bool, optionnel): Si True, utiliser StratifiedKFold.
            Par défaut, True.
        p_kernel (str, optionnel): Le noyau du SVM.
            Par défaut, "linear".

    Retourne:
        tuple:
            np.array: Les prédictions de la classification.
            float: La moyenne de l'accuracy sur les plis de validation.
    """
    df_data = data.copy()

    corresp_y = {state: i for i, state in enumerate(df_data["State"].unique())}
    df_data["State"] = df_data["State"].map(corresp_y)
    
    cols = df_data.columns.tolist()
    cols.insert(0, cols.pop(cols.index('State')))
    cols.insert(1, cols.pop(cols.index('Patient')))
    df_data = df_data[cols]


    model = SVC()

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ["linear", "rbf", "sigmoid", "poly"],
        'decision_function_shape': ['ovo', 'ovr'],
        'probability': [True, False],
        'class_weight' : ['balanced', None]
    }

    if stratified:
        skf = StratifiedKFold(n_splits=5)
    else:
        skf = KFold(n_splits=5)

    accuracies = []
    output = np.zeros(len(df_data))

    for train_index, test_index in skf.split(df_data.iloc[:, 2:], df_data["State"]):
        X_train, X_test = df_data.iloc[train_index, 2:], df_data.iloc[test_index, 2:]
        y_train, y_test = df_data.iloc[train_index]["State"], df_data.iloc[test_index]["State"]

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)

    output = [list(corresp_y.keys())[list(corresp_y.values()).index(i)] for i in output]
    
    print(grid_search.best_params_)
    if verbose:
        print(f"Précision moyenne avec StratifiedKFold  : {mean_accuracy:.2f}")

    return output, mean_accuracy