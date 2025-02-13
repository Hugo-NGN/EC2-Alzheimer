# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 18:09:09 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen
"""
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.svm import SVC
import time
from typing import Literal
import warnings
import xgboost as xgb
from xgboost import XGBClassifier


def svm_skf(data, mode: Literal["multi", "1v1v1"] = "multi", verbose=False,
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
        X_train, X_test = df_data.iloc[train_index,
                          2:], df_data.iloc[test_index, 2:]
        y_train, y_test = df_data.iloc[train_index]["State"], df_data.iloc[test_index]["State"]

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)

    output = [list(corresp_y.keys())[list(corresp_y.values()).index(i)]
              for i in output]

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
        X_train, X_test = df_data.iloc[train_index,
                          2:], df_data.iloc[test_index, 2:]
        y_train, y_test = df_data.iloc[train_index]["State"], df_data.iloc[test_index]["State"]

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)

    output = [list(corresp_y.keys())[list(corresp_y.values()).index(i)]
              for i in output]

    if verbose:
        print(f"Précision moyenne avec StratifiedKFold : {mean_accuracy:.2f}")

    return output, mean_accuracy


def svm_skf_gridsearch(data, verbose=False, stratified=True, k=5):
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
        k (int, optionnel): Le nombre de plis pour la validation croisée.

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
        'class_weight': ['balanced', None]
    }

    if stratified:
        skf = StratifiedKFold(n_splits=k)
    else:
        skf = KFold(n_splits=k)

    accuracies = []
    output = np.zeros(len(df_data))

    for train_index, test_index in skf.split(df_data.iloc[:, 2:], df_data["State"]):
        X_train, X_test = df_data.iloc[train_index,
                          2:], df_data.iloc[test_index, 2:]
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

    output = [list(corresp_y.keys())[list(corresp_y.values()).index(i)]
              for i in output]

    print(grid_search.best_params_)
    if verbose:
        print(f"Précision moyenne avec StratifiedKFold  : {mean_accuracy:.2f}")

    return output, mean_accuracy


def xgb_skf_gridsearch(data, verbose=False, stratified=True, k=5):
    """
    Applique une classification XGB avec (Stratified)KFold et GridSearchCV pour optimiser les hyperparamètres.

    Arguments:
        data (pd.DataFrame): Les données sur lesquelles effectuer la classification.
            Les données sont obtenues par `load_data` puis `dict_to_df`.
        verbose (bool, optionnel): Si True, afficher les résultats.
            Par défaut, False.
        stratified (bool, optionnel): Si True, utiliser StratifiedKFold.
            Par défaut, True.
        k (int, optionnel): Le nombre de plis pour la validation croisée.

    Retourne:
        tuple:
            np.array: Les prédictions de la classification.
            float: La moyenne de l'accuracy sur les plis de validation.
            XGBClassifier: Le modèle XGB entraîné.
    """
    if verbose:
        time_start = time.perf_counter()
    df_data = data.copy()

    corresp_y = {state: i for i, state in enumerate(df_data["State"].unique())}
    df_data["State"] = df_data["State"].map(corresp_y)

    cols = df_data.columns.tolist()
    cols.insert(0, cols.pop(cols.index('State')))
    cols.insert(1, cols.pop(cols.index('Patient')))
    df_data = df_data[cols]

    model = XGBClassifier()

    param_grid = {
        "booster": ["gbtree", "gblinear", "dart"],  # Default is gbtree
        "gamma": [1, 0.3, 0.01, 0.001],  # Default is 0.3
        "max_depth": [3, 6, 9, 12],  # Default is 6
        "subsample": [0.5, 0.75, 1.0],  # Default is 1
        "sampling_method": ["uniform", "gradient_based"],  # Default is uniform
        "tree_method": ["auto", "exact", "approx", "hist"]  # Default is auto
    }

    if stratified:
        skf = StratifiedKFold(n_splits=k)
    else:
        skf = KFold(n_splits=k)

    accuracies = []
    output = np.zeros(len(df_data))

    # On enlève les warnings car certaines combinaisons d'hyperparamètres
    # en génèrent beaucoup
    warnings.showwarning = lambda *args, **kwargs: None

    for iter, (train_index, test_index) in enumerate(
            skf.split(df_data.iloc[:, 2:], df_data["State"])):
        X_train, X_test = df_data.iloc[train_index,
                          2:], df_data.iloc[test_index, 2:]
        y_train, y_test = df_data.iloc[train_index]["State"], df_data.iloc[test_index]["State"]

        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        if verbose:
            print(f"Meilleurs paramètres pour le pli {iter} : {grid_search.best_params_}")

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    # On réinitialise le gestionnaire de warning
    warnings.showwarning = warnings._showwarning_orig

    mean_accuracy = np.mean(accuracies)

    output = [list(corresp_y.keys())[list(corresp_y.values()).index(i)]
              for i in output]

    if verbose:
        print(f"grid_search.best_params_ : {grid_search.best_params_}")
        if stratified:
            avec = "avec"
        else:
            avec = "sans"
        print(f"Précisions {avec} StratifiedKFold :\n"
              f"- moyenne : {mean_accuracy}\n- toutes : {accuracies}")
        print(f"Temps d'exécution : {time.perf_counter() - time_start:.2f}s")

    return output, mean_accuracy, best_model, corresp_y
