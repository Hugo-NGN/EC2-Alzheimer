# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 16:26:38 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.global_variables import FREQUENCIES, STATES


def get_value(data_dict, frequency=FREQUENCIES, states=STATES):
    return data_dict[frequency][states]


def get_upper_matrix(data_freq_dstate):
    return data_freq_dstate[np.triu_indices_from(data_freq_dstate, k=1)]


def get_mean_electrods(data_dict, frequencies=FREQUENCIES, states=STATES):
    mean_results = {}

    for freq in frequencies:
        mean_results[freq] = {}
        for dstate in states.keys():

            data = get_value(data_dict, freq, dstate)
            all_patients_data = []

            for patient in data:
                data_patient = get_upper_matrix(patient)
                all_patients_data.append(data_patient)

            all_patients_data = np.array(all_patients_data)
            mean_per_cell = np.mean(all_patients_data, axis=0)
            mean_results[freq][dstate] = mean_per_cell

    return mean_results


def group_mean_by_electrods(data_dict, frequencies=FREQUENCIES, states=STATES):
    """
    Groupe les matrices de données par patient et par état en faisant des moyennes

    Arguments:
        data_dict (dict): Dictionnaire de données obtenu avec `load_data`.
        frequencies (list): Liste des fréquences.
        states (dict): Dictionnaire des états des patients.

    Retourne:
        dict: Dictionnaire de matrices groupées par patient et par état, de la forme:
            {"état": [matrice_patient1, matrice_patient2, ...]}, où chaque matrice est
            la moyenne des matrices de ce patient pour chaque fréquence (30x4)
    """
    grouped_matrices = dict()
    for dstate in states.keys():
        grouped_matrices[dstate] = []
        data_by_frequency = []

        for freq in frequencies:
            data = get_value(data_dict, freq, dstate)

            data_mean = [np.mean(patient, axis=0) for patient in data]
            data_by_frequency.append(data_mean)

        num_patients = len(data_by_frequency[0])  # Nombre de patients
        for i in range(num_patients):
            patient_matrix = np.stack(
                [data_by_frequency[freq_idx][i] for freq_idx in range(len(frequencies))], axis=1)
            grouped_matrices[dstate].append(patient_matrix)

    return grouped_matrices


def group_triangular_matrices(data_dict, frequencies=FREQUENCIES, states=STATES):
    """
    Regroupe les matrices triangulaires supérieures des données par état et par fréquence.

    Arguments:
        data_dict (dict): Dictionnaire contenant les données des matrices.
        frequencies (list): Liste des fréquences à considérer.
        states (dict): Dictionnaire des états à traiter.

    Retourne:
        dict: Un dictionnaire contenant l'ensemble des données utiles sur chaque patient.
            Le dictionnaire est de la forme:
                {"état": {num_patient: df}}, où df est un pandas.Dataframe, ses indices
                sont le numéro des électrodes (0 à 435) et ses colonnes sont les fréquences.
    """

    grouped_vectors = {}

    for dstate in states.keys():
        # Initialise un dictionnaire pour chaque état
        grouped_vectors[dstate] = {}

        # Récupération des données par état et par fréquence
        data_by_frequency = {}
        for freq in frequencies:
            # Liste de matrices 30x30
            data = get_value(data_dict, freq, dstate)
            triangular_vectors = [
                # Extraire les valeurs triangulaires supérieures
                patient[np.triu_indices_from(patient, k=1)]
                for patient in data
            ]
            data_by_frequency[freq] = triangular_vectors

        # Regrouper les vecteurs pour chaque patient dans un DataFrame
        # Nombre de patients
        num_patients = len(data_by_frequency[frequencies[0]])
        for i in range(num_patients):
            # Construire un DataFrame pour chaque patient avec les fréquences comme colonnes
            patient_data = {
                freq: data_by_frequency[freq][i] for freq in frequencies
            }
            # Crée un DataFrame pour le patient
            patient_df = pd.DataFrame(patient_data)
            # Numéro du patient commence à 1
            grouped_vectors[dstate][i + 1] = patient_df

    return grouped_vectors


def patient_info_by_frequency(data_dict, frequencies=FREQUENCIES, states=STATES):
    """
    Crée un DataFrame pour chaque fréquence, contenant chacun tous les patients.

    Args:
        frequencies (list): Liste des fréquences (ALPHA, BETA, DELTA, THETA).
        states (dict): Dictionnaire des états (AD, MCI, SCI) avec leurs données respectives.

    Returns:
        dict: Un dictionnaire contenant un DataFrame par fréquence.
              - Clé : Nom de la fréquence (ALPHA, BETA, etc.).
              - Valeur : DataFrame avec 435 colonnes (électrodes) et les patients en lignes.

    Crée un DataFrame par fréquence (ALPHA, BETA, ...) avec 435 colonnes pour les vecteurs triangulaires supérieurs,
    et les patients en lignes identifiés par un nom unique (ex: AD_1, MCI_1, SCI_1, etc.).
    """
    dict_frequency_matrices = {}

    for freq in frequencies:
        patient_vectors = []
        patient_labels = []

        for dstate, data_list in states.items():
            data = get_value(data_dict, freq, dstate)
            for i, patient in enumerate(data):
                triangular_vector = patient[np.triu_indices_from(patient, k=1)]
                patient_vectors.append(triangular_vector)
                patient_labels.append(f"{dstate}_{i+1}")

        dict_frequency_matrices[freq] = pd.DataFrame(
            patient_vectors,
            columns=[f"Electrode_{i+1}"
                     for i in range(len(triangular_vector))],
            index=patient_labels
        )

    return dict_frequency_matrices


def get_data_big_corr_matrix(data_dict,
                             frequencies=FREQUENCIES,
                             states=STATES,
                             plot_matrix=True):
    """
    Génère la matrice de corrélation pour l'ensemble des données.

    Arguments:
        data_dict (dict): Dictionnaire contenant les données organisées par
            fréquence et état.
        frequencies (list, optional): Liste des fréquences à utiliser.
            Par défaut, utilise la constante FREQUENCIES.
        states (dict, optional): Dictionnaire des états à utiliser.
            Par défaut, utilise la constante STATES.
        plot_matrix (bool, optional): Indique si les matrices de corrélation
            doivent être affichées. Par défaut, True.

    Retourne:
        dict: Dictionnaire contenant les grandes matrices de corrélation
        pour chaque état.
    """

    data_big_correlation = {}
    for dstate in states.keys():
        data_state = []
        for freq in frequencies:
            for i in range(len(data_dict[freq][dstate])):
                if data_dict[freq][dstate][i].shape != (30, 30):
                    print(f"pb taille matrice {freq}, {dstate}, {i}")
                else:
                    vector = np.tril(data_dict[freq][dstate][i], k=-1)
                vector = np.tril(vector, k=-1)
                indices = np.tril_indices(30, k=-1)
                vector = vector[indices].flatten()
                data_state.append(vector)

        # Concaténer les vecteurs
        concatenated_data = np.vstack(data_state)
        data_big_correlation[dstate] = concatenated_data

    if plot_matrix == True:
        for dstate in data_big_correlation.keys():
            sns.heatmap(np.corrcoef(data_big_correlation[dstate]), cmap='gray')
            plt.title(f"Matrice de corrélation pour {dstate}")
            plt.show()

    return data_big_correlation


def pca_on_patients(data_dict, cum_var_threshold=0.95,
                    verbose=True):
    """
    Réalise une ACP sur les données des électrodes.

    Arguments:
        data_dict (dict): Dictionnaire contenant les données des patients.
            Ce dictionnaire est obtenu avec `patient_info_by_frequency`, il doit
            être de la forme: {freq: pd.DataFrame} où chaque DataFrame a en ligne
            les patients avec l'indice "state_nb-patient", et en colonne les
            électrodes.
        cum_var_threshold (float, optionnel): Seuil de variance cumulée à atteindre.
            Par défaut, 0.95.
        frequencies (list, optionnel): Liste des fréquences à considérer.
            Par défaut, la constante FREQUENCIES.
        states (dict, optionnel): Dictionnaire des états des patients.
            Par défaut, la constante STATES.

    Retourne:
        pd.DataFrame: Les composantes principales des données. Les colonnes sont:
            - "State": L'état du patient.
            - "Patient": Le numéro du patient dans cet état.
            - Les composantes principales.

    Cette méthode réduit le nombre de dimensions correspondant à des électrodes
    pour chaque patient en utilisant une ACP. 
    Cette méthode peut être appliquée même si toutes les fréquences ne sont pas
    présentes dans le dictionnaire.
    """
    frequencies = data_dict.keys()
    pca_df = pd.DataFrame()
    
    
    for freq in frequencies:
        if not isinstance(data_dict[freq], pd.DataFrame):
            raise ValueError(f"Les données pour {freq} doivent être un DataFrame.")
        data = data_dict[freq]
        data_norm = (data - data.mean()) / data.std()
        corr = data_norm.corr()

        pca = PCA(n_components=60)
        pca.fit(corr)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_comp = np.argmax(cum_var > cum_var_threshold) + 1

        if verbose:
            print(f"Pour {freq}, {n_comp} composantes principales expliquent"
                  f" {cum_var[n_comp]:.0%} de la variance.")

        pca = PCA(n_components=n_comp)
        pca_data = pca.fit_transform(data_norm)

        pca_data = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_comp)])
        pca_df = pd.concat([pca_data, pca_df], axis=1)
        pca_df["State"] = [assign_class(index) for index in data.index]
        pca_df["Patient"] = [int(index.split("_")[-1]) for index in data.index]

        # Make "State" and "Patient" the first columns
        cols = pca_df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        pca_df = pca_df[cols]

    return pca_df










def proj_acp_4freq(dict_pca, frequencies=FREQUENCIES):
    #récupérer les projections des individus pour les 4 acp des fréquences
    variable_acp = list()

    for freq in frequencies:
        data_pca = dict_pca[freq].T
        data_norm = (data_pca - data_pca.mean()) / data_pca.std()
    
    
        corr = data_norm.corr()
    
    
        pca = PCA(n_components=5).set_output(transform='pandas')
        data_pca_transformed = pca.fit_transform(corr)
        

        index = [label.split('_')[0] for label in data_pca_transformed.index]        
        data_pca_transformed.index = index
        data_pca_transformed.columns = [f"c{_}" for _ in range(1,len(data_pca_transformed.columns)+1)]
        data_pca_transformed.columns = [col+"_"+freq for col in data_pca_transformed.columns]
        
        variable_acp.append(data_pca_transformed)
       

    #coordonnées dans l'espace latent à utiliser pour le double SVM       
    proj_pca_concat = pd.concat(variable_acp, axis = 1)
    return proj_pca_concat

def assign_class(index):
        if index.startswith("AD"):
            return "AD"
        elif index.startswith("MCI"):
            return "MCI"
        elif index.startswith("SCI"):
            return "SCI"
        else:
            return "Unknown"