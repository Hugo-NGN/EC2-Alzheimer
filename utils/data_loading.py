# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 16:26:17 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen
"""

import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

from utils.global_variables import FREQUENCIES, STATES


def load_data(path, frequencies=FREQUENCIES, states=STATES):
    data_dict = dict()

    for freq in frequencies:
        data_dict[freq] = {}
        for condition in states.keys():
            folder_path = os.path.join(path, freq, condition)

            files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
            data_list = []
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    mat_data = loadmat(file_path)
                    data = mat_data[list(mat_data.keys())[-1]]
                    data_list.append(data)
                except Exception as e:
                    print(f"Erreur de chargement de {file_path}: {e}")

            data_dict[freq][condition] = data_list
    return data_dict


def dict_to_df(dict_data):
    """
    Convertit un dictionnaire imbriqué de données en un DataFrame pandas.

    Arguments:
        dict_data (dict): Le dictionnaire obtenu avec `load_data`.

    Retourne:
        pd.DataFrame: Un DataFrame avec les colonnes "State", "Patient", et de nombreuses colonnes pour les données.

    Les données dans le DataFrame ont le nom "freq_elec1_elec2" où freq est la
    fréquence (ALPHA, BETA, etc.), et elec1 et elec2 sont les numéros des électrodes.
    """
    temp_list = []
    count = 0
    for freq, freq_data in dict_data.items():
        for state, state_data in freq_data.items():
            for i, patient_data in enumerate(state_data):
                temp_dict = {
                    "State": state,
                    "Patient": f"{i+1}"
                }
                for elec1 in range(30):
                    for elec2 in range(elec1+1, 30):
                        key = f"{freq}_{elec1}_{elec2}"
                        temp_dict[key] = patient_data[elec1, elec2]
                temp_list.append(temp_dict)
                count += 1

    result_list = []
    # For every patient (identified by the (State, Patient) pair), we make a single row
    nb_patients = len(temp_list) // len(dict_data)
    for i in range(0, nb_patients):
        # We take the data for the 4 frequencies
        list_dicts = [temp_list[i+j*nb_patients] for j in range(4)]

        # We merge the dictionaries
        merged_dict = {}
        for d in list_dicts:
            merged_dict.update(d)

        result_list.append(merged_dict)

    return pd.DataFrame(result_list)


def mci_aleatoires(liste_entree, nombre_mci=28):
    """
    Tire aléatoirement un nombre donné de patients MCI parmi une liste de patients.

    Arguments:
        liste_entree (list): La liste des patients.
        nombre_mci (int, optionnel): Le nombre de patients MCI à tirer aléatoirement.
            Par défaut, 28.
    
    Retourne:
        list: La liste des patients tirés aléatoirement
    """
    mci_elements = [
        element for element in liste_entree if element.startswith('MCI')
    ]

    mci_aleatoires = np.random.choice(mci_elements, nombre_mci, replace=False).tolist()

    ad_sci_elements = [element for element in liste_entree if element.startswith(
        'AD') or element.startswith('SCI')]

    resultat = ad_sci_elements + mci_aleatoires

    return resultat
