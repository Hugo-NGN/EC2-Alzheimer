# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 16:27:01 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from utils.data_processing import (
    get_value,
    get_upper_matrix,
)
from utils.global_variables import FREQUENCIES, STATES

def heat_map_fq_state(data_dict, frequency, state, annot = False, vmin=None, vmax=None):
    for i, _ in enumerate(data_dict[frequency][state]):
        plt.title(f'Matrice {frequency}-{state} {i+1}')
        sns.heatmap(_, annot=annot, cmap ='viridis', vmin=vmin, vmax=vmax)
        plt.show()

def plot_histo(data_dict, frequencies = FREQUENCIES, states = STATES):
    for freq in frequencies:
        for dstate in states.keys():
            data = (get_value(data_dict, freq, dstate))
            plt.figure(figsize=(20,10))
            for i, _ in enumerate(data):
                plt.hist(get_upper_matrix(_), label= f'patient {i+1}', bins=30)
            plt.legend()
            plt.grid()
            plt.title(f'{freq}-{dstate}\n mean={np.mean(data):.2f} std={np.std(data):.2f}')
            plt.show()

def plot_scatter_mean_std(data_dict, frequencies = FREQUENCIES, states = STATES, group_states=False):
    """
    Tracer des graphiques de dispersion de la moyenne vs. l'écart-type pour 
    chaque fréquence et état.

    Args:
        data_dict (dict): Le dictionnaire de données contenant les données.
        frequencies (list): Liste des fréquences. Par défaut, FREQUENCIES.
        states (dict): Dictionnaire des états des patients. Par défaut, STATES.
        group_states (bool, optionnel): Si True, regrouper les états dans le 
            même graphique. Par défaut, False.
    """
    colors = plt.cm.get_cmap('tab10', len(states.keys()))  # Define a color map for states

    for freq in frequencies:
        if group_states:
            plt.figure(figsize=(15, 10))

        for i, dstate in enumerate(states.keys()):
            data = get_value(data_dict, freq, dstate)
            color = colors(i) if group_states else None

            if not group_states:
                plt.figure(figsize=(15, 10))

            for j, patient in enumerate(data):
                data_patient = get_upper_matrix(patient)
                plt.scatter(np.mean(data_patient), np.std(data_patient), label=f'{dstate}' if j == 0 and not group_states else "", color=color)
                plt.text(np.mean(data_patient), np.std(data_patient), f'{j+1}', fontsize=9, ha='right', va='bottom')

            if not group_states:
                plt.title(f'{freq} - {dstate}')
                plt.xlabel('mean')
                plt.ylabel('std')
                plt.grid()
                plt.legend()
                plt.show()
            else:
                plt.scatter([], [], label=dstate, color=color)

        if group_states:
            plt.title(f'Scatter plot for frequency: {freq}')
            plt.xlabel('mean')
            plt.ylabel('std')
            plt.grid()
            plt.legend(title="State")
            plt.show()

def get_summary(data_dict, frequencies = FREQUENCIES, states = STATES, by_patient=False):
    """
    Affiche un résumé 1D des données pour chaque fréquence et état.

    Arguments:
        data_dict (dict): Dictionnaire de données obtenu avec `load_data`.
        frequencies (list): Liste des fréquences. Par défaut, FREQUENCIES.
        states (dict): Dictionnaire des états des patients. Par défaut, STATES.
        by_patient (bool, optionnel): Si True, affiche les statistiques pour chaque
            patient. Par défaut, False.
    """
    for freq in frequencies:
        for dstate in states.keys():
            data = (get_value(data_dict, freq, dstate))
            
            print(f'{freq}-{dstate} mean : ', np.mean(data))
            print(f'{freq}-{dstate} std : ', np.std(data))
            # obtenir le minimum non nul
            min_ = np.min([np.min(get_upper_matrix(patient)) for patient in data if np.min(get_upper_matrix(patient)) != 0])
            print(f'{freq}-{dstate} min : ', min_)
            print(f'{freq}-{dstate} max : ', np.max(data))
            if by_patient:
                for i, patient in enumerate(data):
                    data_patient = get_upper_matrix(patient)
                    print(f'      patient {i+1:3d} mean: {np.mean(data_patient):.3f}   | std: {np.std(data_patient):.3f}')
            print()
        print("=====================================")


def complete_classification_report(y_true, y_pred, method_title = None):
    print("Résultats de la classification :")
    print(classification_report(y_true, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_true, y_pred))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True,   xticklabels=["AD", "MCI", "SCI"], yticklabels=["AD", "MCI", "SCI"])
    if method_title !=None:
        plt.title(f"Matrice de confusion ({method_title})")
    else:
        plt.title("Matrice de confusion")
    plt.show()