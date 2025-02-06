# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:52:10 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen
"""

# Déplacé dans main.py
#%% TODO-list
# TODO : mettre tous les résultats (ex: accuracy) en %
# TODO : matrices corrélations 4x(4x435) (3 différentes pour les 3 états de maladie)
# TODO : Vérifier structure des données en dimensions >2D
# TODO : Implémenter méthode d'équilibre des classes (ex: SMOTE ...)
# TODO :  Comparer performances sur variables espace latent de l'ACP vs variable "reelles"
# TODO : Vérifier si SVM ADvsMCI/SCI marche mieux que MCIvsAD/SCI
# %%
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import random

# Déplacé dans utils/data_loading.py
def load_data(path, greek, state):
    data_dict = dict()
    
    for freq in greek:
        data_dict[freq] = {} 
        for condition in state.keys():
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

# Déplacé dans utils/visualization.py
def heat_map_fq_state(data_dict, greek, state, annot = False, vmin=None, vmax=None):
    for i, _ in enumerate(data_dict[greek][state]):
        plt.title(f'Matrice {greek}-{state} {i+1}')
        sns.heatmap(_, annot=annot, cmap ='viridis', vmin=vmin, vmax=vmax)
        plt.show()

# Déplacé dans utils/data_processing.py, renommé en get_value
def get_data_greek_state(data_dict, greek, state):
    return (data_dict[greek][state])

# Déplacé dans utils/data_processing.py, renommé en get_upper_matrix
def get_upper(data_freq_dstate):
    return data_freq_dstate[np.triu_indices_from(data_freq_dstate, k=1)]

# Déplacé dans utils/visualization.py
def plot_histo(data_dict, greek, state):
    for freq in greek:
        for dstate in state.keys():
            data = (get_data_greek_state(data_dict, freq, dstate))
            plt.figure(figsize=(20,10))
            for i, _ in enumerate(data):
                plt.hist(get_upper(_), label= f'patient {i+1}', bins=30)
            plt.legend()
            plt.grid()
            plt.title(f'{freq}-{dstate}\n mean={np.mean(data):.2f} std={np.std(data):.2f}')
            plt.show()

# Déplacé dans utils/visualization.py
def plot_scatter_mean_std(data_dict, greek, state):
    # Créer un graphique pour chaque fréquence
    for freq in greek:
        plt.figure(figsize=(15, 10))
        
        # Définir une liste de couleurs pour chaque état (state)
        colors = plt.cm.get_cmap('tab10', len(state.keys()))  # Utilisation d'un cmap avec différentes couleurs
        
        # Pour chaque état dans state.keys()
        for i, dstate in enumerate(state.keys()):
            data = get_data_greek_state(data_dict, freq, dstate)
            
            # Tracer les patients pour chaque état avec une couleur différente
            for j, patient in enumerate(data):
                data_patient = get_upper(patient)
                # Afficher le scatter pour chaque patient
                plt.scatter(np.mean(data_patient), np.std(data_patient), label=f'{dstate}' if j == 0 else "", color=colors(i))
                # Ajouter le texte avec le numéro du patient
                plt.text(np.mean(data_patient), np.std(data_patient), f'{j+1}', fontsize=9, ha='right', va='bottom')
        
        plt.title(f'Scatter plot for frequency: {freq}')
        plt.grid(True)
        plt.xlabel('mean')
        plt.ylabel('std')

        plt.legend(title="State")

        plt.show()

# Fusionné avec plot_scatter_mean_std
def plot_scatter_mean_std2(data_dict, greek, state, group_states=False):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'][:len(greek)]
    for freq in greek:
        c = None
        if group_states:
            plt.figure(figsize=(15,10))
        for j, dstate in enumerate(state.keys()):
            data = (get_data_greek_state(data_dict, freq, dstate))
            if not group_states:
                plt.figure(figsize=(15,10))
            else:
                # mettre les points de même état dans la même couleur
                c = colors[j]
            for i, patient in enumerate(data):
                data_patient = get_upper(patient)
                plt.scatter(np.mean(data_patient),np.std(data_patient), label=f'patient {i+1}' if not group_states else None, c = c)
                plt.text(np.mean(data_patient), np.std(data_patient), f'{i+1}', fontsize=9, ha='right', va='bottom')
            if not group_states:
                plt.title(f'{freq}-{dstate}')
                plt.xlabel('mean')
                plt.ylabel('std')
                plt.grid()
                plt.legend()
                plt.show()
            else:
                # ajouter une légende pour les états
                plt.scatter([], [], label=dstate, c=colors[j])
        if group_states:
            plt.title(f'{freq}')

        plt.xlabel('mean')
        plt.ylabel('std')
        plt.grid()
        plt.legend()
        plt.show()

# Déplacé dans utils/data_processing.py, renommé en get_mean_electrods
def get_mean_electrod(data_dict, greek, state):
    mean_results = {} 

    for freq in greek:
        mean_results[freq] = {} 
        for dstate in state.keys():
        
            data = get_data_greek_state(data_dict, freq, dstate)
            all_patients_data = []  

            for patient in data:
                data_patient = get_upper(patient)
                all_patients_data.append(data_patient)
                
            all_patients_data = np.array(all_patients_data)
            mean_per_cell = np.mean(all_patients_data, axis=0)  
            mean_results[freq][dstate] = mean_per_cell

    return mean_results

# Déplacé dans utils/data_processing.py, renommé en group_mean_by_electrods      
def group_mean_vectors_as_matrices(data_dict, greek, state):
    """
    Retourne la moyenne par électrode pour chaque patient (matrice 30x4) dans un dictionnaire avec pour clé les états
    """
    grouped_matrices = dict()
    for dstate in state.keys():
        grouped_matrices[dstate] = [] 
        data_by_frequency = []  
        
        for freq in greek:
            data = get_data_greek_state(data_dict, freq, dstate)
            
            data_mean = [np.mean(patient, axis=0) for patient in data]
            data_by_frequency.append(data_mean)  
        
        num_patients = len(data_by_frequency[0])  # Nombre de patients
        for i in range(num_patients):
            patient_matrix = np.stack([data_by_frequency[freq_idx][i] for freq_idx in range(len(greek))], axis=1)
            grouped_matrices[dstate].append(patient_matrix)
    
    return grouped_matrices

# Déplacé dans utils/data_processing.py, renommé en group_triangular_matrices
def extract_and_group_triangular_matrices_df(data_dict, greek, state):
    grouped_vectors = {}
    
    for dstate in state.keys():
        grouped_vectors[dstate] = {}  # Initialise un dictionnaire pour chaque état
        
        # Récupération des données par état et par fréquence
        data_by_frequency = {}
        for freq in greek:
            data = get_data_greek_state(data_dict, freq, dstate)  # Liste de matrices 30x30
            triangular_vectors = [
                patient[np.triu_indices_from(patient, k=1)]  # Extraire les valeurs triangulaires supérieures
                for patient in data
            ]
            data_by_frequency[freq] = triangular_vectors
        
        # Regrouper les vecteurs pour chaque patient dans un DataFrame
        num_patients = len(data_by_frequency[greek[0]])  # Nombre de patients
        for i in range(num_patients):
            # Construire un DataFrame pour chaque patient avec les fréquences comme colonnes
            patient_data = {
                freq: data_by_frequency[freq][i] for freq in greek
            }
            patient_df = pd.DataFrame(patient_data)  # Crée un DataFrame pour le patient
            grouped_vectors[dstate][i + 1] = patient_df  # Numéro du patient commence à 1
    
    return grouped_vectors

# Déplacé dans utils/data_processing.py, renommé en patient_info_by_frequency
def create_frequency_matrices_with_patient_labels(data_dict, greek, state):
    """
    Crée un DataFrame par fréquence (ALPHA, BETA, ...) avec 435 colonnes pour les vecteurs triangulaires supérieurs,
    et les patients en lignes identifiés par un nom unique (ex: AD_1, MCI_1, SCI_1, etc.).
    
    Args:
        greek (list): Liste des fréquences (ALPHA, BETA, DELTA, THETA).
        state (dict): Dictionnaire des états (AD, MCI, SCI) avec leurs données respectives.
    
    Returns:
        dict: Un dictionnaire contenant un DataFrame par fréquence.
              - Clé : Nom de la fréquence (ALPHA, BETA, etc.).
              - Valeur : DataFrame avec 435 colonnes (électrodes) et les patients en lignes.
    """
    frequency_matrices = {}
    
    for freq in greek:
        patient_vectors = []
        patient_labels = []
        
        for dstate, data_list in state.items():
            data = get_data_greek_state(data_dict, freq, dstate)  
            for i, patient in enumerate(data):
                triangular_vector = patient[np.triu_indices_from(patient, k=1)]
                patient_vectors.append(triangular_vector)
                patient_labels.append(f"{dstate}_{i+1}")
        
        frequency_matrices[freq] = pd.DataFrame(
            patient_vectors, 
            columns=[f"Electrode_{i+1}" for i in range(len(triangular_vector))],
            index=patient_labels  
        )
    
    return frequency_matrices


# Déplacé dans utils/visualization.py
def get_summary(data_dict, greek, state, by_patient=False):
   for freq in greek:
       for dstate in state.keys():
           data = (get_data_greek_state(data_dict, freq, dstate))
           
           print(f'{freq}-{dstate} mean : ', np.mean(data))
           print(f'{freq}-{dstate} std : ', np.std(data))
           # obtenir le minimum non nul
           min_ = np.min([np.min(get_upper(patient)) for patient in data if np.min(get_upper(patient)) != 0])
           print(f'{freq}-{dstate} min : ', min_)
           print(f'{freq}-{dstate} max : ', np.max(data))
           if by_patient:
               for i, patient in enumerate(data):
                   data_patient = get_upper(patient)
                   print(f'      patient {i+1:3d} mean: {np.mean(data_patient):.3f}   | std: {np.std(data_patient):.3f}')
           print()
       print("=====================================")

# Déplacé dans utils/data_processing.py
def get_data_big_corr_matrix(data_dict, state={'AD':28, 'MCI':40, 'SCI':22}, greek=['ALPHA', 'BETA', 'DELTA', 'THETA'], plot_matrix = True):
    data_big_correlation = {}
    for dstate in state.keys():
        data_state = []
        for freq in greek:
            for _ in range(len(data_dict[freq][dstate])):
                if data_dict[freq][dstate][_].shape != (30,30):
                    print(f"pb taille matrice {freq}, {state}, {_}")
                else:
                    vector = np.tril(data_dict[freq][dstate][_], k=-1)            
                vector = np.tril(vector, k=-1)
                indices = np.tril_indices(30, k=-1)
                vector = vector[indices].flatten()
                data_state.append(vector)
        
        # Concaténer les vecteurs
        concatenated_data = np.vstack(data_state)
        data_big_correlation[dstate] = concatenated_data
        
    
    if plot_matrix == True:
        for dstate in data_big_correlation.keys():
            sns.heatmap(np.corrcoef(data_big_correlation[dstate]), cmap = 'gray' )
            plt.title(f"Matrice de corrélation pour {dstate}")
            plt.show()
        
        
    return data_big_correlation     
       
# Déplacé dans utils/data_processing.py
# A éviter, privilégier la fonction pca_on_patients
def proj_acp_4freq(dict_pca, greek=['ALPHA', 'BETA', 'DELTA', 'THETA']):
    #récupérer les projections des individus pour les 4 acp des fréquences
    variable_acp = list()

    for freq in greek:
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

# Déplacé dans analysis/svm.py
def svm_skf(data, class_svm, verbose = False):
    """
    Perform SVM classification with Stratified K-Fold cross-validation.

    Args:
        data (pd.DataFrame): The input data with features and labels.
        class_svm (str or int): The class label to be used as the positive class for binary classification.
        verbose (bool, optional): If True, print the mean accuracy. Default is False.
    Returns:
        tuple: A tuple containing:
            - output (np.ndarray): The predicted labels for each instance in the data.
            - mean_accuracy (float): The mean accuracy across all folds.
    """
    index = data.index
    
    binary_labels = [1 if label == class_svm else 0 for label in index]
    copy_latent_4freq = data.copy()
    copy_latent_4freq['label'] = binary_labels

    svm = SVC(kernel='linear')
    skf = StratifiedKFold(n_splits=5)
    accuracies = []
    output = np.zeros(len(data))
    
    
    for train_index, test_index in skf.split(copy_latent_4freq.iloc[:, :-1], copy_latent_4freq['label']):
        X_train, X_test = copy_latent_4freq.iloc[train_index, :-1], copy_latent_4freq.iloc[test_index, :-1]
        y_train, y_test = copy_latent_4freq.iloc[train_index, -1], copy_latent_4freq.iloc[test_index, -1]

        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)


    if verbose:
        print(f"Mean accuracy with StratifiedKFold (Classe discriminante : {state}) : {mean_accuracy:.2f}")
    
    return output, mean_accuracy

# Déplacé dans utils/methods.py
def dict_to_df(dict_data):
    """
    Converts a nested dictionary of data into a pandas DataFrame.

    Args:
        dict_data (dict): The dictionary obtained with `load_data`.

    Returns:
        pd.DataFrame: A DataFrame with columns "State", "Patient", and many columns for the data.

    The data in the Dataframe has the nam "freq_elec1_elec2" where freq is the
    frequency (ALPHA, BETA, etc.), and elec1 and elec2 are the electrode numbers.
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
    
# Déplacé dans analysis/methods.py
def xgboost_analysis(data : dict | pd.DataFrame, verbose = False, k=5,
                     mode = "total", use_stratified_kfold = True):
    """
    Perform XGBoost classification with Stratified K-Fold cross-validation.

    Args:
        data (dict | pd.DataFrame): The input data with features and labels.
        verbose (bool, optional): If True, print the mean accuracy. Default is False.
        k (int, optional): The number of folds for Stratified K-Fold cross-validation. Default is 5.
        mode (str, optional): The mode to use for the analysis ("total" to use on all of 
            the data, or "pca" to use on the result of pca using proj_acp_4freq). Default is "total".
        use_stratified_kfold (bool, optional): If True, use Stratified K-Fold cross-validation. Default is True.
    
    Returns:
        tuple: A tuple containing:
            - output (np.ndarray): The predicted labels for each instance in the data.
            - mean_accuracy (float): The mean accuracy across all folds.

    This runs both a training and testing phase for the XGBoost model using Stratified K-Fold cross-validation.
    """
    if data.__class__ == dict:
        df_data = dict_to_df(data)
    else:
        # Copy the DataFrame to avoid modifying the original data
        df_data = data.copy()

    if mode == "pca":
        # Make the index a column named "State"
        df_data["State"] = df_data.index
    else:
        # Remove "Patient" column
        df_data = df_data.drop(columns=["Patient"])
        # Make the "State" column the last one (prevents it from being used as a feature)
        df_data = df_data[[col for col in df_data.columns if col != "State"] + ["State"]]
    
    

    # Map labels to integers
    corresp_y = {state: i for i, state in enumerate(df_data["State"].unique())}
    df_data["State"] = df_data["State"].map(corresp_y)

    model = XGBClassifier()
    if use_stratified_kfold:
        skf = StratifiedKFold(n_splits=k)
    else:
        # Use K-Fold cross-validation
        skf = KFold(n_splits=k)
    accuracies = []
    output = np.zeros(len(df_data))


    for train_index, test_index in skf.split(df_data.iloc[:, :-1], df_data["State"]):
        X_train, X_test = df_data.iloc[train_index, :-1], df_data.iloc[test_index, :-1]
        y_train, y_test = df_data.iloc[train_index]["State"], df_data.iloc[test_index]["State"]
        
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        output[test_index] = y_pred
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracies)

    if verbose:
        print(f"Mean accuracy with StratifiedKFold (mode {mode.capitalize()}): {mean_accuracy:.2f}")

    return output, mean_accuracy

# Déplacé dans utils/data_processing.py
def assign_class(index):
        if index.startswith("AD"):
            return "AD"
        elif index.startswith("MCI"):
            return "MCI"
        elif index.startswith("SCI"):
            return "SCI"
        else:
            return "Unknown"

# Déplacé dans utils/data_loading.py
def mci_aleatoires(liste_entree, nombre_mci=28):
    mci_elements = [element for element in liste_entree if element.startswith('MCI')]

    mci_aleatoires = random.sample(mci_elements, nombre_mci)

    ad_sci_elements = [element for element in liste_entree if element.startswith('AD') or element.startswith('SCI')]

    resultat = ad_sci_elements + mci_aleatoires

    return resultat
        

#%%
# Déplacé dans main.py
path = './EpEn Data_sans diag_norm_90 sujets/EpEn Data_sans diag_norm_90 sujets'
if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    greek = ['ALPHA', 'BETA', 'DELTA', 'THETA'] #bp-frequence  
    
    state = {'AD':28, 'MCI':40, 'SCI':22} #etat du trouble (keys) : nombre de patient (values)
    
    NB_ELEC = 30 #nombre d'électrode (constante)

    data_dict = load_data(path, greek, state)
    
    #heat_map_fq_state(data_dict, 'ALPHA', 'AD')
    
    #get_summary(greek, state)
    
    #plot_histo(greek, state)

    plot_scatter_mean_std(data_dict, greek, state)

    #dict_mean = get_mean_by_electrod(greek, state)  
    
    #mean_electrod = get_mean_electrod(greek, state)
    
    #mean_vectors_by_state = group_mean_vectors_as_matrices(greek, state)
    
    #grouped_vectors_df = extract_and_group_triangular_matrices_df(greek, state)
    
    data_big_correlation = get_data_big_corr_matrix(data_dict, greek=greek)

    
    dict_pca = create_frequency_matrices_with_patient_labels(data_dict, greek, state)
    
    

    #%% ACP
    acp_explo = True
    
    if acp_explo == True:

    
        do_ACP_for = 'patient' # 'patient' ou 'electrode' selon la variable à explorer (quand on projette les patients on peut voir les classes)
        
        if do_ACP_for == 'electrode':
    
            for freq in greek:
                data = dict_pca[freq]
                data_norm = (data-data.mean())/data.std()
                
                corr = (data_norm).corr()
                
                pca = PCA(n_components=20)
                pca.fit(corr)
                
                data_pca = pca.fit_transform(corr)
                
                
                #plot variance cumulée
                exp_var= (pca.explained_variance_ratio_)
                plt.plot(np.insert(np.cumsum(exp_var),0,0), color='red')
                plt.bar(np.arange(1,len(exp_var)+1,1),np.cumsum(exp_var))
                plt.axhline(.9, linestyle='--', color= 'gray')
                plt.xlabel('valeurs propres')
                plt.ylabel('%')
                plt.title(f'{freq}-variance expliquée cumulée')
                plt.show()
                
                explained_variance_ratio = pca.explained_variance_ratio_  # Variance expliquée par chaque composante
                cum_explained_variance_ratio = np.cumsum(explained_variance_ratio)
                number_of_compo_to_keep = 20 - len(cum_explained_variance_ratio[cum_explained_variance_ratio > 0.95])

                print(f"Nombre de composantes principales à garder pour {freq}: {number_of_compo_to_keep}")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Projection des individus sur les deux premières composantes principales
                ax.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', edgecolors='k', alpha=0.7)
                
                # Titres et labels
                ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.2f}%)')
                ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.2f}%)')
                ax.set_title(f'Projection des liaisons électrodes dans le plan principal ({(explained_variance_ratio[0]+explained_variance_ratio[1]) * 100:.2f}%)\n{freq}')
                plt.grid()
                plt.show()
                    
                
                # Projection des individus dans l'espace principal (3D)
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c='blue', edgecolors='k', alpha=0.7)
            
                # Titres et labels
                ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.2f}%)')
                ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.2f}%)')
                ax.set_zlabel(f'PC3 ({explained_variance_ratio[2] * 100:.2f}%)')
                ax.set_title(f'Projection des des liaisons électrodes dans l\'espace principal ({(explained_variance_ratio[0]+explained_variance_ratio[1]+explained_variance_ratio[2]) * 100:.2f}%)\n{freq}')
            
                plt.show()
            
                do_circle = False
                if do_circle == True:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    for i in range(corr.shape[1]):
                        ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], head_width=0.05, head_length=0.05, fc='k', ec='k')
                        #ax.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, X.columns[i], color='black', ha='center', va='center')
                    
                    # Tracer le cercle unitaire
                    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
                    ax.add_artist(circle)
                    
                    # Limites du graphique
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    ax.set_aspect('equal', 'box')
                    
                    # Ajouter des labels et un titre
                    ax.set_xlabel('Composante principale 1')
                    ax.set_ylabel('Composante principale 2')
                    ax.set_title('Cercle de corrélation')
                    
                    plt.grid(True)
                    plt.show()
             
                    
        else: #ACP Projection des patients dans le plan principal
            for freq in greek:
                data = dict_pca[freq].T
                data_norm = (data-data.mean())/data.std()
                
                corr = (data_norm).corr()
                
                pca = PCA(n_components=60)
                pca.fit(corr)
                
                data_pca = pca.fit_transform(corr)
                
                
                exp_var= (pca.explained_variance_ratio_)
                plt.plot(np.insert(np.cumsum(exp_var),0,0), color='red')
                plt.bar(np.arange(1,len(exp_var)+1,1),np.cumsum(exp_var))
                plt.axhline(.99, linestyle='--', color= 'gray')
                plt.xlabel('valeurs propres')
                plt.ylabel('%')
                plt.title(f'{freq}-variance expliquée cumulée')
                plt.show()
              
                
                explained_variance_ratio = pca.explained_variance_ratio_  # Variance expliquée par chaque composante
                cum_explained_variance_ratio = np.cumsum(explained_variance_ratio)
                number_of_compo_to_keep = 60 - len(cum_explained_variance_ratio[cum_explained_variance_ratio > 0.99])

                print(f"Nombre de composantes principales à garder pour {freq}: {number_of_compo_to_keep}")
                
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
        
                ax.scatter(data_pca[0:27, 0], data_pca[0:27, 1], c='blue', edgecolors='k', label='AD', alpha=0.7)
                ax.scatter(data_pca[28:68, 0], data_pca[28:68, 1], c='yellow', edgecolors='k', label='MCI', alpha=0.7)
                ax.scatter(data_pca[68:, 0], data_pca[68:, 1], c='green', edgecolors='k', label='SCI', alpha=0.7)
                # Titres et labels
                ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.2f}%)')
                ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.2f}%)')
                ax.set_title(f'Projection des individus (patients) dans le plan principal ({(explained_variance_ratio[0]+explained_variance_ratio[1]) * 100:.2f}%)\n{freq}')
                plt.grid()
                plt.legend()
                plt.show()
                
                
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
    
        
                ax.scatter(data_pca[0:28, 0], data_pca[0:28, 1], data_pca[0:28, 2], c='blue', edgecolors='k', label='AD', alpha=0.7)
                ax.scatter(data_pca[29:70, 0], data_pca[29:70, 1],  data_pca[29:70, 2], c='yellow', edgecolors='k', label='MCI', alpha=0.7)
                ax.scatter(data_pca[71:, 0], data_pca[71:, 1], data_pca[71:, 2], c='green', edgecolors='k', label='SCI', alpha=0.7)
                # Titres et labels
                ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.2f}%)')
                ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.2f}%)')
                ax.set_zlabel(f'PC3 ({explained_variance_ratio[2] * 100:.2f}%)')
                ax.set_title(f'Projection des individus (patients) dans l\'espace principal ({(explained_variance_ratio[0]+explained_variance_ratio[1]+explained_variance_ratio[2]) * 100:.2f}%)\n{freq}')
                plt.grid()
                plt.legend()
                plt.show()
    

    #%% SVM comparatif sur Beta 
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report

    #1 ACP sur BETA
    
    beta_data = dict_pca["BETA"].T
    beta_data_norm = (beta_data-beta_data.mean())/beta_data.std()
    
    beta_corr = (beta_data_norm).corr()
    
    beta_pca = PCA(n_components=2)
    
    beta_pca.fit(beta_corr)
    beta_pca.set_output(transform = "pandas")
    beta_data_pca = beta_pca.fit_transform(beta_corr)
    
    

    liste_individus = beta_data.T.index
    liste_individus_equilibre = mci_aleatoires(liste_individus)
    
    beta_data_pca_filtre = beta_data_pca.loc[liste_individus_equilibre]
    
    

    # Ajout de la colonne classe aux individus sélectionnés
    beta_data_pca_filtre = beta_data_pca.loc[liste_individus_equilibre].copy()
    beta_data_pca_filtre["classe"] = beta_data_pca_filtre.index.map(assign_class)
    
    # Séparer les features et labels
    X_train = beta_data_pca_filtre.drop(columns=["classe"])
    y_train = beta_data_pca_filtre["classe"]
    
    # Standardiser les données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(beta_data_pca)
    
    # SVM avec multi-classe direct (One-vs-One + class_weight)
    svm_model = SVC(kernel="rbf", class_weight="balanced", decision_function_shape="ovo")
    svm_model.fit(X_train_scaled, y_train)
    
    # Prédiction sur l'ensemble des données
    y_pred = svm_model.predict(X_test_scaled)
    
    # Générer les labels réels pour beta_data_pca
    y_true = np.array([assign_class(idx) for idx in beta_data_pca.index])
    
    # Matrice de confusion et rapport de classification
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["AD", "MCI", "SCI"])
    sns.heatmap(conf_matrix, cmap='gray', annot=True,  xticklabels=["AD", "MCI", "SCI"], yticklabels=["AD", "MCI", "SCI"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité")
    plt.title("Matrice de confusion (BETA only)")
    plt.show()
    
    class_report = classification_report(y_true, y_pred, target_names=["AD", "MCI", "SCI"])
    
    print("Matrice de confusion :\n", conf_matrix)
    print("\nRapport de classification :\n", class_report)
        
#%% SVM multiclasse 4 FREQ CONCAT

    #1 ACP sur ALPHA
    
    alpha_data = dict_pca["ALPHA"].T
    alpha_data_norm = (alpha_data-alpha_data.mean())/alpha_data.std()
    
    alpha_corr = (alpha_data_norm).corr()
    
    alpha_pca = PCA(n_components=5)
    
    alpha_pca.fit(alpha_corr)
    alpha_pca.set_output(transform = "pandas")
    alpha_data_pca = alpha_pca.fit_transform(alpha_corr)
    alpha_data_pca.columns = ["ALPHA_"+ col for col in alpha_data_pca.columns.to_list()] 
    
    #1 ACP sur BETA
    
    beta_data = dict_pca["BETA"].T
    beta_data_norm = (beta_data-beta_data.mean())/beta_data.std()
    
    beta_corr = (beta_data_norm).corr()
    
    beta_pca = PCA(n_components=2)
    
    beta_pca.fit(beta_corr)
    beta_pca.set_output(transform = "pandas")
    beta_data_pca = beta_pca.fit_transform(beta_corr)
    beta_data_pca.columns = ["BETA_"+ col for col in beta_data_pca.columns.to_list()] 


    #1 ACP sur DELTA
    
    delta_data = dict_pca["DELTA"].T
    delta_data_norm = (delta_data-delta_data.mean())/delta_data.std()
    
    delta_corr = (delta_data_norm).corr()
    
    delta_pca = PCA(n_components=2)
    
    delta_pca.fit(delta_corr)
    delta_pca.set_output(transform = "pandas")
    delta_data_pca = delta_pca.fit_transform(delta_corr)
    delta_data_pca.columns = ["DELTA_"+ col for col in delta_data_pca.columns.to_list()] 

    
    
    #1 ACP sur theta
    
    theta_data = dict_pca["THETA"].T
    theta_data_norm = (theta_data-theta_data.mean())/theta_data.std()
    
    theta_corr = (theta_data_norm).corr()
    
    theta_pca = PCA(n_components=32)
    
    theta_pca.fit(theta_corr)
    theta_pca.set_output(transform = "pandas")
    theta_data_pca = theta_pca.fit_transform(theta_corr)
    theta_data_pca.columns = ["THETA_"+ col for col in theta_data_pca.columns.to_list()] 

    
    
    # Concaténer les données des 4 bandes de fréquence
    df_4freq = pd.concat([alpha_data_pca, beta_data_pca, delta_data_pca, theta_data_pca], axis=1)
    
    # Ajouter la colonne "classe"
    df_4freq["classe"] = df_4freq.index.map(assign_class)
    
    # Séparer les features et labels
    X = df_4freq.drop(columns=["classe"]).values
    y = df_4freq["classe"].values
    
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialisation de LOOCV
    loo = LeaveOneOut()
    
    y_true_list = []
    y_pred_list = []
    
    
    # Validation en LOOCV
    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Entraînement du modèle SVM
        svm_model = SVC(kernel="rbf", class_weight="balanced", decision_function_shape="ovo")
        svm_model.fit(X_train, y_train)
    
        # Prédiction
        y_pred = svm_model.predict(X_test)
    
        # Stocker les résultats
        y_true_list.append(y_test[0])
        y_pred_list.append(y_pred[0])
    
    # Matrice de confusion et rapport de classification
    conf_matrix = confusion_matrix(y_true_list, y_pred_list, labels=["AD", "MCI", "SCI"])
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="gray", xticklabels=["AD", "MCI", "SCI"], yticklabels=["AD", "MCI", "SCI"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité")
    plt.title("Matrice de confusion 4 fréquences concaténées")
    plt.show()
    
    class_report = classification_report(y_true_list, y_pred_list, target_names=["AD", "MCI", "SCI"])
    print("Matrice de confusion :\n", conf_matrix)
    print("\nRapport de classification :\n", class_report)
    
#%% SVM 1 VS OTHERS sur 4 FREQ

    #1 ACP sur ALPHA
    
    alpha_data = dict_pca["ALPHA"].T
    alpha_data_norm = (alpha_data-alpha_data.mean())/alpha_data.std()
    
    alpha_corr = (alpha_data_norm).corr()
    
    alpha_pca = PCA(n_components=5)
    
    alpha_pca.fit(alpha_corr)
    alpha_pca.set_output(transform = "pandas")
    alpha_data_pca = alpha_pca.fit_transform(alpha_corr)
    alpha_data_pca.columns = ["ALPHA_"+ col for col in alpha_data_pca.columns.to_list()] 
    
    #1 ACP sur BETA
    
    beta_data = dict_pca["BETA"].T
    beta_data_norm = (beta_data-beta_data.mean())/beta_data.std()
    
    beta_corr = (beta_data_norm).corr()
    
    beta_pca = PCA(n_components=2)
    
    beta_pca.fit(beta_corr)
    beta_pca.set_output(transform = "pandas")
    beta_data_pca = beta_pca.fit_transform(beta_corr)
    beta_data_pca.columns = ["BETA_"+ col for col in beta_data_pca.columns.to_list()] 
    
    
    #1 ACP sur DELTA
    
    delta_data = dict_pca["DELTA"].T
    delta_data_norm = (delta_data-delta_data.mean())/delta_data.std()
    
    delta_corr = (delta_data_norm).corr()
    
    delta_pca = PCA(n_components=2)
    
    delta_pca.fit(delta_corr)
    delta_pca.set_output(transform = "pandas")
    delta_data_pca = delta_pca.fit_transform(delta_corr)
    delta_data_pca.columns = ["DELTA_"+ col for col in delta_data_pca.columns.to_list()] 
    
    
    
    #1 ACP sur theta
    
    theta_data = dict_pca["THETA"].T
    theta_data_norm = (theta_data-theta_data.mean())/theta_data.std()
    
    theta_corr = (theta_data_norm).corr()
    
    theta_pca = PCA(n_components=32)
    
    theta_pca.fit(theta_corr)
    theta_pca.set_output(transform = "pandas")
    theta_data_pca = theta_pca.fit_transform(theta_corr)
    theta_data_pca.columns = ["THETA_"+ col for col in theta_data_pca.columns.to_list()] 
    
    
    
    # Concaténer les données des 4 bandes de fréquence
    df_4freq = pd.concat([alpha_data_pca, beta_data_pca, delta_data_pca, theta_data_pca], axis=1)
    
    # Ajouter la colonne "classe"
    df_4freq["classe"] = df_4freq.index.map(assign_class)
    
    # Séparer les features et labels
    X = df_4freq.drop(columns=["classe"]).values
    y = df_4freq["classe"].values
    
    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialisation de LOOCV
    loo = LeaveOneOut()
    
    
    y_true_list = []
    y_pred_list = []
    
    # 2. Initialisation des SVM binaires (One-Versus-Other)
    svm_models = {
        "AD": SVC(kernel="rbf", class_weight="balanced", probability=True),
        "MCI": SVC(kernel="rbf", class_weight="balanced", probability=True),
        "SCI": SVC(kernel="rbf", class_weight="balanced", probability=True),
    }
    
    # 3. Validation croisée en LOOCV
    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Créer une matrice de probabilités pour chaque classe
        probas = []
        
        # Entraîner un SVM binaire pour chaque classe
        for classes in svm_models.keys():
            y_train_binary = (y_train == classes).astype(int)  # Classe = 1 pour la classe cible, 0 sinon
            svm_models[classes].fit(X_train, y_train_binary)
            
            # Prédire la probabilité pour la classe cible
            probas.append(svm_models[classes].predict_proba(X_test)[:, 1])  # Probabilité pour la classe 1 (positive)
        
        # Convertir les probabilités en un array numpy
        probas = np.array(probas).T  # Shape (n_samples, n_classes)
        
        # Prendre la classe avec la plus grande probabilité
        y_pred = np.argmax(probas, axis=1)  # Index de la classe
        classes = list(svm_models.keys())
        y_pred_label = classes[y_pred[0]]
        
        # Stocker la vérité terrain et la prédiction
        y_true_list.append(y_test[0])
        y_pred_list.append(y_pred_label)
    
    # 4. Matrice de confusion et rapport de classification
    conf_matrix = confusion_matrix(y_true_list, y_pred_list, labels=["AD", "MCI", "SCI"])
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="gray", xticklabels=["AD", "MCI", "SCI"], yticklabels=["AD", "MCI", "SCI"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité")
    plt.title("Matrice de confusion 4 fréquences concaténées (One-Versus-Other)")
    plt.show()
    
    class_report = classification_report(y_true_list, y_pred_list, target_names=["AD", "MCI", "SCI"])
    print("Matrice de confusion :\n", conf_matrix)
    print("\nRapport de classification :\n", class_report)
