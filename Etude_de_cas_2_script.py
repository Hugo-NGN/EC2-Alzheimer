# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:52:10 2025

@author: nathan.piatte, felix.courtin, hugo.nguyen
"""
# %%
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import os 

def load_data(greek, state):
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

def heat_map_fq_state(greek, state, annot = False, vmin=None, vmax=None):
    for i, _ in enumerate(data_dict[greek][state]):
        plt.title(f'Matrice {greek}-{state} {i+1}')
        sns.heatmap(_, annot=annot, cmap ='viridis', vmin=vmin, vmax=vmax)
        plt.show()
        
def get_data_greek_state(greek, state):
    return (data_dict[greek][state])

def get_upper(data_freq_dstate):
    return data_freq_dstate[np.triu_indices_from(data_freq_dstate, k=1)]

def plot_histo(greek, state):
    for freq in greek:
        for dstate in state.keys():
            data = (get_data_greek_state(freq, dstate))
            plt.figure(figsize=(20,10))
            for i, _ in enumerate(data):
                plt.hist(get_upper(_), label= f'patient {i+1}', bins=30)
            plt.legend()
            plt.grid()
            plt.title(f'{freq}-{dstate}\n mean={np.mean(data):.2f} std={np.std(data):.2f}')
            plt.show()
            

def plot_scatter_mean_std(greek, state):
    # Créer un graphique pour chaque fréquence
    for freq in greek:
        plt.figure(figsize=(15, 10))
        
        # Définir une liste de couleurs pour chaque état (state)
        colors = plt.cm.get_cmap('tab10', len(state.keys()))  # Utilisation d'un cmap avec différentes couleurs
        
        # Pour chaque état dans state.keys()
        for i, dstate in enumerate(state.keys()):
            data = get_data_greek_state(freq, dstate)
            
            # Tracer les patients pour chaque état avec une couleur différente
            for j, patient in enumerate(data):
                data_patient = get_upper(patient)
                # Afficher le scatter pour chaque patient
                plt.scatter(np.mean(data_patient), np.std(data_patient), label=f'{dstate}' if j == 0 else "", color=colors(i))
                # Ajouter le texte avec le numéro du patient
                plt.text(np.mean(data_patient), np.std(data_patient), f'{j+1}', fontsize=9, ha='right', va='bottom')
        
        plt.title(f'Scatter plot for frequency: {freq}')
        plt.grid(True)

        plt.legend(title="State")

        plt.show()
        
def plot_scatter_mean_std2(greek, state, group_states=False):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'][:len(greek)]
    for freq in greek:
        c = None
        if group_states:
            plt.figure(figsize=(15,10))
        for j, dstate in enumerate(state.keys()):
            data = (get_data_greek_state(freq, dstate))
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

            
def get_mean_electrod(greek, state):
    mean_results = {} 

    for freq in greek:
        mean_results[freq] = {} 
        for dstate in state.keys():
        
            data = get_data_greek_state(freq, dstate)
            all_patients_data = []  

            for patient in data:
                data_patient = get_upper(patient)
                all_patients_data.append(data_patient)
                
            all_patients_data = np.array(all_patients_data)
            mean_per_cell = np.mean(all_patients_data, axis=0)  
            mean_results[freq][dstate] = mean_per_cell

    return mean_results

                
def group_mean_vectors_as_matrices(greek, state):
    """
    Retourne la moyenne par électrode pour chaque patient (matrice 30x4) dans un dictionnaire avec pour clé les états
    """
    grouped_matrices = dict()
    for dstate in state.keys():
        grouped_matrices[dstate] = [] 
        data_by_frequency = []  
        
        for freq in greek:
            data = get_data_greek_state(freq, dstate)
            
            data_mean = [np.mean(patient, axis=0) for patient in data]
            data_by_frequency.append(data_mean)  
        
        num_patients = len(data_by_frequency[0])  # Nombre de patients
        for i in range(num_patients):
            patient_matrix = np.stack([data_by_frequency[freq_idx][i] for freq_idx in range(len(greek))], axis=1)
            grouped_matrices[dstate].append(patient_matrix)
    
    return grouped_matrices

def extract_and_group_triangular_matrices_df(greek, state):
    grouped_vectors = {}
    
    for dstate in state.keys():
        grouped_vectors[dstate] = {}  # Initialise un dictionnaire pour chaque état
        
        # Récupération des données par état et par fréquence
        data_by_frequency = {}
        for freq in greek:
            data = get_data_greek_state(freq, dstate)  # Liste de matrices 30x30
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

def create_frequency_matrices_with_patient_labels(greek, state):
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
            data = get_data_greek_state(freq, dstate)  
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


def get_summary(greek, state, by_patient=False):
   for freq in greek:
       for dstate in state.keys():
           data = (get_data_greek_state(freq, dstate))
           
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

#%%
path = './EpEn Data_sans diag_norm_90 sujets/EpEn Data_sans diag_norm_90 sujets'
if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    greek = ['ALPHA', 'BETA', 'DELTA', 'THETA'] #bp-frequence  
    
    state = {'AD':28, 'MCI':40, 'SCI':22} #etat du trouble (keys) : nombre de patient (values)
    
    NB_ELEC = 30 #nombre d'électrode (constante)

    greek = ['ALPHA', 'BETA', 'DELTA', 'THETA']  # bp-frequence  
    
    state = {'AD':28, 'MCI':40, 'SCI':22}  # etat du trouble (keys) : nombre de patient (values)
    
    NB_ELEC = 30  # nombre d'électrode (constante)


    data_dict = load_data(greek, state)
    
    #heat_map_fq_state('ALPHA', 'AD')
    
    #get_summary(greek, state)
    
    #plot_histo(greek, state)

    plot_scatter_mean_std(greek, state)

    #dict_mean = get_mean_by_electrod(greek, state)  
    
    mean_electrod = get_mean_electrod(greek, state)
    
    #mean_vectors_by_state = group_mean_vectors_as_matrices(greek, state)
    
    #grouped_vectors_df = extract_and_group_triangular_matrices_df(greek, state)
    
    dict_pca = create_frequency_matrices_with_patient_labels(greek, state)

    #%%
    from sklearn.decomposition import PCA
    
    freq = 'DELTA'
    for freq in greek:
        data = dict_pca[freq]
        data_norm = (data-data.mean())/data.std()
        
        corr = (data_norm).corr()
        
        pca = PCA(n_components=5)
        pca.fit(corr)
        
        data_pca = pca.fit_transform(corr)
        
        exp_var= (pca.explained_variance_ratio_)
        plt.plot(np.insert(np.cumsum(exp_var),0,0), color='red')
        plt.bar(np.arange(1,len(exp_var)+1,1),np.cumsum(exp_var))
        #plt.axhline(.9, linestyle='--', color= 'gray')
        plt.xlabel('valeurs propres')
        plt.ylabel('%')
        plt.title(f'{freq}-variance expliquée cumulée')
        plt.show()
        
        explained_variance_ratio = pca.explained_variance_ratio_  # Variance expliquée par chaque composante
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Projection des individus sur les deux premières composantes principales
        ax.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', edgecolors='k', alpha=0.7)
        
        # Titres et labels
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.2f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.2f}%)')
        ax.set_title(f'Projection des individus dans le plan principal ({(explained_variance_ratio[0]+explained_variance_ratio[1]) * 100:.2f}%)\n{freq}')
        
        plt.show()
            
    
        do_circle = True
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
    
#%%

