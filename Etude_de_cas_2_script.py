# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:52:10 2025

@author: hugo.nguyen
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
    for freq in greek:
        for dstate in state.keys():
            data = (get_data_greek_state(freq, dstate))
            plt.figure(figsize=(15,10))
            for i, patient in enumerate(data):
                data_patient = get_upper(patient)
                plt.scatter(np.mean(data_patient),np.std(data_patient), label=f'patient {i+1}')
                plt.text(np.mean(data_patient), np.std(data_patient), f'{i+1}', fontsize=9, ha='right', va='bottom')
            plt.title(f'{freq}-{dstate}')
            plt.xlabel('mean')
            plt.ylabel('std')
            plt.grid()
            plt.legend()
            plt.show()

def get_summary(greek, state, by_patient=False):
    for freq in greek:
        for dstate in state.keys():
            data = (get_data_greek_state(freq, dstate))
            
            print(f'{freq}-{dstate} mean : ', np.mean(data))
            print(f'{freq}-{dstate} std : ', np.std(data))
            # get min that is not 0
            min_ = np.min([np.min(get_upper(patient)) for patient in data if np.min(get_upper(patient)) != 0])
            print(f'{freq}-{dstate} min : ', min_)
            print(f'{freq}-{dstate} max : ', np.max(data))
            for i, patient in enumerate(data):
                data_patient = get_upper(patient)
                print(f'      patient {i+1} mean: {np.mean(data_patient):.3f}   | std: {np.std(data_patient):.3f}')
            print()   
                
#%%
path = './EpEn Data_sans diag_norm_90 sujets/EpEn Data_sans diag_norm_90 sujets'
if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    greek = ['ALPHA', 'BETA', 'DELTA', 'THETA'] #bp-frequence  
    
    state = {'AD':28, 'MCI':40, 'SCI':22} #etat du trouble (keys) : nombre de patient (values)
    
    NB_ELEC = 30 #nombre d'électrode (constante)


    data_dict = load_data(greek, state)
    
    heat_map_fq_state('ALPHA', 'AD')
    get_summary(greek, state)
    
    plot_histo(greek, state)

    plot_scatter_mean_std(greek, state)
    