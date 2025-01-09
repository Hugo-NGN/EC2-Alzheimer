# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:52:10 2025

@author: hugo.nguyen
"""

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

def heat_map_fq_state(greek, state, annot = False):
    for i, _ in enumerate(data_dict[greek][state]):
        plt.title(f'Matrice {greek}-{state} {i+1}')
        sns.heatmap(_, annot=annot, cmap ='viridis')
        plt.show()
    

#%%
if __name__ == '__main__':
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path = './EpEn Data_sans diag_norm_90 sujets/EpEn Data_sans diag_norm_90 sujets'
    
    greek = ['ALPHA', 'BETA', 'DELTA', 'THETA'] #bp-frequence  
    
    state = {'AD':28, 'MCI':40, 'SCI':22} #etat du trouble (keys) : nombre de patient (values)
    
    NB_ELEC = 30 #nombre d'électrode (constante)


    data_dict = load_data(greek, state)
    
    #heat_map_fq_state('ALPHA', 'AD')
    
    