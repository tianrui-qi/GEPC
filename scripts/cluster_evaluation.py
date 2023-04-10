# -*- coding: utf-8 -*-

import json
import time
import os
import sys
import copy
import glob
import numpy as np
import pandas as pd

sys.path.insert(0,'/project/dunlop/shared_python_packages/')
import qsub

# TODO: use different deepcellcontrol folder:
username='hklumpe'
dcc_data_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"
dcc_repo_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"

# Make sure the proper path is used:
sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

#%% Functions
def params_change(params):
    """
    Store params (dict) in `~/qsub_scripts/assets/` in specific folder for
    current simulation; return location of this dict
    """
    
    # Create relevant directories:
    assets = os.path.expanduser("~") + "/qsub_scripts/assets/"
    subfolders = [ f.path for f in os.scandir(assets) if f.is_dir() ]
    foldername = assets + time.strftime("%Y-%m-%d_%H-%M-%S") + f"_submission_{len(subfolders)}/"
    os.makedirs(foldername)
    print(foldername)
    
    # Save parameters to json file:
    with open(foldername+"parameters.json","w") as f:
        json.dump(params,f, indent=4)
    
    # Return path to json file:
    return foldername+"parameters.json"

def get_params(simul_id):
    """
    Given: 
        simul_id: str (some unique part of file name in /assets/models/) 
    Returns:
        model parameters: dict
        training_params: dict
    """
    simul_dir = glob.glob(dcc_repo_path + f'/assets/models/{simul_id}*')[0]+'/'

    with open(simul_dir+'training_parameters.json','r') as f:
        training_params = json.load(f)
        
    with open(training_params['datasets_folder']+'/model_parameters.json', 'r') as f:
        model_params = json.load(f)
        
    return model_params, training_params

#%% Get information about simulations

simul_dir_list = glob.glob(dcc_repo_path + '/assets/models/2023*')
                    
simul_id_list = [simul_dir.split('/')[-1] for simul_dir in simul_dir_list]
horizon_list = []
training_folder_list = []
solver_list = []
camera_sim_list = []
past_steps_list = []
cell_class_list = []
epochs_list = []
datasets_list = []
h1_list = []
h2_list = []
mu_list = []
sigma_list = []

for simul_id in simul_id_list:
    
    model_params, training_params = get_params(simul_id)
    
    horizon_list += [training_params['horizon']]
    past_steps_list += [training_params['past_steps']]
    training_folder_list += [training_params['training_sets'][0]]
    cell_class_list += [training_params['cell_class']]
    epochs_list += [training_params['training_parameters']['epochs']]
    datasets_list += [training_params['datasets_folder']]
    
    if 'h1' in model_params.keys():
        h1_list += [model_params['h1']]
    else: h1_list += [np.nan]
    if 'h2' in model_params.keys():
        h2_list += [model_params['h2']]
    else: h2_list += [np.nan]
    if 'mu' in model_params.keys():
        mu_list += [model_params['mu']]
    else: mu_list += [np.nan]
    if 'sigma' in model_params.keys():
        sigma_list += [model_params['sigma']]
    else: sigma_list += [np.nan]
    if 'solver' in model_params.keys():
        solver_list += [model_params['solver']]
    else:
        solver_list += ['original']
    if 'camera_sim' in model_params.keys():
        camera_sim_list += [model_params['camera_sim']]
    else:
        camera_sim_list += [True]
    
df_meta = pd.DataFrame({'simul_id': simul_id_list,
                        'horizon': horizon_list,
                        'past_steps': past_steps_list,
                        'camera_sim': camera_sim_list,
                        'solver': solver_list,
                        'h1': h1_list,
                        'h2': h2_list,
                        'sigma': sigma_list,
                        'mu': mu_list,
                        'cell_class': cell_class_list,
                        'training_sets': training_folder_list,
                        'epochs': epochs_list,
                        'datasets_folder': datasets_list,})

# Drop repeats
df_meta = df_meta.sort_values(by=['simul_id','past_steps','horizon','h1','h2'])
df_meta = df_meta.drop_duplicates(subset=[col for col in df_meta.columns[1:]], keep='last')

# Only keep fully trained models
df_meta.drop(df_meta[df_meta['epochs']<200].index, inplace=True)


#%% Evaluate all models with non-standard training set set sizes

model_dir_list = df_meta.loc[df_meta['training_sets']!='training_set','simul_id'].values
    
for model_dir in [np.sort(model_dir_list)[0]]:
    
    model_dir = dcc_repo_path + '/assets/models/' + model_dir 
    
    with open(model_dir+'/training_parameters.json', 'r') as f:
        params = json.load(f)
    
    training_file = params['training_sets'][0]
    datasets_folder = params['datasets_folder']
    
    cell_class = datasets_folder.split('/')[-2]
    
    models_folder = model_dir.split('models')[0] + '/models/'
    save_folder = model_dir.split('models')[1]
    
    # # Fields to change in config.py:
    # # Training epochs, locations of training and evaluation datsets,
    # # which cell class to use for simulations
    # config = dict(
    #     training_parameters = dict(
    #         epochs = 200, # 200 is typically enough
    #         ),
    #     datasets_folder = datasets_folder, # Point to generated sets folder
    #     training_sets = (training_file,), # Training subfolder(s)
    #     eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
    #     features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
    #     cell_class = cell_class, # Cell class to use in dcc.simulations
    #     models_folder = models_folder, # Where to find the model and where to save output
    #     save_folder = save_folder,
    #     model = 'model.hdf5',
    #     eval_save_path = '/evaluation_no_validation/', # where to save evaluation output
    #     )
    
    # # Updated config and save it to disk:
    # saved_config_file = params_change(config)
    
    # # Submit qsub request for single job:
    # job_id = qsub.submit(
    #     dcc_repo_path + "scripts/simulate_evaluation.py",
    #     args = [saved_config_file],
    #     conda_env="dcc_env_shared",
    #     hardware_requirements = dict(
    #         time_limit = 1, #2
    #         cores=1, #4
    #         gpus=1,
    #         mem_per_core=4,
    #         )
    #     )
