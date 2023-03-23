# -*- coding: utf-8 -*-

import json
import time
import os
import sys
import copy
import glob
import numpy as np

sys.path.insert(0,'/project/dunlop/shared_python_packages/')
import qsub

# TODO: use different deepcellcontrol folder:
username='hklumpe'
dcc_data_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"
dcc_repo_path = f"/projectnb/dunlop/{username}/deepcellcontrol/"

# Make sure the proper path is used:
sys.path.insert(0,dcc_repo_path)
import deepcellcontrol as dcc

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

#%% Train simple activation models with standard settings (e.g. past, future, training set size)

# Cell class: in training data path, and name of class in dcc.simulations
cell_class_list = ['CcaSR_gillespie_simple_noE',
                   'CcaSR_gillespie']

for cell_class in cell_class_list:
        
    training_dir_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023-03-19*/training*")
    
    # # Keep only those with base parameter values (3/16 first, all on 3/19)
    # training_dir = np.sort(np.array(training_dir_list))[0]
    
    for training_dir in training_dir_list:
        
        training_file = training_dir.split('/')[-1]
        datasets_folder = '/'.join(training_dir.split('/')[:-1])
        
                
        # Fields to change in config.py:
        # Training epochs, locations of training and evaluation datsets,
        # which cell class to use for simulations
        config = dict(
            training_parameters = dict(
                epochs = 200, # 200 is typically enough
                ),
            datasets_folder = datasets_folder, # Point to generated sets folder
            training_sets = (training_file,), # Training subfolder(s)
            eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
            features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
            cell_class = cell_class, # Cell class to use in dcc.simulations
            models_folder = dcc_repo_path + '/assets/models/',
            )
        
        # Updated config and save it to disk:
        saved_config_file = params_change(config)
        
        # Submit qsub request for single job:
        job_id = qsub.submit(
            dcc_repo_path + "scripts/simulated_pipeline.py",
            args = [saved_config_file],
            conda_env="dcc_env_shared",
            hardware_requirements = dict(
                time_limit = 5, #2
                cores=6, #4
                gpus=1,
                mem_per_core=4,
                )
            )



#%% E dynamics (Gillespie) x [past, future]

# Cell class: in training data path, and name of class in dcc.simulations
cell_class = 'CcaSR_gillespie'

# horizon_list = [12, 24, 48]

# for horizon in horizon_list:
        
#     training_dir_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023*/training*")
#     training_file_list = [train_dir.split('/')[-1] for train_dir in training_dir_list]
    
#     for training_dir in training_dir_list:
        
#         training_file = training_dir.split('/')[-1]
#         datasets_folder = '/'.join(training_dir.split('/')[:-1])
                        
#         # Fields to change in config.py:
#         # Training epochs, locations of training and evaluation datsets,
#         # which cell class to use for simulations
#         config = dict(
#             training_parameters = dict(
#                 epochs = 200, # 200 is typically enough
#                 ),
#             datasets_folder = datasets_folder, # Point to generated sets folder
#             training_sets = (training_file,), # Training subfolder(s)
#             eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
#             features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
#             cell_class = cell_class, # Cell class to use in dcc.simulations
#             models_folder = dcc_repo_path + '/assets/models/',
#             horizon = horizon,
#             )
        
#         # Updated config and save it to disk:
#         saved_config_file = params_change(config)
        
#         # Submit qsub request for single job:
#         job_id = qsub.submit(
#             dcc_repo_path + "scripts/simulated_pipeline.py",
#             args = [saved_config_file],
#             conda_env="dcc_env_shared",
#             hardware_requirements = dict(
#                 time_limit = 5, #2
#                 cores=6, #4
#                 gpus=1,
#                 mem_per_core=4,
#                 )
#             )
        
# past_steps_list = [12, 24, 36]
past_steps_list = [3, 6]

for past_steps in past_steps_list:
        
    training_dir_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023-03-16*/training*")
    training_file_list = [train_dir.split('/')[-1] for train_dir in training_dir_list]
    
    for training_dir in training_dir_list:
        
        training_file = training_dir.split('/')[-1]
        datasets_folder = '/'.join(training_dir.split('/')[:-1])
                        
        # Fields to change in config.py:
        # Training epochs, locations of training and evaluation datsets,
        # which cell class to use for simulations
        config = dict(
            training_parameters = dict(
                epochs = 200, # 200 is typically enough
                ),
            datasets_folder = datasets_folder, # Point to generated sets folder
            training_sets = (training_file,), # Training subfolder(s)
            eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
            features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
            cell_class = cell_class, # Cell class to use in dcc.simulations
            models_folder = dcc_repo_path + '/assets/models/',
            past_steps = past_steps,
            )
        
        # Updated config and save it to disk:
        saved_config_file = params_change(config)
        
        # Submit qsub request for single job:
        job_id = qsub.submit(
            dcc_repo_path + "scripts/simulated_pipeline.py",
            args = [saved_config_file],
            conda_env="dcc_env_shared",
            hardware_requirements = dict(
                time_limit = 5, #2
                cores=6, #4
                gpus=1,
                mem_per_core=4,
                )
            )


#%% Gillespie simple (E sampled) for different training set sizes

# Cell class: in training data path, and name of class in dcc.simulations
cell_class = 'CcaSR_gillespie_simple'
        
cell_class_dir = f'{dcc_repo_path}/assets/simulated/data/{cell_class}/'
# training_dir_list = glob.glob(cell_class_dir + '2023-03-18*3d5*/training*') + \
#                     glob.glob(cell_class_dir + '2023-03-18*413*/training*')
training_dir_list = glob.glob(cell_class_dir + '2023-03-18_11-48*/*_10_*') + \
                    glob.glob(cell_class_dir + '2023-03-18_11-48*/*_2_*')
training_file_list = [train_dir.split('/')[-1] for train_dir in training_dir_list]

for training_dir in training_dir_list:
    
    training_file = training_dir.split('/')[-1]
    datasets_folder = '/'.join(training_dir.split('/')[:-1])
                
    # Fields to change in config.py:
    # Training epochs, locations of training and evaluation datsets,
    # which cell class to use for simulations
    config = dict(
        training_parameters = dict(
            epochs = 200, # 200 is typically enough
            ),
        datasets_folder = datasets_folder, # Point to generated sets folder
        training_sets = (training_file,), # Training subfolder(s)
        eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
        features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
        cell_class = cell_class, # Cell class to use in dcc.simulations
        models_folder = dcc_repo_path + '/assets/models/',
        )
    
    # Updated config and save it to disk:
    saved_config_file = params_change(config)
    
    # Submit qsub request for single job:
    job_id = qsub.submit(
        dcc_repo_path + "scripts/simulated_pipeline.py",
        args = [saved_config_file],
        conda_env="dcc_env_shared",
        hardware_requirements = dict(
            time_limit = 5, #2
            cores=6, #4
            gpus=1,
            mem_per_core=4,
            )
        )



#%% Cascade x [past, future]

# Cell class: in training data path, and name of class in dcc.simulations
cell_class = 'CcaSR_Cascade'
training_dir_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023-03-23*/training*")
training_file_list = [train_dir.split('/')[-1] for train_dir in training_dir_list]

horizon_list = [12, 24, 48]

for horizon in horizon_list:
    
    for training_dir in training_dir_list:
        
        training_file = training_dir.split('/')[-1]
        datasets_folder = '/'.join(training_dir.split('/')[:-1])
        
        # print(datasets_folder)
        # print(training_file)
                        
        # Fields to change in config.py:
        # Training epochs, locations of training and evaluation datsets,
        # which cell class to use for simulations
        config = dict(
            training_parameters = dict(
                epochs = 200, # 200 is typically enough
                ),
            datasets_folder = datasets_folder, # Point to generated sets folder
            training_sets = (training_file,), # Training subfolder(s)
            eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
            features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
            cell_class = cell_class, # Cell class to use in dcc.simulations
            models_folder = dcc_repo_path + '/assets/models/',
            horizon = horizon,
            )
        
        # Updated config and save it to disk:
        saved_config_file = params_change(config)
        
        # Submit qsub request for single job:
        job_id = qsub.submit(
            dcc_repo_path + "scripts/simulated_pipeline.py",
            args = [saved_config_file],
            conda_env="dcc_env_shared",
            hardware_requirements = dict(
                time_limit = 5, #2
                cores=6, #4
                gpus=1,
                mem_per_core=4,
                )
            )
        
past_steps_list = [6, 12, 24]

for past_steps in past_steps_list:
    
    for training_dir in training_dir_list:
        
        training_file = training_dir.split('/')[-1]
        datasets_folder = '/'.join(training_dir.split('/')[:-1])
        
        # print(datasets_folder)
        # print(training_file)
                        
        # Fields to change in config.py:
        # Training epochs, locations of training and evaluation datsets,
        # which cell class to use for simulations
        config = dict(
            training_parameters = dict(
                epochs = 200, # 200 is typically enough
                ),
            datasets_folder = datasets_folder, # Point to generated sets folder
            training_sets = (training_file,), # Training subfolder(s)
            eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
            features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
            cell_class = cell_class, # Cell class to use in dcc.simulations
            models_folder = dcc_repo_path + '/assets/models/',
            past_steps = past_steps,
            )
        
        # Updated config and save it to disk:
        saved_config_file = params_change(config)
        
        # Submit qsub request for single job:
        job_id = qsub.submit(
            dcc_repo_path + "scripts/simulated_pipeline.py",
            args = [saved_config_file],
            conda_env="dcc_env_shared",
            hardware_requirements = dict(
                time_limit = 5, #2
                cores=6, #4
                gpus=1,
                mem_per_core=4,
                )
            )
        
#%% Feedforward Positive x [past, future]

# Cell class: in training data path, and name of class in dcc.simulations
cell_class = 'CcaSR_FeedforwardPositive'

horizon_list = [12, 48]

for horizon in horizon_list:
        
    training_dir_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023*/training*")
    training_file_list = [train_dir.split('/')[-1] for train_dir in training_dir_list]
    
    for training_dir in training_dir_list:
        
        training_file = training_dir.split('/')[-1]
        datasets_folder = '/'.join(training_dir.split('/')[:-1])
        
        print(datasets_folder)
        print(training_file)
                        
        # Fields to change in config.py:
        # Training epochs, locations of training and evaluation datsets,
        # which cell class to use for simulations
        config = dict(
            training_parameters = dict(
                epochs = 200, # 200 is typically enough
                ),
            datasets_folder = datasets_folder, # Point to generated sets folder
            training_sets = (training_file,), # Training subfolder(s)
            eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
            features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
            cell_class = cell_class, # Cell class to use in dcc.simulations
            models_folder = dcc_repo_path + '/assets/models/',
            horizon = horizon,
            )
        
        # Updated config and save it to disk:
        saved_config_file = params_change(config)
        
        # Submit qsub request for single job:
        job_id = qsub.submit(
            dcc_repo_path + "scripts/simulated_pipeline.py",
            args = [saved_config_file],
            conda_env="dcc_env_shared",
            hardware_requirements = dict(
                time_limit = 5, #2
                cores=6, #4
                gpus=1,
                mem_per_core=4,
                )
            )
        
past_steps_list = [12, 24, 36]

for past_steps in past_steps_list:
        
    training_dir_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023*/training*")
    training_file_list = [train_dir.split('/')[-1] for train_dir in training_dir_list]
    
    for training_dir in training_dir_list:
        
        training_file = training_dir.split('/')[-1]
        datasets_folder = '/'.join(training_dir.split('/')[:-1])
                        
        # Fields to change in config.py:
        # Training epochs, locations of training and evaluation datsets,
        # which cell class to use for simulations
        config = dict(
            training_parameters = dict(
                epochs = 200, # 200 is typically enough
                ),
            datasets_folder = datasets_folder, # Point to generated sets folder
            training_sets = (training_file,), # Training subfolder(s)
            eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
            features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
            cell_class = cell_class, # Cell class to use in dcc.simulations
            models_folder = dcc_repo_path + '/assets/models/',
            past_steps = past_steps,
            )
        
        # Updated config and save it to disk:
        saved_config_file = params_change(config)
        
        # Submit qsub request for single job:
        job_id = qsub.submit(
            dcc_repo_path + "scripts/simulated_pipeline.py",
            args = [saved_config_file],
            conda_env="dcc_env_shared",
            hardware_requirements = dict(
                time_limit = 5, #2
                cores=6, #4
                gpus=1,
                mem_per_core=4,
                )
            )
        
#%% Launch several single trainings (change horizon, past steps, or training set size):
    
# Cell class: in training data path, and name of class in dcc.simulations
cell_class_list = ['CcaSR_gillespie_simple_noE',
                   'CcaSR_gillespie']

# horizon_list = [12, 24, 48]
# past_steps_list = [12, 24, 36]

for cell_class in cell_class_list:
        
    training_dir_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023-03-09*/training*")
    training_file_list = [train_dir.split('/')[-1] for train_dir in training_dir_list]
    
    for training_dir in training_dir_list:
        
        training_file = training_dir.split('/')[-1]
        datasets_folder = '/'.join(training_dir.split('/')[:-1])
                
        datasets_folder = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023*")[0]

        # Fields to change in config.py:
        # Training epochs, locations of training and evaluation datsets,
        # which cell class to use for simulations
        config = dict(
            training_parameters = dict(
                epochs = 200, # 200 is typically enough
                ),
            datasets_folder = datasets_folder, # Point to generated sets folder
            training_sets = (training_file,), # Training subfolder(s)
            eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
            features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
            cell_class = cell_class, # Cell class to use in dcc.simulations
            models_folder = dcc_repo_path + '/assets/models/',
            )
        
        # Updated config and save it to disk:
        saved_config_file = params_change(config)
        
        # Submit qsub request for single job:
        job_id = qsub.submit(
            dcc_repo_path + "scripts/simulated_pipeline.py",
            args = [saved_config_file],
            conda_env="delta_env",
            hardware_requirements = dict(
                time_limit = 3, #2
                cores=4, #4
                gpus=1,
                mem_per_core=4,
                )
            )



#%% Launch trainings on multiple different datsets

# Cell class: in training data path, and name of class in dcc.simulations
cell_class = 'CcaSR_Cascade'

datasets_folder_list = glob.glob(dcc_data_path + f"assets/simulated/data/{cell_class}/2023-03-1*")

for datasets_folder in datasets_folder_list:    

    # Fields to change in config.py:
    # Training epochs, locations of training and evaluation datsets,
    # which cell class to use for simulations
    config = dict(
        training_parameters = dict(
            epochs = 200, # 200 is typically enough
            ),
        datasets_folder = datasets_folder+'/', # Point to generated sets folder
        training_sets = ("training_set",), # Training subfolder(s)
        eval_sets = ("evaluation_set",), # Evaluation subfolder(s)
        features = ("fluo1", "stims"), # Features to use (probably will only be fluo1 and stims)
        cell_class = cell_class, # Cell class to use in dcc.simulations
        models_folder = dcc_repo_path + '/assets/models/',
        )
    
    # Updated config and save it to disk:
    saved_config_file = params_change(config)
    
    # Submit qsub request for single job:
    job_id = qsub.submit(
        dcc_repo_path + "scripts/simulated_pipeline.py",
        args = [saved_config_file],
        conda_env="delta_env",
        hardware_requirements = dict(
            time_limit = 5, #2
            cores=6, #4
            gpus=1,
            mem_per_core=4,
            )
        )

#%% Launch job array:

# Base config to update:
base_config = dict(
    training_parameters = dict(
        epochs = 200,
        ),
    datasets_folder = dcc_data_path + "assets/simulated/data/CcaSR_gillespie/",
    training_sets = ("training_set",),
    features = ("fluo1", "stims"),
    eval_sets = ("evaluation_set",),
    cell_class = "CcaSR_gillespie"
    )

# List of different configs for different jobs
configs = []


new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 36}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 48}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 12}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"horizon": 6}) # for example
configs.append(params_change(new_config))

new_config = copy.deepcopy(base_config)
new_config.update({"mlp_layers": 20, "mlp_dim": 256}) # for example
configs.append(params_change(new_config))

# Submit job array:
job_id = qsub.submit(
    dcc_repo_path + "scripts/training.py",
    job_array = True,
    args = configs,
    kwargs = [{}] * len(configs),
    conda_env="delta_env",
    hardware_requirements = dict(
        time_limit = 12,
        cores=4,
        gpus=1,
        mem_per_core=4,
        )
    )
