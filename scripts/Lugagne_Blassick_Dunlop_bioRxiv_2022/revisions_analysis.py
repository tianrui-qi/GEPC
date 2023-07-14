# -*- coding: utf-8 -*-
"""
Script to analyze the outcome of various re-trainings of the networks
Fig S1, S2, S5 and S6 plotted here
You will need scikit-learn (v1.2.2 at time of this writing)

Created on Sat Apr 15 16:01:50 2023

@author: jeanbaptiste
"""
import copy
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import deepcellcontrol as dcc

# Datasets (zenodo archive)
datasets_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/datasets/"

# Trained models (zenodo archive):
models_folder = "Z:/data/Microscope/Papers/Lugagne_Blassick_Dunlop_NatComm_2023/models/"

# Save images to:
save_folder = "D:/papers/deepmpc/revisions/"

def predict(model, inputs):
    
    step_size = 20_000
    predictions = []
    for i in range(0, inputs[0].shape[0], step_size):
        j = min(i+step_size, inputs[0].shape[0])
        predictions += [model.predict(
                [inputs[0][i:j,:], inputs[1][i:j,:]],
                verbose = 1,
                batch_size=1000,
            )]
    predictions = np.concatenate(predictions, axis=0)
    
    return predictions

#%% Load saved eval data

features = (
    'fluo1',
    'area',
    'sharpness',
    'cell_count',
    'chamber_mean_fluo1',
    'chamber_std_fluo1',
    'neighbor_stims',
    'stims'
    )

import pickle
# This data is from ode_mpc.py, we re-use the same evaluation set to keep
# results comparable. The exact file we used is on zenodo, but it can be 
# randomly generated via ode_mpc.py too.
with open(save_folder + "/ODE_final_eval_inputs.pkl", "rb") as f:
    data = pickle.load(f)
    eval_inputs = data["inputs"]
    groundtruth = data["groundtruth"]

# Normalize inputs:
fake_dataset = {}
for f, feature in enumerate(features):
    fake_dataset[feature] = eval_inputs[0][:,:,f].copy()
fake_dataset = dcc.data.Normalization().normalize(fake_dataset)
norm_inputs = np.empty_like(eval_inputs[0])
for f, feature in enumerate(features):
    norm_inputs[:,:,f] = fake_dataset[feature]
eval_inputs[0] = norm_inputs

#%% Fig S1 - Evaluate Increasing networtk size

folders = {
    "replicate 1": (
        "2023-04-16_10-24-42_2e3c0f37-75a5-44b4-8b96-a722ffb42f7c/",
        "2023-04-16_10-24-42_77b3c0f3-bca6-42a3-8d9e-d4b32f0cbe5a/",
        "2023-04-16_10-24-42_ddb2e7e9-7afa-4a21-a44b-5fa58b391f3c/",
        "2023-04-16_10-24-42_1ff25381-1d24-4260-add2-3f67e8d7f820/",
        "2023-04-16_10-24-42_e9473569-e19c-4666-ac8a-c8e02851c9ab/",
        "2023-04-16_15-58-35_60fcc1b6-e07a-4a77-b54e-11c90ca1d43f/",
        ),
    "replicate 2": (
        "2023-04-16_19-25-16_145e84ea-9154-4d22-a28d-803e0720f5b0/",
        "2023-04-16_19-25-16_82e761cd-1ea2-4710-aca8-bcf14527fbb3/",
        "2023-04-16_19-25-16_0be46940-796b-4ec0-9b64-2680747296b3/",
        "2023-04-16_19-25-16_a75d89e5-f715-4b3a-ae36-c602fb8dede0/",
        "2023-04-16_19-25-16_d80489b8-9c49-45f3-8406-7761a262efa9/",
        "2023-04-16_19-25-16_5bca912c-ff2c-4047-9a30-3f9325226800/",
        ),
    "replicate 3": (
        "2023-04-16_19-25-33_024aa672-c994-4c75-a24e-e38bf50eee54/",
        "2023-04-16_19-25-33_d9ae792f-949c-4afe-8ea8-334217a78e52/",
        "2023-04-16_19-25-33_b693cc6e-dbf7-42cb-a3ef-a09f90b8840b/",
        "2023-04-16_19-25-33_530e4d81-580b-4eb3-b5d6-031633ccdd39/",
        "2023-04-16_19-25-33_5f26c93a-6197-48f6-a9d9-c1fd9ca0a2b1/",
        "2023-04-16_19-25-33_6fed7ab9-8bf3-4a9f-8bed-5ebe510bb26c/",
        ),
    }

rmse_record = {}
for replicate, folders_list in folders.items():
    
    for f, folder in enumerate(folders_list):
        
        # Run prediction:
        model = tf.keras.models.load_model(
            models_folder + folder + "/model_besteval.hdf5"
            )
        nb_params = model.count_params()
        predictions = predict(model, eval_inputs)
        
        # Compute RMSE:
        rmse = np.sqrt(
            np.mean(((4095*predictions-groundtruth))**2,axis=1)
            )
        if nb_params not in rmse_record:
            rmse_record[nb_params] = [rmse]
        else:
            rmse_record[nb_params].append(rmse)
        
        print(f"{replicate}, {nb_params} params. RMSE - mean: {np.mean(rmse)}, median: {np.median(rmse)}")

#%% Fig S1 - Plot increasing network size

plt.figure(1, dpi=300)
plt.figure(2, dpi=300)

# Log-spaced bins:
nbins = 100
bins = np.logspace(0, 4, nbins + 1, base=10)

f = 0
for name, rmses in rmse_record.items():
    
    rmse_cat = np.concatenate(rmses)
    # Plot hist:
    plt.figure(1)
    n, _, _ = plt.hist(rmse_cat, bins=bins, histtype="step", label=name)
    
    plt.figure(2)
    avg_rmse = 0
    for ind, rmse in enumerate(rmses):
        plt.plot(f+ind*.2-.2, np.median(rmse), "ko", fillstyle="none")
        avg_rmse += np.median(rmse)/3
    plt.bar(f,avg_rmse, label=name)
    
    f+=1

plt.figure(1)
plt.xscale("log")
plt.xlim([5, 5000])
plt.xlabel("Root mean square error (a.u.)")
plt.ylabel("Count")
plt.grid(axis="x", which="major")
plt.legend(title="Parameters")
plt.savefig(save_folder+"IncreasingNetworks_100_distros.png", dpi=300)
plt.savefig(save_folder+"IncreasingNetworks_100_distros.svg", dpi=300)
plt.savefig(save_folder+"IncreasingNetworks_100_distros.pdf", dpi=300)

plt.figure(2)
plt.ylabel("Root mean square error (a.u.)")
plt.xticks(range(len(rmse_record)), rmse_record.keys())
plt.title("median")
plt.savefig(save_folder+"IncreasingNetworks_100_bars.png", dpi=300)
plt.savefig(save_folder+"IncreasingNetworks_100_bars.svg", dpi=300)
plt.savefig(save_folder+"IncreasingNetworks_100_bars.pdf", dpi=300)


#%% Fig S1 - Evaluate time

folders_list = folders["replicate 1"]
timing = {}
for f, folder in enumerate(folders_list):
    
    # Run prediction:
    model = tf.keras.models.load_model(
        models_folder + folder + "/model_besteval.hdf5"
        )
    nb_params = model.count_params()
    encoder, decoder = dcc.models.split(model)
    _t = time.time()
    latent = encoder.predict(eval_inputs[0], verbose=1, batch_size=1000)
    encoder_timing = time.time() - _t
    _t = time.time()
    predictions = decoder.predict([latent, eval_inputs[1]], verbose=1, batch_size=1000)
    decoder_timing = time.time() - _t
    
    timing[nb_params] = [encoder_timing/100_000, decoder_timing/100_000]

    print(timing)
    time.sleep(2*60) # Give time to GPU to cool down

#%% Fig S1 - Plot time

plt.figure(dpi=300)
r = 0
ticks = {'ticks': [], 'labels': []}
for nbparams, record in timing.items():
    ticks['ticks'].append(r)
    ticks['labels'].append(nbparams)
    plt.bar(r-.2, record[0], color="xkcd:light purple", width=.4, edgecolor="k")
    plt.bar(r+.2, record[1], color="purple", width=.4, edgecolor="k")
    r+=1
    
plt.yscale("log")
plt.ylabel("Computation time (s)")
plt.xticks(**ticks)
# figname = f"Models_comparison_timing"
plt.savefig(save_folder+"IncreasingNetworks_timing.png", dpi=300)
plt.savefig(save_folder+"IncreasingNetworks_timing.svg", dpi=300)
plt.savefig(save_folder+"IncreasingNetworks_timing.pdf", dpi=300)
plt.show()

#%% Fig S5 - Evaluate Leave one out features

folders = {
    "replicate 1": (
        "2023-04-16_13-04-16_aab6c0ff-bbdd-4c11-baad-8b1eaa73c971/",
        "2023-04-16_13-04-16_7b0b67ad-67ba-492c-a8f0-fb13188ad9ff/",
        "2023-04-16_13-04-16_36472c95-59b4-4708-ab62-4318ac0e7f64/",
        "2023-04-16_13-04-16_df4a68a6-a38d-4e14-8fff-08489c63b851/",
        "2023-04-16_13-04-16_8e142a50-68bd-4b62-ae3a-3e3db4a34665/",
        "2023-04-16_13-04-16_8970a939-46f5-426d-a3e8-0e426f4f83d1/",
        "2023-04-16_13-04-16_946e3250-e8c8-4681-9e0f-09a32a9d3623/",
        "2023-04-16_13-04-16_bbc14bf3-0762-49f8-8327-60fd3e28720d/",
        "2023-04-17_10-29-30_f3b85ec3-880d-4bb6-b522-e12b2c968866/",
        "2023-04-17_16-31-25_b021ce13-5ff1-4d8e-95e4-4ac3b022634f/",
        ),
    "replicate 2": (
        "2023-04-16_18-03-45_9d02cddd-b5b2-479f-8cea-d82226fe2290/",
        "2023-04-16_18-03-45_34fd8b6c-b76c-47a4-9611-c55a377e5826/",
        "2023-04-16_18-03-45_6c6b55b7-9eec-411b-81f9-5e176db0b1cc/",
        "2023-04-16_18-03-45_32f5268d-a259-47d1-9832-ab05c929b7fd/",
        "2023-04-16_18-03-45_f57d6c0b-5d98-423a-b97f-ad31cf37e44a/",
        "2023-04-16_18-03-45_85e93ec1-8749-40d9-9c5f-f0ada69f31c7/",
        "2023-04-16_18-03-45_c856fd5b-796b-4e28-aee3-f753d87a086a/",
        "2023-04-16_18-03-45_69afddb1-9c0e-4a5c-a9d6-6f9f1c9d63e8/",
        "2023-04-17_10-31-57_ccecf385-b8cb-42a8-a3dd-11288227a658/",
        "2023-04-17_16-31-25_666182c2-e1b7-4b91-aeb6-112bfea9d0c3/",
        ),
    "replicate 3": (
        "2023-04-16_18-13-13_66a428d9-99d8-4262-9a99-f4c2687a2441/",
        "2023-04-16_18-13-13_3b4df3d7-2913-4a6c-8928-2dc89a4ee63d/",
        "2023-04-16_18-13-13_346b1a00-cb16-4e65-a3ac-c8a7b40f22ab/",
        "2023-04-16_18-13-13_f796ae85-6b90-4011-a315-bfdb728f9181/",
        "2023-04-16_18-13-13_baaadbe0-c5e2-4b97-a9bd-009ffa0f88a0/",
        "2023-04-16_18-13-13_26478073-c266-4db5-a7b9-9da83cfd27da/",
        "2023-04-16_18-13-13_7e899838-6930-4be1-9388-491ba1f29334/",
        "2023-04-16_18-13-13_6f3a4e58-a839-48da-8dde-908e7921c2d2/",
        "2023-04-17_10-33-51_f913f9b4-9e50-4108-aa14-6d9d071e3626/",
        "2023-04-17_16-31-25_46442dac-09b1-4e4b-bda0-f442aa79e4d9/",
        ),
    }
    
all_features = copy.copy(dcc.config.defaults["features"])
rmse_record = {}
for replicate, folders_list in folders.items():
    
    for f, folder in enumerate(folders_list):
        
        # Figure out which feature is missing:
        with open(models_folder + folder + "submission_parameters.json","r") as file:
            features = json.load(file)["features"]
        missing = list(set(all_features) - set(features))
        if len(missing)==0:
            name = "All"
        else:
            name = "\n".join(missing)
        if name not in rmse_record:
            rmse_record[name] = []
        
        # Remove from eval inputs:
        keep = [x for x, fname in enumerate(all_features) if fname not in missing]
        masked_past, light = eval_inputs
        masked_past = masked_past[:,:,keep]
        
        # Run prediction:
        model = tf.keras.models.load_model(
            models_folder + folder + "model_besteval.hdf5"
            )
        predictions = predict(model, [masked_past, light])
        
        # Compute RMSE:
        rmse = np.sqrt(
            np.mean(((4095*predictions-groundtruth))**2,axis=1)
            )
        rmse_record[name].append(rmse)
        
        print(f"{replicate}, {missing}. RMSE - mean: {np.mean(rmse)}, median: {np.median(rmse)}")

# No-past MLP:
past, light = eval_inputs
model = tf.keras.models.load_model(
    models_folder + "2023-06-29_14-50-42_db965cf9-0460-4a83-9b0a-d2a2ec99dd97/model_besteval.hdf5"
    )
predictions = predict(model, [past, light])
nopast_rmse = np.sqrt(
    np.mean(((4095*predictions-groundtruth))**2,axis=1)
    )


#%% Fig S5 - Plot leave one out features
plt.figure(1, dpi=300)
plt.figure(2, dpi=300)
# plt.figure(3, dpi=300)

# Log-spaced bins:
nbins = 100
bins = np.logspace(0, 4, nbins + 1, base=10)

# Order:
ordered_list = []
for name, rmses in rmse_record.items():
    replicates_mean = 0
    for ind, rmse in enumerate(rmses):
        replicates_mean += np.median(rmse)/3
    ordered_list.append((name, replicates_mean, rmses))
ordered_list.sort(key=lambda x: x[1], reverse=True)

f = 0
for name, avg_rmse, rmses in ordered_list:
    
    rmse_cat = np.concatenate(rmses)
    # Plot hist:
    plt.figure(1)
    n, _, _ = plt.hist(rmse_cat, bins=bins, histtype="step", label=name)
    
    plt.figure(2)
    for ind, rmse in enumerate(rmses):
        plt.plot(f+ind*.2-.2, np.median(rmse), "ko", fillstyle="none")
    plt.bar(f,avg_rmse, label=name)
    
    f+=1

plt.figure(1)
plt.xscale("log")
plt.xlim([5, 5000])
plt.xlabel("Root mean square error (a.u.)")
plt.ylabel("Count")
plt.grid(axis="x", which="major")
plt.legend(title="Missing")
plt.savefig(save_folder+"Features_distros.png", dpi=300)
plt.savefig(save_folder+"Features_distros.svg", dpi=300)
plt.savefig(save_folder+"Features_distros.pdf", dpi=300)

plt.figure(2)
plt.ylabel("Root mean square error (a.u.)")
plt.xticks(range(len(ordered_list)), [x[0] for x in ordered_list], rotation=45, ha="right")
plt.title("median")
plt.savefig(save_folder+"Features_bars.png", dpi=300)
plt.savefig(save_folder+"Features_bars.svg", dpi=300)
plt.savefig(save_folder+"Features_bars.pdf", dpi=300)

#%% Fig S6 - Load and split encoder in all-features network:

folder = "2023-04-17_10-29-30_f3b85ec3-880d-4bb6-b522-e12b2c968866/"
model = tf.keras.models.load_model(
    models_folder + folder + "/model_besteval.hdf5"
    )
encoder, _ = dcc.models.split(model)

past = np.load(save_folder+"tsne_past.npy")
past_nonan = past.copy()
past_nonan[np.isnan(past_nonan)] = 0
encoded_past = encoder.predict(past_nonan, verbose=1, batch_size=1000)
encoded_past = np.concatenate(encoded_past, axis = -1)

#%% Fig S6 - Configure and fit t-SNE function. 
tsne = TSNE(
    n_components=2, # default=2, Dimension of the embedded space.
    perplexity=30, # default=30.0, The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    early_exaggeration=12, # default=12.0, Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. 
    learning_rate=10, # default=200.0, The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
    n_iter=1000, # default=1000, Maximum number of iterations for the optimization. Should be at least 250.
    n_iter_without_progress=300, # default=300, Maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration. 
    min_grad_norm=0.0000001, # default=1e-7, If the gradient norm is below this threshold, the optimization will be stopped.
    metric='euclidean', # default=’euclidean’, The metric to use when calculating distance between instances in a feature array.
    init='random', # {‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), default=’random’. Initialization of embedding
    verbose=2, # default=0, Verbosity level.
    random_state=42, # RandomState instance or None, default=None. Determines the random number generator. Pass an int for reproducible results across multiple function calls.
    method='barnes_hut', # default=’barnes_hut’. By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. 
    angle=0.5, # default=0.5, Only used if method=’barnes_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
    n_jobs=-1, # default=None, The number of parallel jobs to run for neighbors search. -1 means using all processors. 
)

# Transform X
tsne_embed = tsne.fit_transform(encoded_past[:])

##%%
plt.scatter(tsne_embed[:,0], tsne_embed[:,1], s=.1, alpha=1)
plt.grid("both", "both")

# np.save(save_folder+"tsne_past.npy", past)
# np.save(save_folder+"tsne_embedded.npy", tsne_embed)

#%% Fig S6 - Fit PCA

pca = PCA(n_components=2, random_state=1)
pca_embed = pca.fit_transform(encoded_past)


#%% Fig S6 - Plot t-SNE & PCA + samples

tsne_embed = np.load(save_folder+"tsne_embedded.npy")
all_features = copy.copy(dcc.config.defaults["features"])

def closest(x, y, embedded, number = 10):
    
    dist = np.sqrt((embedded[:,0] - x)**2 + (embedded[:,1] - y)**2)
    idx = np.argsort(dist)[:number]
    
    return idx



def plot_sample(sample, features = all_features):
    
    data = {}
    for f_ind, feature in enumerate(features):
        data[feature]  = sample[:,f_ind]
    first_index = np.argmax(data["sharpness"]>0)
    
    x = np.arange(-sample.shape[0], 0) / 12
    
    nanfeat = (
        'fluo1', 'area', 'cell_count', 'chamber_mean_fluo1', 'chamber_std_fluo1',
        )
    for feature in nanfeat:
        data[feature][data[feature]==0] = np.nan
    
    # Fluo & stims plot:
    plt.subplot(2,1,1)
    
    # Plot stimulations:
    dcc.utilities.OptoPlotBackground(
        data["stims"],
        x = x,
        ymin = 0,
        ymax = 1,
        )
    
    # Plot fluorescence:
    plt.plot(x,data["fluo1"],"k",label="Mother")
    plt.ylabel("Fluorescence")
    plt.xlim([x[first_index], x[-1]])
    plt.ylim(0, 1)
    
    # Plot area:
    plt.subplot(2,1,2)
    plt.plot(x,data["area"],"k",label="area", zorder = 10)
    plt.ylabel('Area')
    plt.xlim([x[first_index], x[-1]])
    plt.grid(which="both", axis="both")
    plt.ylim(0, 1)


points = [
    {"letter":"C", "xy": (-60, 25), "name": "Empty chambers"},
    {"letter":"D", "xy": (-65, -10), "name": "Segmentation false-positives"},
    {"letter":"E", "xy": (-45, -43), "name": "Non-responsive"},
    {"letter":"F", "xy": (-30, -10), "name": "Noisy data"},
    {"letter":"G", "xy": (-44, 30), "name": "Filamenting"},
    {"letter":"H", "xy": (0, -50), "name": "Low response"},
    {"letter":"I", "xy": (0, 0), "name": "Medium response"},
    {"letter":"J", "xy": (-10, 45), "name": "High response"},
    ]

plt.figure(len(points), figsize=(10, 3.5), dpi=300)
plt.subplot(1,2,1)
plt.scatter(tsne_embed[:,0], tsne_embed[:,1], s=.1, alpha=.1, color="gray")
plt.title("t-SNE embedding")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.subplot(1,2,2)
plt.scatter(pca_embed[:,0], pca_embed[:,1], s=.1, alpha=.025, color="gray")
plt.title("PCA embedding")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")

for p, point in enumerate(points):
    x, y = point["xy"]
    idx = closest(x,y, tsne_embed, 1)[0]

    # plt.figure(p, dpi=300)
    # plot_sample(past[idx])
    # plt.subplot(2,1,1)
    # plt.title(point["name"])
    # plt.savefig(save_folder+f"Embeddings_panel{point['letter']}.png", dpi=300)
    # plt.savefig(save_folder+f"Embeddings_panel{point['letter']}.svg", dpi=300)
    # plt.savefig(save_folder+f"Embeddings_panel{point['letter']}.pdf", dpi=300)
    
    plt.figure(len(points))
    x, y = tsne_embed[idx]
    plt.subplot(1,2,1)
    plt.scatter(x, y, s=4, color="k")
    plt.annotate(
        point["letter"], 
        (x, y), 
        xytext = (.5,3.5),
        color="k", 
        textcoords = 'offset points',
        fontsize = 14,
        font = "Consolas",
        )
    
    if p == 3:
        xytext = (-5,3.5)
    else:
        xytext = (.5,3.5)
    x, y = pca_embed[idx]
    plt.subplot(1,2,2)
    plt.scatter(x, y, s=4, color="k")
    plt.annotate(
        point["letter"], 
        (x, y), 
        xytext = xytext, 
        color="k", 
        textcoords = 'offset points',
        fontsize = 14,
        font = "Consolas",
        )

plt.figure(len(points))
plt.savefig(save_folder+"Embeddings_tsne_pca.png", dpi=300)
plt.savefig(save_folder+"Embeddings_tsne_pca.svg", dpi=300)
plt.savefig(save_folder+"Embeddings_tsne_pca.pdf", dpi=300)

#%% Fig S2 - Load evaluation dataset for datasets shuffle and generate eval data:

files = (
 'Z:/data/Microscope/jeanbaptiste/deepmpc/trainingsets/2022-04-24_TrainingSet8/deepcellcontrol_dataset/2022-04-24_TrainingSet8_dataset.pkl',
 )
params = dcc.config.defaults
test_datasets = dcc.data.Datasets(
    files,
    formatter=dcc.data.LSTMFormatter,
    parameters=params
    )
test_datasets.test_ratio = 1
test_datasets.mode = "evaluation"
test_datasets.data_type='raw_dataset'
test_datasets.load()
test_datasets.normalize()
test_datasets.data_type='normalized_dataset'

# Get eval data:
test_datasets.batch_size = 200_000
eval_inputs, groundtruth = next(test_datasets)
groundtruth = groundtruth[:,:,0]

np.save(save_folder+"setstest_past.npy", eval_inputs[0])
np.save(save_folder+"setstest_stims.npy", eval_inputs[1])
np.save(save_folder+"setstest_groundtruth.npy", groundtruth)

#%% Fig S2 - Evaluate datasets shuffle

eval_inputs = (
    np.load(save_folder+"setstest_past.npy"),
    np.load(save_folder+"setstest_stims.npy")
    )
groundtruth = np.load(save_folder+"setstest_groundtruth.npy")

folders = (
    '2023-04-23_23-20-59_0d071bfe-ea37-4aa1-8250-ee52f06cb956',
    '2023-04-23_23-20-59_3bae1d7a-e1fe-43bc-b1d4-a123252ed056',
    '2023-04-23_23-20-59_4dd4f7ad-8013-4683-b507-30923da4d4e9',
    '2023-04-23_23-20-59_92ba8625-f873-4b60-925d-f66bbbd1b687',
    '2023-04-23_23-20-59_497c66d5-f15f-42cd-8d7c-ef43d328719a',
    '2023-04-23_23-20-59_0577bf50-fdef-4b7e-b52e-ab14ce94a533',
    '2023-04-23_23-20-59_4175ad1b-1b72-4ca6-aced-edb512f37ca7',
    '2023-04-23_23-20-59_49859dc4-97da-402c-a1ea-e292ebba41de',
    '2023-04-23_23-20-59_78119a0e-b5fc-4069-a9bf-f978a5e076b4',
    '2023-04-23_23-20-59_a916d769-9cca-49d0-9066-d227b33bfccb',
    '2023-04-23_23-20-59_b5e8aca9-2ac3-40f5-83b2-0d162312a00d',
    '2023-04-23_23-20-59_b97938cc-0b7b-4f29-89f3-652aeaef0a6f',
    '2023-04-23_23-20-59_c22d6713-d03d-4ea4-af82-721d92c44d86',
    '2023-04-23_23-20-59_c23ee585-2415-441a-a0c2-06a06356d01d',
    '2023-04-23_23-20-59_c800658a-de7b-4dd5-bfc3-f85fe1f5e321',
    '2023-04-25_16-07-13_b58f7b2c-502f-4167-874b-2c6d9f04d5e9',
    '2023-04-25_16-07-13_94539583-fd41-4d68-80f8-f70231a3d1c9',
    '2023-04-25_16-07-13_89c782fc-092d-46b9-87b8-49c3d0d6231d'
     )

rmse_records = []
for folder in folders:
    
    with open(models_folder+folder+"/submission_parameters.json", "r") as f:
        record = json.load(f)
    record["rmse"] = []
    
    for oldrecord in rmse_records:
        if oldrecord["training_sets"] == record["training_sets"]:
            record = oldrecord
            break
    
    print(record["training_sets"], end="\n\n")
    
    model = tf.keras.models.load_model(
        models_folder + folder + "/model.hdf5"
        )
    predictions = predict(model, eval_inputs)
    
    # Compute RMSE:
    rmse = np.sqrt(
        np.mean((4095*(predictions-groundtruth))**2,axis=1)
        )
    record["rmse"] += [rmse]
    
    if record not in rmse_records:
        rmse_records.append(record)

#%% Fig S2 - Plot dataset shuffle

plt.figure(1, dpi=300)
plt.figure(2, dpi=300)

# Log-spaced bins:
nbins = 100
bins = np.logspace(0, 4, nbins + 1, base=10)

f = 0
names = []
order = dcc.config.defaults["training_sets"] + dcc.config.defaults["eval_sets"]
for record in rmse_records:
    
    name = [str(order.index(x)+1) for x in record["training_sets"]]
    name.sort()
    name = " + ".join(name)
    names.append(name)

rmse_records = [rmse_records[i] for i in np.argsort(names)]
names.sort()
for record, name in zip(rmse_records, names):
    rmses = record["rmse"]
    rmse_cat = np.concatenate(rmses)
    # Plot hist:
    plt.figure(1)
    n, _, _ = plt.hist(rmse_cat, bins=bins, histtype="step", label=name)
    
    plt.figure(2)
    avg_rmse = 0
    for ind, rmse in enumerate(rmses):
        plt.plot(f+ind*.2-.2, np.median(rmse), "ko", fillstyle="none")
        avg_rmse += np.median(rmse)/len(rmses)
    plt.bar(f,avg_rmse, label=name)
    
    f+=1

plt.figure(1)
plt.xscale("log")
plt.xlim([5, 5000])
plt.xlabel("Root mean square error (a.u.)")
plt.ylabel("Count")
plt.grid(axis="x", which="major")
plt.legend(title="Datasets used for training")
plt.savefig(save_folder+"ShuffledSets_distros.png", dpi=300)
plt.savefig(save_folder+"ShuffledSets_distros.svg", dpi=300)
plt.savefig(save_folder+"ShuffledSets_distros.pdf", dpi=300)
# plt.show()

plt.figure(2)
plt.ylabel("Root mean square error (a.u.)")
plt.xticks(range(len(rmse_records)), names, rotation=45, ha="right")
plt.title("median")
plt.xlabel("Datasets used for training")
# plt.ylim(160,240)
plt.savefig(save_folder+"ShuffledSets_bars.png", dpi=300)
plt.savefig(save_folder+"ShuffledSets_bars.svg", dpi=300)
plt.savefig(save_folder+"ShuffledSets_bars.pdf", dpi=300)
# plt.show()
