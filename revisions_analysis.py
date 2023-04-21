# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:01:50 2023

@author: jeanbaptiste
"""
import copy
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from sklearn.manifold import TSNE # for t-SNE dimensionality reduction
from sklearn.decomposition import PCA


import deepcellcontrol as dcc

datasets_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/data/"
models_folder = "Y:/projectnb2/dunlop/JB/deepcellcontrol/assets/models/"
figures_folder = "D:/deepmpc_paper/revisions/"

#%% Load evaluation dataset:
params = copy.deepcopy(dcc.config.defaults)
params["datasets_folder"] = datasets_folder
params["training_sets"] = [] # this way we only load the eval sets

_, evaluation_set = dcc.data.load_datasets(params)

# Get eval data:
evaluation_set.batch_size = 200_000
eval_inputs, groundtruth = next(evaluation_set)
groundtruth = groundtruth[:,:,0]

#%% Increasing networtk size, 1000 batch size

folders_100 = {
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

folders_1000 = {
    "replicate 1": (
        "2023-04-15_22-25-08_3c6a40a3-c0b4-4cd9-8908-1954a9ebbc63/",
        "2023-04-15_22-25-08_023d8358-8646-4d2b-a43b-17d9d51001df/",
        "2023-04-15_22-25-08_0057df48-b773-477d-9621-37bf1b6fe520/",
        "2023-04-15_22-25-08_379fdac0-bf14-40ce-856a-559c8f633a99/",
        "2023-04-16_11-22-00_337a5ab7-33f6-40a8-b9fe-5b4755f5589b/",
        "2023-04-16_11-22-00_81b4eff3-d8f7-435b-bafe-43f77c4bc43d/",
        ),
    "replicate 2": (
        "2023-04-16_19-27-18_c9c944f7-f48c-4283-9da2-98560472eb28/",
        "2023-04-16_19-27-18_064b1c3f-401c-45a5-968c-05247026984f/",
        "2023-04-16_19-27-18_0b75981a-89c7-4549-a6b4-2bb97de9db0f/",
        "2023-04-16_19-27-18_2391fb47-3eb7-480a-a1dd-0b3b41f78160/",
        "2023-04-16_19-27-18_b350b82f-63e1-44aa-903e-075cbe3bdfae/",
        "2023-04-16_19-27-18_18e084e9-4cbb-4e91-9a6c-8468ade62ff3/",
        ),
    "replicate 3": (
        "2023-04-16_19-27-31_be917615-ebb1-4a4a-a9af-a5548804d2bd/",
        "2023-04-16_19-27-31_c6da6b94-29c7-481c-81dd-1e3ead639eb5/",
        "2023-04-16_19-27-31_18a4a8d2-acd9-4710-89c9-51dff96d28fd/",
        "2023-04-16_19-27-31_83fb5d35-fcb9-4846-abde-5d950da764f4/",
        "2023-04-16_19-27-31_f22b7134-9ed6-4b00-afb2-3e8945d384bc/",
        "2023-04-16_19-27-31_adb5eac7-1a85-4558-80c0-73270b27ac24/",
        ),
    }

folders = folders_1000
rmse_record = {}
for replicate, folders_list in folders.items():
    
    for f, folder in enumerate(folders_list):
        
        # Run prediction:
        model = tf.keras.models.load_model(
            models_folder + folder + "/model_besteval.hdf5"
            )
        nb_params = model.count_params()
        predictions = model.predict(eval_inputs, verbose=1)
        
        # Compute RMSE:
        rmse = np.sqrt(
            np.mean((4095*(predictions-groundtruth))**2,axis=1)
            )
        if nb_params not in rmse_record:
            rmse_record[nb_params] = [rmse]
        else:
            rmse_record[nb_params].append(rmse)
        
        print(f"{replicate}, {nb_params} params. RMSE - mean: {np.mean(rmse)}, median: {np.median(rmse)}")

#%%
plt.figure(1, dpi=300)
plt.figure(2, dpi=300)
# plt.figure(3, dpi=300)

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
plt.savefig(figures_folder+"IncreasingNetworks_1000_distros.png", dpi=300)
plt.savefig(figures_folder+"IncreasingNetworks_1000_distros.svg", dpi=300)
plt.savefig(figures_folder+"IncreasingNetworks_1000_distros.pdf", dpi=300)

plt.figure(2)
plt.ylabel("Root mean square error (a.u.)")
plt.xticks(range(len(rmse_record)), rmse_record.keys())
plt.title("median")
# plt.ylim(160,240)
plt.savefig(figures_folder+"IncreasingNetworks_1000_bars.png", dpi=300)
plt.savefig(figures_folder+"IncreasingNetworks_1000_bars.svg", dpi=300)
plt.savefig(figures_folder+"IncreasingNetworks_1000_bars.pdf", dpi=300)

plt.ylim(160,240)
plt.savefig(figures_folder+"IncreasingNetworks_1000_bars_cut.png", dpi=300)
plt.savefig(figures_folder+"IncreasingNetworks_1000_bars_cut.svg", dpi=300)
plt.savefig(figures_folder+"IncreasingNetworks_1000_bars_cut.pdf", dpi=300)


#%% Leave one out features

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
        predictions = model.predict([masked_past, light], verbose=1)
        
        # Compute RMSE:
        rmse = np.sqrt(
            np.mean((4095*(predictions-groundtruth))**2,axis=1)
            )
        rmse_record[name].append(rmse)
        
        print(f"{replicate}, {missing}. RMSE - mean: {np.mean(rmse)}, median: {np.median(rmse)}")

#%%
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
plt.savefig(figures_folder+"Features_distros.png", dpi=300)
plt.savefig(figures_folder+"Features_distros.svg", dpi=300)
plt.savefig(figures_folder+"Features_distros.pdf", dpi=300)

plt.figure(2)
plt.ylabel("Root mean square error (a.u.)")
plt.xticks(range(len(ordered_list)), [x[0] for x in ordered_list])
plt.title("median")
plt.savefig(figures_folder+"Features_bars.png", dpi=300)
plt.savefig(figures_folder+"Features_bars.svg", dpi=300)
plt.savefig(figures_folder+"Features_bars.pdf", dpi=300)

plt.ylim(160,320)
plt.savefig(figures_folder+"Features_bars_cut.png", dpi=300)
plt.savefig(figures_folder+"Features_bars_cut.svg", dpi=300)
plt.savefig(figures_folder+"Features_bars_cut.pdf", dpi=300)

plt.ylim(160,240)
plt.savefig(figures_folder+"Features_bars_cut2.png", dpi=300)
plt.savefig(figures_folder+"Features_bars_cut2.svg", dpi=300)
plt.savefig(figures_folder+"Features_bars_cut2.pdf", dpi=300)
# plt.figure(3)
# plt.ylabel("Root mean square error (a.u.)")
# plt.xticks(list(missings.keys()), list(missings.values()))
# plt.title("median")
#%%

folder = "2023-04-17_10-29-30_f3b85ec3-880d-4bb6-b522-e12b2c968866/"
model = tf.keras.models.load_model(
    models_folder + folder + "/model_besteval.hdf5"
    )
encoder, _ = dcc.models.split(model)
encoded_past = encoder.predict(eval_inputs[0], verbose=1)
encoded_past = np.concatenate(encoded_past, axis = -1)

#%% Configure t-SNE function. 
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


# np.save(figures_folder+"tsne_past.npy", eval_inputs[0])
# np.save(figures_folder+"tsne_embedded.npy", embedded)


#%%

pca = PCA(n_components=2, random_state=1)
pca_embed = pca.fit_transform(encoded_past[:])

plt.scatter(pca_embed[:,0], pca_embed[:,1], s=.1, alpha=1)
plt.grid("both", "both")

#%%

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

#%%

# tsne_embed = np.load(figures_folder+"tsne_embedded.npy")

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

    plt.figure(p, dpi=300)
    past = eval_inputs[0][idx]
    plot_sample(past)
    plt.subplot(2,1,1)
    plt.title(point["name"])
    plt.savefig(figures_folder+f"Embeddings_panel{point['letter']}.png", dpi=300)
    plt.savefig(figures_folder+f"Embeddings_panel{point['letter']}.svg", dpi=300)
    plt.savefig(figures_folder+f"Embeddings_panel{point['letter']}.pdf", dpi=300)
    
    plt.figure(len(points))
    x, y = tsne_embed[idx]
    plt.subplot(1,2,1)
    plt.scatter(x, y, s=4, color="k")
    plt.annotate(
        point["letter"], 
        (x, y), 
        xytext = (0.5,3), 
        color="k", 
        textcoords = 'offset points',
        fontsize = 15,
        )
    
    x, y = pca_embed[idx]
    plt.subplot(1,2,2)
    plt.scatter(x, y, s=4, color="k")
    plt.annotate(
        point["letter"], 
        (x, y), 
        xytext = (0.5,3), 
        color="k", 
        textcoords = 'offset points',
        fontsize = 15,
        )

plt.figure(len(points))
plt.savefig(figures_folder+"Embeddings_tsne_pca.png", dpi=300)
plt.savefig(figures_folder+"Embeddings_tsne_pca.svg", dpi=300)
plt.savefig(figures_folder+"Embeddings_tsne_pca.pdf", dpi=300)    
    