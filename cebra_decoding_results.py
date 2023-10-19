# %%

import glob
import os

import sys
import glob
from pathlib import Path, PurePath
path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, 'neuroformer')))
sys.path.append('neuroformer')
sys.path.append('.')
sys.path.append('../')

import pandas as pd
import numpy as np

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader

import math

from neuroformer.model_neuroformer import GPT, GPTConfig
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils_2 import (set_seed, update_object, 
                                 check_common_attrs, running_jupyter, 
                                 all_device, load_config, update_config, 
                                 dict_to_object, object_to_dict, recursive_print)
from neuroformer.visualize import set_plot_params
from neuroformer.SpikeVidUtils import make_intervals, round_n, SpikeTimeVidData2
import gdown

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

import argparse
from neuroformer.SpikeVidUtils import round_n

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# %%
DT = 0.05

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Specify the dataset')

# Add an argument
parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')

# Parse the arguments
args = parser.parse_args()

# Use the argument
DATASET = args.dataset

# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""

from neuroformer.prepare_data import DataLinks
from neuroformer.DataUtils import round_n, get_frame_idx
from neuroformer.DataUtils import resample_spikes


if DATASET in ["first", "visnav"]:
    data_path = "./data/VisNav_VR_Expt"
elif DATASET == "medial":
    data_path = "./data/VisNav_VR_Expt/MedialVRDataset/"
elif DATASET == "lateral":
    data_path = "./data/VisNav_VR_Expt/LateralVRDataset"


train_data = pd.read_csv(os.path.join(data_path, "train_data.csv"))

spikes_path = f"{data_path}/NF_1.5/spikerates_dt_0.01.npy"
speed_path = f"{data_path}/NF_1.5/behavior_speed_dt_0.05.npy"
stim_path = f"{data_path}/NF_1.5/stimulus.npy"

spikes = resample_spikes(np.load(spikes_path), 0.01, DT).transpose()
speed = np.round(np.load(speed_path), 3).transpose()
stimulus = np.load(stim_path)

frame_feats = None
print(f"spikes: {spikes.shape}, speed: {speed.shape}")

# %%
train_indexes = set([get_frame_idx(value, DT) for value in train_data['Interval']])
test_indexes = set(range(len(speed) - 1)) - train_indexes

print(max(train_indexes), max(test_indexes))

spikes_train = spikes[list(train_indexes)]
print(spikes.shape)
spikes_test = spikes[list(test_indexes)]

speed_train = speed[list(train_indexes)]
speed_test = speed[list(test_indexes)]

print(f"spikes_train: {spikes_train.shape}, spikes_test: {spikes_test.shape}")

# %%

max_iterations = np.arange(100, 10000, 900) #default is 5000.
output_dimension = [2, 3, 8, 16, 32] #here, we set as a variable for hypothesis testing below.

print(f"max_iterations: {max_iterations}, output_dimension: {output_dimension}")

# %%
import pickle
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

max_iterations = np.arange(100, 10000, 1900) #default is 5000.
output_dimension = [2, 3, 8, 16, 32] #here, we set as a variable for hypothesis testing below.
OFFSET = 1

# Store results
results = []

for max_iter in max_iterations:
    for out_dim in output_dimension:
        print(f"max_iterations: {max_iter}, output_dimension: {out_dim}")

        cebra_model = CEBRA(model_architecture=f'offset{OFFSET}-model',
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=1,
                            output_dimension=out_dim,
                            max_iterations=max_iter,
                            distance='cosine',
                            device='cuda_if_available',
                            verbose=True)

        # 1. Train a CEBRA-Time model on the whole dataset
        cebra_model.fit(spikes_train, speed_train)

        print("finished training")
        embedding = cebra_model.transform(spikes_train)
        print("finished embedding")
        embedding_test = cebra_model.transform(spikes_test)
        print("finished embedding test")

        # 3. Train the decoder on the training set
        decoder = cebra.KNNDecoder()
        decoder.fit(embedding, speed_train)

        print(f"embedding_test: {embedding_test.shape}, speed_test: {speed_test.shape}")

        # 5. Get the discrete labels predictions
        prediction = decoder.predict(embedding_test)

        print(f"prediction: {prediction.shape}, speed_test: {speed_test.shape}")

        # compute pearson correlation
        corr, _ = pearsonr(prediction, speed_test)
        print(f"Pearson correlation: {corr}")

        # save predictions
        results.append({
            'max_iterations': max_iter,
            'output_dimension': out_dim,
            'embedding': embedding,
            'embedding_test': embedding_test,
            'prediction': prediction,
            'pearson_correlation': corr
        })

        # Save results
        if DATASET == "medial":
            save_dir = "/share/edc/home/antonis/neuroformer/results/behavior/Medial"
        elif DATASET == "lateral":
            save_dir = "/share/edc/home/antonis/neuroformer/results/behavior/Lateral"
        with open(f'cebra_results_{OFFSET}.pkl', 'wb') as f:
            pickle.dump(results, f)

