import os
import glob

import sys
import os
import glob
from pathlib import Path, PurePath
path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, 'neuroformer')))
sys.path.append('.')
sys.path.append('../')

import pandas as pd
import numpy as np

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

from scipy import io as scipyio
import skimage

import matplotlib.pyplot as plt
# from neuroformer.utils import *
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

from neuroformer.SpikeVidUtils import image_dataset, trial_df_combo3, make_intervals


class DataLinks:
    VisNav_VR_Expt = {
        "url": "https://drive.google.com/drive/folders/117S-7NmbgrqjmjZ4QTNgoa-mx8R_yUso?usp=sharing",
        "DIRECTORY": "data/",
        "RESPONSE_PATH": "./data/VisNav_VR_Expt/experiment_data.mat",
        "STIMULUS_PATH": "./data/VisNav_VR_Expt/stimulus.npy"
    }
    LateralVRDataset = {
        "url": "https://drive.google.com/drive/folders/1F8n1qhqnxm-SIi4Vq9_C8mb82gc09P17?usp=share_link",
        "DIRECTORY": "data/",
        "RESPONSE_PATH": "./data/VisNav_VR_Expt/LateralVRDataset/experiment_data.mat",
        "STIMULUS_PATH": "data/VisNav_VR_Expt/LatervalVRDataset/stimulus.npy"
    }
    MedialVRDataset = {
        "url": "https://drive.google.com/drive/folders/1XJmKvlXNMMBHj_1JsiblfwDMhnk2VJ2C?usp=sharing",
        "RESPONSE_PATH": "./data/VisNav_VR_Expt/MedialVRDataset/experiment_data.mat",
        "DIRECTORY": "data/",
        "STIMULUS_PATH": "./data/VisNav_VR_Expt/MedialVRDataset/stimulus.npy"
    }
    Combo3_V1AL = {
        "url" : "https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=share_link",
        "RESPONSE_PATH": "./data/Combo3_V1AL/Combo3_V1AL_response.csv",
        "STIMULUS_PATH": "./data/Combo3_V1AL/Combo3_V1AL_stimulus.pt"
    }
    Combo3_SimNeu3D = {
        "url": "https://drive.google.com/drive/folders/1k9LrPWpHzpsrsgLMCzSyzcLyWpMRut63?usp=share_link",
        "DIRECTORY": "./data/Combo3_SimNeu3D",
        "RESPONSE_PATH": "./data/Combo3_SimNeu3D/simNeu_3D_Combo4_1000Rep.csv",
        "STIMULUS_PATH": "./data/Combo3_SimNeu3D/OneCombo3_(2,3)_stimuli.pt"
    }
    NaturalStim = {
        "url" : "https://drive.google.com/drive/folders/1jgYBERZpXdbAP-E5xcSAHsWSa95Z9IFe?usp=sharing",
        "RESPONSE_PATH" : "./data/NaturalMovie/stimulus/docuMovie.pt",
        "STIMULUS_PATH" : "./data/NaturalMovie/stimulus/docuMovie.pt"
    }
    NaturalStim_SimNeu3D = {
        "url" : "https://drive.google.com/drive/folders/1jgYBERZpXdbAP-E5xcSAHsWSa95Z9IFe?usp=sharing",
        "DIRECTORY" : "./data/NaturalStim_SimNeu3D",
    }
    HippocampusPos = {
        "url": "https://drive.google.com/drive/folders/15VDz6r8y8-nFpAKPcdGtK6nBtMCOQKQA?usp=share_link",
        "DIRECTORY": "./data/cebra/hippocampus_pos",
        "RESPONSE_PATH": "./data/cebra/hippocampus_pos/Achilles_Spikes.csv",
        "BEHAVIOR_PATH": "./data/cebra/hippocampus_pos/Achilles_Behavior.csv",
        
    }
    HippocampusPosCEBRA = {
    "url": "https://drive.google.com/drive/folders/15VDz6r8y8-nFpAKPcdGtK6nBtMCOQKQA?usp=share_link",
    "DIRECTORY": "./data/cebra/hippocampus_pos",
    "RESPONSE_PATH": "./data/cebra/hippocampus_pos/hippocampus_neural_cerba.csv",
    "BEHAVIOR_PATH": "./data/cebra/hippocampus_pos/hippocampus_pos_cerba.csv",
    
    }




def prepare_onecombo_real(window, dt):
    # R3D: (3 x T x H x W)
    stim_folder = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/OneCombo3/stimuli"
    im_path = ['/Combined Stimuli 3-grating.tif',
            '/Combined Stimuli 3-Movie2.tif',
            '/Combined Stimuli 3-Movie3.tif']

    train_path = "/content/stimulus"
    train_path = "/Users/antonis/Downloads/OneCombo3/stimuli"
    video_stack = [skimage.io.imread(stim_folder + vid) for vid in im_path]
    print(glob.glob(train_path + '/*.tif'))
    video_stack = np.concatenate(video_stack, axis=0)

    video_stack = image_dataset(video_stack)
    video_stack = video_stack[::3]  # convert from 60 to 20 fps
    video_stack = video_stack.view(3, video_stack.shape[0] // 3, video_stack.shape[1], video_stack.shape[2], video_stack.shape[3])

    spike_data = scipyio.loadmat("/home/antonis/projects/slab/git/slab/transformer_exp/code/data/OneCombo3/spiketrain.mat")
    spike_data = np.squeeze(spike_data['spiketrain'].T, axis=-1)
    spike_data = [trial_df_combo3(spike_data, n_stim) for n_stim in range(3)]
    spike_data = pd.concat(spike_data, axis=0)

    spike_data['Trial'] = spike_data['Trial'] + 1
    spike_data['Time'] = spike_data['Time'] * 0.0751
    spike_data = spike_data[(spike_data['Time'] > 0) & (spike_data['Time'] <= 32)]

    # vid_duration = [len(vid) * 1/20 for vid in vid_list]

    df = spike_data
    del spike_data

    # df = pd.read_csv(parent_path + "code/data/OneCombo3/Combo3_all_stim.csv")
    window = 0.5
    dt = 0.05

    df['Interval'] = make_intervals(df, window)
    # df['Interval_dt'] = make_intervals(df, dt)
    # df['Interval_dt'] = (df['Interval_dt'] - df['Interval'] + window).round(3)
    df = df.reset_index(drop=True)

    return video_stack


def load_LRN():
    base_path = "data/LargeRandNet/"
    stim_path = "data/LargeRandNet/LargeRandNet_cosinput.csv"
    response_path = "data/LargeRandNet/LargeRandNet_SpikeTime.csv"

    if not os.path.exists(response_path):
        print("Downloading data...")
        import gdown
        url = "https://drive.google.com/drive/folders/1vxHg7FaFQDjQZUNMgvo5wAOlFIcZ2uv-?usp=share_link"
        gdown.download_folder(id=url, quiet=False, use_cookies=False, output="data/")


    # Load Data
    stimulus = np.loadtxt(stim_path, delimiter=',')
    df = pd.read_csv(response_path, names=['Time', 'ID'])
    df['Time'] = df['Time'].round(4)
    df['Trial'] = df['Time'].apply(lambda x: x // 100 + 1).astype(int)
    df['Time'] = df['Time'].apply(lambda x: x - ((x // 100) * 100)).round(2)

    return df, stimulus


def load_LRL2():
    stim_path = "data/LargeRandLIF2-2/LargeRandNet2_PoissonRate.csv"
    response_path = "data/LargeRandLIF2-2/LargeRandNet2_SpikeTime.csv"

    if not os.path.exists(response_path):
        print("Downloading data...")
        import gdown
        url = "https://drive.google.com/drive/folders/1yDWde9rJ_9nOYN5Ic-_JoAYaW2a2jYOY?usp=sharing"
        gdown.download_folder(id=url, quiet=False, use_cookies=False, output="data/")
    
    stimulus = np.transpose(np.loadtxt(stim_path, delimiter=','), (1, 0))
    df = pd.read_csv(response_path, names=['Time', 'ID'])
    dt_res = 10000
    df['Time'] = df['Time'].round(4)
    df['Trial'] = df['Time'].apply(lambda x: x // dt_res + 1).astype(int)
    df['Time'] = df['Time'].apply(lambda x: x - ((x // dt_res) * dt_res)).round(2)
    df['ID'] = df['ID'].astype(int)

    return df, stimulus


def load_LRL2_small():
    stim_path = "data/LargeRandLIF2_small/LargeRandNet2_PoissonRate_v2.csv"
    response_path = "data/LargeRandLIF2_small/LargeRandNet2_SpikeTime_v2.csv"

    if not os.path.exists(response_path):
        print("Downloading data...")
        import gdown
        url = "https://drive.google.com/drive/folders/1T-xEao7riNu2936nlxuQPh98JrukdkuG?usp=sharing"
        gdown.download_folder(id=url, quiet=False, use_cookies=False, output="data/")
    
    stimulus = np.transpose(np.loadtxt(stim_path, delimiter=','), (1, 0))
    df = pd.read_csv(response_path, names=['Time', 'ID'])
    dt_res = 10000
    df['Time'] = df['Time'].round(4)
    df['Trial'] = df['Time'].apply(lambda x: x // dt_res + 1).astype(int)
    df['Time'] = df['Time'].apply(lambda x: x - ((x // dt_res) * dt_res)).round(2)
    df['ID'] = df['ID'].astype(int)

    return df, stimulus


def load_natmovie_real(response_path, stimulus_path, dt_frames=0.05):
    spike_data = scipyio.loadmat(response_path)
    spike_data = trial_df_real(np.squeeze(spike_data['spiketrain']['st'].T, axis=-1))
    df = spike_data[spike_data['Time'] > 0]
    df['Time'] = df['Time'] * 0.1499

    stimulus = torch.load(stimulus_path)
    for trial in df['Trial'].unique():
        video_trial = stimulus[trial]
        video_duration = video_trial.shape[1] * dt_frames
        df = df[(df['Trial'] != trial) | (df['Time'] <= video_duration)]
    df = df.reset_index(drop=True)

    """
    for trial in df['Trial'].unique():
        max_interval = df[df['Trial'] == trial]['Interval'].max()
        video_dur = stimulus[trial].shape[1] * dt_frames
        print(f"n_movie: {trial}, max_interval: {max_interval}, video_dur: {video_dur}, diff: {max_interval - video_dur}")
    """

    return df, stimulus

# import mat73
# from SpikeVidUtils import make_intervals, get_df_visnav
# def load_visnav_1(behavior, behavior_vars):
#     data_path = "../data/VisNav_VR_Expt/experiment_data.mat"
#     data = mat73.loadmat(data_path)['neuroformer']
#     stimulus = data['vid_sm']
#     response = data['spiketimes']['spks']
#     trial_data = data['trialsummary']
#     # response = data_response['spiketime_sel2']['spks']

#     print(data.keys())

#     df = get_df_visnav(response, trial_data, dt_vars)
#     # df = df[df['ID'].isin(neurons_sel1)].reset_index(drop=True)

#     if behavior is True:
#         behavior = pd.DataFrame({k: data[k] for k in behavior_vars + ['t']})
#         # rename t to time
#         behavior = behavior.rename(columns={'t': 'Time'}) if behavior is not None else None
#         behavior['Interval'] = make_intervals(behavior, window)
#         behavior['Interval_2'] = make_intervals(behavior, window_prev)

#         # prepare speed variables
#         behavior['speed'] = behavior['speed'].apply(lambda x: round_n(x, dt_speed))
#         dt_range_speed = behavior['speed'].min(), behavior['speed'].max()
#         dt_range_speed = np.arange(dt_range_speed[0], dt_range_speed[1] + dt_speed, dt_speed)
#         n_behavior = len(dt_range_speed)

#         stoi_speed = { round_n(ch, dt_speed):i for i,ch in enumerate(dt_range_speed) }
#         itos_speed = { i:round_n(ch, dt_speed) for i,ch in enumerate(dt_range_speed) }
#         assert (window_behavior) % dt_vars < 1e-5, "window + window_prev must be divisible by dt_vars"
#         samples_per_behavior = int((window + window_prev) // dt_vars)
#         behavior_block_size = int((window + window_prev) // dt_vars) * (len(behavior.columns) - 1)
#     else:
#         behavior = None
#         behavior_vars = None
#         behavior_block_size = 0
#         samples_per_behavior = 0
#         stoi_speed = None
#         itos_speed = None
#         dt_range_speed = None
#         n_behavior = None

