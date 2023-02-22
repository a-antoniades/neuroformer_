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
from torch.utils.data.dataloader import DataLoader

import math
from torch.utils.data import Dataset

from model_perceiver import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import set_seed


from scipy import io as scipyio
from scipy.special import softmax
import skimage
import skvideo.io
from utils import print_full

import matplotlib.pyplot as plt
from utils import *
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

from SpikeVidUtils import image_dataset
from SpikeVidUtils import trial_df_combo3


def prepare_onecombo_real(window, dt,):
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

    from SpikeVidUtils import make_intervals

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