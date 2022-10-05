import os
import glob
import collections

import pickle
import json
import sys
import glob
from pathlib import Path, PurePath
path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, 'neuroformer')))
sys.path.append('.')
sys.path.append('../')


import pandas as pd
import numpy as np
from einops import rearrange

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data.dataloader import DataLoader

import math
from torch.utils.data import Dataset

from attentionVis import AttentionVis
from trainer import Trainer, TrainerConfig
from utils import set_seed


from scipy import io as scipyio
from scipy.special import softmax
import skimage
import skvideo.io
from utils import print_full
from scipy.ndimage.filters import gaussian_filter, uniform_filter


import matplotlib.pyplot as plt
from utils import *
from visualize import *
set_plot_params()
# parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"


from model_neuroformer import GPT, GPTConfig, neuralGPTConfig, Decoder
from trainer import Trainer, TrainerConfig

# for i in {1..10}; do python3 -m gather_atts.py; done

from utils import set_seed
n_seed = 25
set_seed(n_seed)

from SpikeVidUtils import image_dataset, r3d_18_dataset
from PIL import Image
from SpikeVidUtils import trial_df_combo3
from analysis import *
from utils import *
from SpikeVidUtils import create_full_trial



def load_V1_AL(stimulus_path, response_path):
    if stimulus_path is None:
        stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
    if response_path is None:
        response_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/neural/NatureMoviePart1-A/20-NatureMovie_part1-A_spikes(1).mat"
    
    stim_names = ['/Combined Stimuli 3-grating.tif',
                  '/Combined Stimuli 3-Movie2.tif',
                  '/Combined Stimuli 3-Movie3.tif']

    # video_stack = [skimage.io.imread(os.path.join(stimulus_path, vid)) for vid in stim_names]
    video_stack = [skimage.io.imread(str(stimulus_path + vid)) for vid in stim_names]
    # print(glob.glob(train_path + '/*.tif'))
    video_stack = np.concatenate(video_stack, axis=0)

    video_stack = image_dataset(video_stack)
    video_stack = video_stack[::3]  # convert from 60 to 20 fps
    video_stack = video_stack.view(3, video_stack.shape[0] // 3, video_stack.shape[1], video_stack.shape[2], video_stack.shape[3])
    video_stack = video_stack.transpose(1, 2)
    print(video_stack.shape)
    # spike_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/neural/NatureMoviePart1-A" # "code/data/SImIm/simNeu_3D_WithNorm__Combo3.mat" 
    n_V1_AL = (351, 514)


    df = None
    filenames = ['Combo3_V1.mat', 'Combo3_AL.mat']
    files = []
    for filename in filenames: 
        spike_data = scipyio.loadmat(os.path.join(response_path, filename))
        spike_data = np.squeeze(spike_data['spiketrain'].T, axis=-1)
        spike_data = [trial_df_combo3(spike_data, n_stim) for n_stim in range(3)]
        spike_data = pd.concat(spike_data, axis=0)

        spike_data['Trial'] = spike_data['Trial'] + 1
        spike_data['Time'] = spike_data['Time'] * 0.0751
        spike_data = spike_data[(spike_data['Time'] > 0) & (spike_data['Time'] <= 32)]

        if df is None:
            df = spike_data.reset_index(drop=True)
        else:
            spike_data['ID'] += df['ID'].max() + 1
            df = pd.concat([df, spike_data], axis=0)

    # vid_duration = [len(vid) * 1/20 for vid in vid_list]

    df = df.sort_values(['Trial', 'Time']).reset_index(drop=True)
    df_full = df.copy()
    del spike_data

    top_p = 0.75
    top_p_ids = df.groupby('ID').count().sort_values(by='Trial', ascending=False)[:int(top_p * len(df['ID'].unique()))].index.tolist()

    return video_stack, df, top_p


def load_natural_movie(stimulus_path=None, response_path=None):
    if stimulus_path is None:
        stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
    if response_path is None:
        response_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/neural/NatureMoviePart1-A/20-NatureMovie_part1-A_spikes(1).mat"
    vid_paths = sorted(glob.glob(stimulus_path + '/*.tif'))
    vid_list = [skimage.io.imread(vid)[::3] for vid in vid_paths]
    video_stack = [torch.nan_to_num(image_dataset(vid)).transpose(1, 0) for vid in vid_list]

    spike_data = scipyio.loadmat(response_path)
    spike_data = trial_df_real(np.squeeze(spike_data['spiketrain']['st'].T, axis=-1))
    spike_data = spike_data[spike_data['Time'] > 0]
    vid_duration = [len(vid) * 1/60 for vid in vid_list]

    df = spike_data
    df['Time'] = df['Time'] * 0.1499

    top_p = 0.75
    top_p_ids = df.groupby('ID').count().sort_values(by='Trial', ascending=False)[:int(top_p * len(df['ID'].unique()))].index.tolist()

    return video_stack, df, top_p







def generate_simulation(dataset_name, model_dir, stimulus_path=None, response_path=None, save_data=True):

    '''
    from simulation import generate_simulation

    data = generate_simulation('V1_AL', model_dir)
    '''
    if dataset_name == 'V1_AL':
        video_stack, df_full, top_p_ids = load_V1_AL(stimulus_path, response_path)
    elif dataset_name == 'natural_movie':
        video_stack, df_full, top_p_ids = load_natural_movie(stimulus_path, response_path)
    
    model_path = glob.glob(os.path.join(model_dir, "*.pt"))[0]
    mconf_path = glob.glob(os.path.join(model_dir, "*_mconf.pkl"))[0]
    tconf_path = glob.glob(os.path.join(model_dir, "*_tconf.pkl"))[0]

    with open(mconf_path, 'rb') as handle:
        mconf = pickle.load(handle)
    with open(tconf_path, 'rb') as handle:
        tconf = pickle.load(handle)
    
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    frame_window = tconf.frame_window
    window = tconf.window
    window_prev = window
    # frame_window = round(mconf.kernel_size[0] / 20, 2)
    # window = round(float(w_w.split(':')[-1]), 2)
    # window_prev = window
    dt = tconf.dt
    model_path
    print(f"--- Window {window}, Frame Window {frame_window} ---")

    df = df_full.copy()

    # n_dt = sorted((df['Interval_dt'].unique()).round(2)) 
    max_window = max(window, window_prev)
    dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
    n_dt = [round(dt * n, 2) for n in range(dt_range)]
    df['Time'] = df['Time'].round(4)

    df['Interval'] = make_intervals(df, window)
    # df['Interval_2'] = make_intervals(df, window_prev)
    # df['Interval_dt'] = make_intervals(df, dt)
    # df['Interval_dt'] = (df['Interval_dt'] - df['Interval'] + window).round(3)
    df = df.reset_index(drop=True)
    # top_p = 0.75
    top_p_ids = df.groupby('ID').count().sort_values(by='Trial', ascending=False)[:int(top_p_ids * len(df['ID'].unique()))].index.tolist()
    df = df[df['ID'].isin(top_p_ids)].reset_index(drop=True)

    # n_dt = sorted((df['Interval_dt'].unique()).round(2)) 
    max_window = max(window, window_prev)
    dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
    n_dt = [round(dt * n, 2) for n in range(dt_range)]
    df['Time'] = df['Time'].round(4)

    # df.groupby(['Interval', 'Trial']).size().plot.bar()
    # df.groupby(['Interval', 'Trial']).agg(['nunique'])
    n_unique = len(df.groupby(['Interval', 'Trial']).size())
    df.groupby(['Interval', 'Trial']).size().nlargest(int(0.2 * n_unique))
    # df.groupby(['Interval_2', 'Trial']).size().mean()

    ## resnet3d feats
    kernel_size = (int(20 * frame_window), 8, 8)
    n_embd = 256
    n_embd_frames = 64

    frame_block_size = int(((frame_window * 20 // kernel_size[0] * 64 * 112) // (n_embd_frames)))
    # frame_block_size = 5 * 14 * 14
    frame_feats = video_stack

    # frame_block_size = 560
    block_size = tconf.block_size
    id_block_size = tconf.id_block_size   # 95
    prev_id_block_size = id_block_size
    frame_memory = int(tconf.frame_window * 20)   # how many frames back does model see

    # translate neural embeddings to separate them from ID embeddings
    # frames = frames + [*id_stoi.keys()][-1] 
    # neurons = [i for i in range(df['ID'].min(), df['ID'].max() + 1)]
    neurons = sorted(list(set(df['ID'].unique())))
    # pixels = sorted(np.unique(frames).tolist())
    trial_tokens = [f"Trial {n}" for n in df['Trial'].unique()]
    feat_encodings = neurons + ['SOS'] + ['EOS'] + ['PAD']  # + pixels 
    stoi = { ch:i for i,ch in enumerate(feat_encodings) }
    itos = { i:ch for i,ch in enumerate(feat_encodings) }
    stoi_dt = { ch:i for i,ch in enumerate(n_dt) }
    itos_dt = { i:ch for i,ch in enumerate(n_dt) }
    max(list(itos_dt.values()))


    # train_len = round(len(df)*(4/5))
    # test_len = round(len(df) - train_len)

    # train_data = df[:train_len]
    # test_data = df[train_len:train_len + test_len].reset_index().drop(['index'], axis=1)

    test_trials = []
    n_trial = [2, 8, 14, 19]
    for n_stim in range(df['Trial'].max() // 20):
        # n_trial = [2, 4, 6, 8, 10, 12, 14, 18]
        for n_t in n_trial:
            trial = (n_stim + 1) * 20 - (n_t)
            test_trials.append(trial)
    train_data = df[~df['Trial'].isin(test_trials)].reset_index(drop=True)
    test_data = df[df['Trial'].isin(test_trials)].reset_index(drop=True)
    small_data = df[df['Trial'].isin([5])].reset_index(drop=True)

    # trials = np.random.choice(train_data['Trial'].unique(), size=12)
    trials = test_data['Trial'].unique()
    # trials = train_data['Trial'].unique()[::4]

    # trials = [start + (i * 20) for i in range(3)]
    # trials = df['Trial'].unique()[0::20] 
    results_dict = dict()
    # for n in range(2, 20):
    df_pred = None
    df_true = None
    n_p = 0.3   # (n + 1) * 0.05
    temp = 2
    # stoi['SOS'] = 2000
    for trial in trials:    # test_data['Trial'].unique():
        # with io.capture_output() as captured:
            df_trial = df[df['Trial'] == trial]
            trial_dataset = SpikeTimeVidData2(df_trial, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False,
                                            frame_window=frame_window)
            trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
            results_trial = predict_raster_recursive_time_auto(model, trial_loader, window, window_prev, stoi, itos_dt, itos=itos, sample=True, top_p=0.75, top_p_t=0.9, temp=1.3, temp_t=1, frame_end=0, get_dt=True, gpu=False, pred_dt=True)
            # results_trial = predict_raster_hungarian(model, loader, itos_dt, top_p=0.75, temp=1)
            # print(f"MAX ID ---- {sorted(results_trial['ID'].unique()[-10])}")
            df_trial_pred, df_trial_true = process_predictions(results_trial, stoi, itos, window)
            print(f"pred: {df_trial_pred.shape}, true: {df_trial_true.shape}" )
            if df_pred is None:
                df_pred = df_trial_pred
                df_true = df_trial_true
            else:
                df_pred = pd.concat([df_pred, df_trial_pred])
                df_true = pd.concat([df_true, df_trial_true])

    # df_preds[n] = df_pred
    # print(f"--- n: {n}, n_p: {n_p}, temp: {temp} ---")
    scores = compute_scores(df[df['Trial'].isin(trials)], df_pred)
    print(scores)
    print(f"pred: {len(df_pred)}, true: {len(df_true)}" )
    # results_dict[n] = (scores)

    # train_len = round(len(df)*(4/5))
    # test_len = round(len(df) - train_len)

    # train_data = df[:train_len]
    # test_data = df[train_len:train_len + test_len].reset_index().drop(['index'], axis=1)
    n_trial = [2, 8, 14, 19]

    n_1 = []
    for n_stim in range(3): # range(df['Trial'].max() // 20):
        for n_t in n_trial:
            trial = (n_stim + 1) * 20 - (n_t - 2)
            n_1.append(trial)
    test2_data = df[df['Trial'].isin(n_1)].reset_index(drop=True)
    small_data = df[df['Trial'].isin([5])].reset_index(drop=True)


    n_2 = []
    for n_stim in range(3): # range(df['Trial'].max() // 20):
        for n_t in n_trial:
            trial = (n_stim + 1) * 20 - (n_t - 1)
            n_2.append(trial)
    test3_data = df[df['Trial'].isin(n_2)].reset_index(drop=True)
    # small_data = df[df['Trial'].isin([5])].reset_index(drop=True)

    print(f"trials: {test2_data['Trial'].unique()}")
    print(f"trials: {test3_data['Trial'].unique()}")

    # df_1 = df_pred[df_pred['ID'] < stoi['SO`S']].reset_index(drop=True)
    # df_2 = test_data[test_data['Trial'].isin(trials)].reset_index(drop=True)

    # df_1 = df[df['Trial'].isin(range(10))]
    # df_2 = df[df['Trial'].isin(range(11, 20))]

    df_pred_full = create_full_trial(df_pred)
    df_1 = create_full_trial(df, trials)
    df_2 = create_full_trial(df, n_1)
    df_3 = create_full_trial(df, n_2)

    # df_2['Interval'] += 0.5
    # df_pred_full['Interval'] += 0.5

    # df_1 = create_full_trial(df, t_trial=t_trial, n_start=df_pred['Trial'].min(), n_stim=3, n_step=20, n_trials=10)
    # df_2 = create_full_trial(df, t_trial=t_trial, n_start=df_pred['Trial'].min() + 1, n_stim=3, n_step=20, n_trials=10)
    # df_3 = create_full_trial(df, t_trial=t_trial, n_start=df_pred['Trial'].min() + 2, n_stim=3, n_step=20, n_trials=10)

    # df_1 = df_1[(df_1['Interval'].isin(df_pred_full['Interval'].unique()))].reset_index(drop=True)
    # df_2 = df_2[(df_2['Interval'].isin(df_pred_full['Interval'].unique()))].reset_index(drop=True)
    # df_3 = df_3[(df_3['Interval'].isin(df_pred_full['Interval'].unique()))].reset_index(drop=True)

    window_pred = None

    df_list = [df_pred_full, df_1, df_2, df_3]

    for df_ in df_list:
        window_pred = 0.5
        df_['Interval'] = make_intervals(df_, window_pred)

    window_pred = window if window_pred is None else window_pred
    intervals = np.array(sorted(set(df['Interval'].unique()) & set(df['Interval'].unique())))
    labels = np.array([round(window_pred + window_pred*n, 2) for n in range(0, int(max(df_pred_full['Interval']) / window_pred))])
    ids = sorted(set(df['ID'].unique()) & set(df['ID'].unique()))

    # labels = sorted(set(df_pred_full['Interval'].unique()))
    rates_pred = get_rates_trial(df_pred_full, labels)
    rates_1 = get_rates_trial(df_1, labels)
    rates_2 = get_rates_trial(df_2, labels)
    rates_3 = get_rates_trial(df_3, labels)

    top_corr_pred = calc_corr_psth(rates_pred, rates_1)
    top_corr_real = calc_corr_psth(rates_1, rates_2)
    top_corr_real_2 = calc_corr_psth(rates_1, rates_3)

    # print(f"---- model: {model_title} -----")
    scores = compute_scores(df_1, df_2)
    print(f"real: {scores}")
    pred_scores = compute_scores(df_1, df_pred_full)
    print(f"pred: {pred_scores}")
    
    
    if save_data:
        inference_path = os.path.join(model_dir, "inference")
        if not(os.path.exists(inference_path)):
            os.mkdir(inference_path)

        with open(os.path.join(inference_path, "scores.json"), "w") as f:
            json.dump(pred_scores, f, indent=4)

        df.to_csv(os.path.join(inference_path, "df_pred.csv"))

    return df_pred_full, df_1, df_2, df_3, rates_pred, rates_1, rates_2, rates_3, top_corr_pred, top_corr_real, top_corr_real_2, scores, pred_scores






