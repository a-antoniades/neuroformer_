import os
import glob
import collections

import pickle
import json
import sys
import glob
from pathlib import Path, PurePath
from unicodedata import name
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


def load_V1_AL(stimulus_path, response_path, top_p_ids=None):
    if stimulus_path is None:
        # stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
        stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/OneCombo3/stimuli"
    if response_path is None:
        # response_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/neural/NatureMoviePart1-A/20-NatureMovie_part1-A_spikes(1).mat"
        response_path = "/home/antonis/projects/slab/git/neuroformer/data/Combo3_V1AL"
    
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
    # filenames = ['Combo3_V1_shuffle.mat', 'Combo3_AL_shuffle.mat']

    files = []
    for filename in filenames: 
        spike_data = scipyio.loadmat(os.path.join(response_path, filename))
        spike_data = np.squeeze(spike_data['spiketrain'].T, axis=-1)
        # spike_data = np.squeeze(spike_data['spiketrain_shuffle'].T, axis=-1)
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
    if top_p_ids is not None:
        if isinstance(top_p_ids, int) or isinstance(top_p_ids, float):
            top_p_ids = df.groupby('ID').count().sort_values(by='Trial', ascending=False)[:int(top_p_ids * len(df['ID'].unique()))].index.tolist()
        df = df[df['ID'].isin(top_p_ids)]

    test_trials = []
    n_trial = [2, 8, 14, 19]
    for n_stim in range(df['Trial'].max() // 20):
        # n_trial = [2, 4, 6, 8, 10, 12, 14, 18]
        for n_t in n_trial:
            trial = (n_stim + 1) * 20 - (n_t)
            test_trials.append(trial)

    return video_stack, df, test_trials


def load_natural_movie(stimulus_path=None, response_path=None, top_p_ids=None):
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

    if top_p_ids is not None:
        df = df[df['ID'].isin(top_p_ids)]

    n_trial = [4]

    return video_stack, df, n_trial


@torch.no_grad()
def generate_simulation(dataset_name, model_dir, top_p_ids=0.75, stimulus_path=None, response_path=None, save_data=True):

    '''
    from simulation import generate_simulation

    data = generate_simulation('V1_AL', model_dir)
    '''
    if dataset_name == 'V1_AL':
        video_stack, df_full, test_trials = load_V1_AL(stimulus_path, response_path, top_p_ids=top_p_ids)
    elif dataset_name == 'natural_movie':
        video_stack, df_full, test_trials = load_natural_movie(stimulus_path, response_path, top_p_ids=top_p_ids)
    

    model_path = glob.glob(os.path.join(model_dir, "*.pt"))[0]
    mconf_path = glob.glob(os.path.join(model_dir, "*_mconf.pkl"))[0]
    tconf_path = glob.glob(os.path.join(model_dir, "*_tconf.pkl"))[0]


    with open(mconf_path, 'rb') as handle:
        mconf = pickle.load(handle)
    with open(tconf_path, 'rb') as handle:
        tconf = pickle.load(handle)
    
    # """ used if windows not specified in mconf"""
    # window, window_prev, frame_window, dt = 0.05, 0.5, 0.5 , 0.05
    # for config in [mconf, tconf]:
    #     config.window = window
    #     config.window_prev = window_prev
    #     config.frame_window = frame_window
    #     config.dt = dt
    
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    frame_window = mconf.frame_window if hasattr(mconf, 'frame_window') else tconf.window
    window = mconf.window if hasattr(mconf, 'window') else tconf.window
    window_prev = mconf.window_prev if hasattr(mconf, 'window_prev') else float(model_dir.split('/')[-2].split(':')[-1])
    
    # frame_window = round(float(model_dir.split('/')[-3].split(':')[-1]), 2)
    # window = round(float(model_dir.split('/')[-2].split(':')[-1]), 2)
    # window_prev = round(float(model_dir.split('/')[-1].split(':')[-1]), 2)
    dt = mconf.dt if hasattr(mconf, 'dt') else tconf.dt
    model_path
    print(f"--- Window {window}, Frame Window {frame_window} ---")

    df = df_full.copy()

    df['Interval'] = make_intervals(df, window)
    df = df.reset_index(drop=True)
    print(f"------Unique IDs: {len(df['ID'].unique())}-------")

    # n_dt = sorted((df['Interval_dt'].unique()).round(2)) 
    max_window = max(window, window_prev)
    dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
    n_dt = [round(dt * n, 2) for n in range(dt_range)]
    df['Time'] = df['Time'].round(4)
    
    ## resnet3d feats
    kernel_size = mconf.kernel_size
    n_embd = 256
    n_embd_frames = 64

    frame_block_size = int(((frame_window * 20 // kernel_size[0] * 64 * 112) // (n_embd_frames)))
    frame_feats = video_stack

    # frame_block_size = 560
    block_size = tconf.block_size
    id_block_size = tconf.id_block_size   # 95
    prev_id_block_size = id_block_size
    frame_memory = int(frame_window * 20)   # how many frames back does model see

    # translate neural embeddings to separate them from ID embeddings
    neurons = sorted(list(set(df['ID'].unique())))
    trial_tokens = [f"Trial {n}" for n in df['Trial'].unique()]
    feat_encodings = neurons + ['SOS'] + ['EOS'] + ['PAD']  # + pixels 
    stoi = { ch:i for i,ch in enumerate(feat_encodings) }
    itos = { i:ch for i,ch in enumerate(feat_encodings) }
    stoi_dt = { ch:i for i,ch in enumerate(n_dt) }
    itos_dt = { i:ch for i,ch in enumerate(n_dt) }
    max(list(itos_dt.values()))

    # set model attributes according to configs
    # set_model_attributes(mconf)
    # set_model_attributes(tconf)
    for a in dir(tconf):
        if not a.startswith('__'):
            globals()[a] = getattr(tconf, a)
    for a in dir(mconf):
        if not a.startswith('__'):
            globals()[a] = getattr(mconf, a)

    test_data = df[df['Trial'].isin(test_trials)].reset_index(drop=True)
    trials = test_data['Trial'].unique()

    print(f"TRIALS -------------------{trials}")

    # for n in range(2, 20):
    df_pred = None
    df_true = None
    for trial in trials:    # test_data['Trial'].unique():
        # with io.capture_output() as captured:
        df_trial = df[df['Trial'] == trial]
        trial_dataset = SpikeTimeVidData2(df_trial, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                          window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, 
                                          pred=False, window_prev=window_prev, frame_window=frame_window)
        trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
        results_trial = predict_raster_recursive_time_auto(model, trial_loader, window, window_prev, stoi, itos_dt, itos=itos, 
                        sample=True, top_p=0.75, top_p_t=0.9, temp=1.25, temp_t=1, frame_end=0, get_dt=True, gpu=False, pred_dt=True)
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


    if dataset_name == 'V1_AL':

        df = df[df['Interval'] > 2]
    
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

        df_pred_full = create_full_trial(df_pred)
        df_1 = create_full_trial(df, trials)
        df_2 = create_full_trial(df, n_1)
        df_3 = create_full_trial(df, n_2)
   
    elif dataset_name == 'natural_movie':
        n_1, n_2, n_3 = test_data['Trial'].unique(), 2, 7
        df_1 = test_data
        df_2 = df[df['Trial'] == n_2]
        df_3 = df[df['Trial'] == n_3]
        df_pred_full = df_pred


    window_pred = None

    df_list = [df_pred, df_pred_full, df_1, df_2, df_3]

    for df_ in df_list:
        window_pred = 0.5
        df_['Interval'] = make_intervals(df_, window_pred)

    window_pred = window if window_pred is None else window_pred
    intervals = np.array(sorted(set(df['Interval'].unique()) & set(df['Interval'].unique())))
    labels = np.array([round(window_pred + window_pred*n, 2) for n in range(0, int(max(df_pred_full['Interval']) / window_pred))])
    ids = sorted(set(df['ID'].unique()) & set(df['ID'].unique()))

    rates_pred = get_rates_trial(df_pred_full, labels)
    rates_1 = get_rates_trial(df_1, labels)
    rates_2 = get_rates_trial(df_2, labels)
    rates_3 = get_rates_trial(df_3, labels)

    top_corr_pred = calc_corr_psth(rates_pred, rates_1)
    top_corr_real = calc_corr_psth(rates_1, rates_2)
    top_corr_real_2 = calc_corr_psth(rates_1, rates_3)

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

        df_pred_full.to_csv(os.path.join(inference_path, "df_pred.csv"))


    wanted_vars = ['df_pred_full', 'df_1', 'df_2', 'rates_pred', 'rates_1', 'rates_2', 
                   'top_corr_pred', 'top_corr_real', 'scores', 'pred_scores',
                   'mconf', 'tconf', 'model']
    var_dict = {k:v for (k,v) in locals().items() if k in wanted_vars}

    return var_dict



if __name__ == "__main__":
    # experiment_path = "/home/antonis/projects/slab/git/neuroformer/models/tensorboard/Naturalmovie/window_comparison"
    # f_windows = os.listdir(experiment_path)

    # for f_w in f_windows:
    #     f_w_path = os.path.join(experiment_path, f_w)
    #     w_windows = os.listdir(f_w_path)
    #     for w_w in w_windows:
    #         model_dir = os.path.join(experiment_path, f_w, w_w)
    #         generate_simulation('natural_movie', model_dir, save_data=True)


    from simulation import generate_simulation
    from utils import NestedDefaultDict

    # results = NestedDefaultDict()

    # model_dirs = "/home/antonis/projects/slab/git/neuroformer/models/tensorboard/V1_AL/window_comparison_2"
    # model_paths = glob.glob(os.path.join(model_dirs, "**/*.pt"), recursive=True)
    # model_tit = '25_Cont:True_window:0.3_f_window:1.0_df:0.05_blocksize:35_sparseFalse_conv_True_shuffle:True_batch:192_sparse_(None_None)_blocksz182_pos_emb:False_temp_emb:True_drop:0.35_dt:True_2.0_0.3_max0.05_(2, 2, 4)_2_256_nembframe64_(20, 8, 8).pt'

    # for model_path in model_paths:
    #     model_dir = os.path.dirname(model_path)
    #     model_title = os.path.basename(model_path)
    #     if model_title != model_tit:
    #         continue
    #     # mconf = glob.glob(os.path.join(model_dir, "**/mconf.pkl"), recursive=True)[0]
    #     # tconf = glob.glob(os.path.join(model_dir, "**/tconf.pkl"), recursive=True)[0]

    model_dir = "/home/antonis/projects/slab/git/neuroformer/models/tensorboard/V1_AL/shuffle/not_shuffled2"
    sim = generate_simulation("V1_AL", model_dir)

    mconf = sim['mconf']
    tconf = sim['tconf']
    # window = tconf.window if hasattr(tconf, 'window') else mconf.window
    # window_prev = tconf.window_prev if hasattr(tconf, 'window_prev') else mconf.window_prev
    # frame_window = tconf.frame_window if hasattr(tconf, 'frame_window') else mconf.frame_window
    
    # results[frame_window][window][window_prev] = sim['pred_scores']





