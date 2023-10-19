# %%
import glob
import os
import collections

import pickle
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

from trainer import Trainer, TrainerConfig
from utils import set_seed


from scipy import io as scipyio
from scipy.special import softmax
import skimage
import skvideo.io
from utils import print_full
from scipy.ndimage import gaussian_filter, uniform_filter


import matplotlib.pyplot as plt
from utils import *
from visualize import *
set_plot_params()
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"


from neuroformer.model_neuroformer_ import GPT, GPTConfig, neuralGPTConfig
from trainer import Trainer, TrainerConfig

import json
# for i in {1..10}; do python3 -m gather_atts.py; done


def load_natmovie(data_dir, response_path, stimulus_path, base_path):

    # %%
    DATASET = 'NaturalMovie'
    DATASET = 'NaturalStim'

    # %%
    # data_dir = "./data/NaturalMovie/"
    # data_dir = "./data/NaturalStim/"
    # response_path = "././data/NaturalStim/20-NatureMovie_part1-A_spikes(1).mat"
    # stimulus_path = "././data/NaturalMovie/stimulus/docuMovie.pt"
    # base_path = f"./configs/{DATASET}"

    if not os.path.exists(data_dir):
        print("Downloading data...")
        import gdown
        url = "https://drive.google.com/drive/folders/1jgYBERZpXdbAP-E5xcSAHsWSa95Z9IFe?usp=sharing"
        gdown.download_folder(id=url, quiet=False, use_cookies=False, output="data/")


    # %%
    # load config files
    import yaml

    with open(os.path.join(base_path, 'mconf.yaml'), 'r') as stream:
        mconf = yaml.full_load(stream)

    with open(os.path.join(base_path, 'tconf.yaml'), 'r') as stream:
        tconf = yaml.full_load(stream)

    # with open(os.path.join(base_path, 'dconf.yaml'), 'r') as stream:
    #     dconf = yaml.full_load(stream)

    import omegaconf
    from omegaconf import OmegaConf

    # open yaml as omegacong
    mconf = OmegaConf.create(mconf)
    tconf = OmegaConf.create(tconf)
    # dconf = OmegaConf.create(dconf)


    # %%
    frame_window = 0.05
    window = 0.05
    window_prev = 0.2
    window_behavior = window
    dt = 0.005
    dt_frames = 0.05
    dt_vars = 0.05
    intervals = None
    n_frames = frame_window // dt_frames

    # %%
    ## choose modalities ##

    # behavior
    visual_stim = True

    # %%
    from neuroformer.SpikeVidUtils import trial_df_real, make_intervals
    from neuroformer.prepare_data import load_natmovie_real

    df, stimulus = load_natmovie_real(response_path, stimulus_path, dt_frames)


    df['Interval'] = make_intervals(df, window)
    df['real_interval'] = make_intervals(df, 0.05)
    df['Interval_2'] = make_intervals(df, window_prev)

    # # randomly permute ID column
    # df['ID'] = np.random.permutation(df['ID'].values)

    df = df.reset_index(drop=True)

    max_window = max(window, window_prev)
    dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
    n_dt = [round(dt * n, 2) for n in range(dt_range)] + ['EOS'] + ['PAD']


    # %%
    # int_trials = df.groupby(['Interval', 'Trial']).size()
    # print(int_trials.mean())
    # df.groupby(['Interval', 'Trial']).agg(['nunique'])
    var_group = 'Interval'
    n_unique = len(df.groupby([var_group, 'Trial']).size())
    # df.groupby([var_group, 'Trial']).size().nlargest(int(0.2 * n_unique))
    df.groupby([var_group, 'Trial']).size().sort_values(ascending=False).nlargest(int(0.05 * n_unique))
    # df.groupby([var_group, 'Trial']).size().sort_values(ascending=False).nsmallest(int(0.99 * n_unique))

    # %%
    ## resnet3d feats
    n_embd = 256
    n_embd_frames = 64
    conv_layer = True
    kernel_size = (1, 8, 8)

    frame_block_size = ((20 // kernel_size[0] * 64 * 112) // (n_embd_frames))
    frame_feats = stimulus if visual_stim else None
    frame_block_size = (20 * 64 * 112) // (n_embd_frames)
    # frame_block_size = 560
    prev_id_block_size = 67
    id_block_size = prev_id_block_size   # 95
    block_size = frame_block_size + id_block_size + prev_id_block_size # frame_block_size * 2  # small window for faster training
    frame_memory = 20   # how many frames back does model see
    window = window

    neurons = sorted(list(set(df['ID'])))
    id_stoi = { ch:i for i,ch in enumerate(neurons) }
    id_itos = { i:ch for i,ch in enumerate(neurons) }

    neurons = [i for i in range(df['ID'].min(), df['ID'].max() + 1)]
    feat_encodings = neurons + ['SOS'] + ['EOS'] + ['PAD']  # + pixels 
    stoi = { ch:i for i,ch in enumerate(feat_encodings) }
    itos = { i:ch for i,ch in enumerate(feat_encodings) }
    stoi_dt = { ch:i for i,ch in enumerate(n_dt) }
    itos_dt = { i:ch for i,ch in enumerate(n_dt) }


    # %%
    import random

    r_split = 0.8
    all_trials = sorted(df['Trial'].unique())
    train_trials = random.sample(all_trials, int(len(all_trials) * r_split))

    train_data = df[df['Trial'].isin(train_trials)]
    test_data = df[~df['Trial'].isin(train_trials)]

    # %%
    from neuroformer.SpikeVidUtils import SpikeTimeVidData2


    train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                    window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                    pred=False, window_prev=window_prev, frame_window=frame_window,
                                    dt_frames=dt_frames, intervals=None, dataset=DATASET,
                                    window_behavior=window_behavior)
    test_dataset = SpikeTimeVidData2(test_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                    window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                    pred=False, window_prev=window_prev, frame_window=frame_window,
                                    dt_frames=dt_frames, intervals=None, dataset=DATASET,
                                    dt_vars=dt_vars, 
                                    window_behavior=window_behavior)

    print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')




    # %%
    loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    iterable = iter(loader)

    # %%
    x, y = next(iterable)
    # print(x['behavior'].shape, x['behavior_dt'].shape)
    for k in x.keys():
        print(k, x[k].shape)

    # %%
    # for trial in df['Trial'].unique():
    #     video_trial = stimulus[trial]
    #     video_duration = video_trial.shape[1] * dt_frames
    #     df.drop(df[(df['Trial'] == trial) & (df['Time'] > video_duration)].index, inplace=True)
    # df = df.reset_index(drop=True)

    # %%
    from neuroformer.model_neuroformer_ import GPT, GPTConfig

    layers = (mconf.n_state_layers, mconf.n_state_history_layers, mconf.n_stimulus_layers)   
    max_epochs = 300
    batch_size = round((32 * 6))
    shuffle = True

    title =  f'window:{window}_prev:{window_prev}_contatpad'

    # model_path = f"""./models/tensorboard/{DATASET}/{title}/sparse_f:{mconf.sparse_topk_frame}_id:{mconf.sparse_topk_id}/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(stoi_dt.values())}_max{dt}_{layers}_{mconf.n_head}_{mconf.n_embd}.pt"""

    model_conf = GPTConfig(train_dataset.population_size, block_size,    # frame_block_size
                            id_vocab_size=train_dataset.id_population_size,
                            frame_block_size=frame_block_size,
                            id_block_size=id_block_size,  # frame_block_size
                            prev_id_block_size=prev_id_block_size,
                            sparse_mask=False, p_sparse=None, 
                            sparse_topk_frame=None, sparse_topk_id=None, sparse_topk_prev_id=None,
                            n_dt=len(n_dt),
                            data_size=train_dataset.size,
                            class_weights=None,
                            pretrain=False,
                            n_state_layers=mconf.n_state_layers, n_state_history_layers=mconf.n_state_history_layers,
                            n_stimulus_layers=mconf.n_stimulus_layers, self_att_layers=mconf.self_att_layers,
                            n_behavior_layers=mconf.n_behavior_layers,
                            n_head=mconf.n_head, n_embd=mconf.n_embd, 
                            contrastive=mconf.contrastive, clip_emb=1024, clip_temp=mconf.clip_temp,
                            conv_layer=conv_layer, kernel_size=kernel_size,
                            temp_emb=mconf.temp_emb, pos_emb=False,
                            id_drop=0.35, im_drop=0.35, b_drop=0.45,
                            window=window, window_prev=window_prev, frame_window=frame_window, dt=dt,
                            neurons=neurons, stoi_dt=stoi_dt, itos_dt=itos_dt, n_embd_frames=n_embd_frames,
                            ignore_index_id=stoi['PAD'], ignore_index_dt=stoi_dt['PAD'])  # 0.35

    model = GPT(model_conf)

    return df, stimulus, train_dataset, test_dataset, model