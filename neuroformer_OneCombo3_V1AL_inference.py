# %%
import os
import glob
import collections

import pickle
import sys
import glob
from pathlib import Path, PurePath
path = Path.cwd()
parent_path = path.parents[1]
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
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"


# for i in {1..10}; do python3 -m gather_atts.py; done

# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""

RESPONSE_PATH = "data/Combo3_V1AL/Combo3_V1AL_response.csv"
STIMULUS_PATH = "data/Combo3_V1AL/Combo3_V1AL_stimulus.pt"

if not os.path.exists(RESPONSE_PATH):
    print("Downloading data...")
    import gdown
    url = "https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=share_link"
    gdown.download_folder(id=url, quiet=False, use_cookies=False)



df = pd.read_csv(RESPONSE_PATH)
video_stack = torch.load(STIMULUS_PATH)

# %%
print(video_stack.shape)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    ax[i].imshow(video_stack[i, 1, 0].permute(0, 1))

# %%
# df = pd.read_csv(parent_path + "code/data/OneCombo3/Combo3_all_stim.csv")
frame_window = 0.5
window = 0.05
window_prev = 0.5
dt = 0.05

from SpikeVidUtils import make_intervals

df['Interval'] = make_intervals(df, window)
df['real_interval'] = make_intervals(df, 0.05)
# df['Interval_2'] = make_intervals(df, window_prev)
# df['Interval_dt'] = make_intervals(df, dt)
# df['Interval_dt'] = (df['Interval_dt'] - df['Interval'] + window).round(3)
df = df.reset_index(drop=True)

# %%
from SpikeVidUtils import get_interval_trials

# intervals = get_interval_trials(df, window, window_prev, dt)
intervals = None

# %%
top_p = 0.75
top_p_ids = df.groupby('ID').count().sort_values(by='Trial', ascending=False)[:int(top_p * len(df['ID'].unique()))].index.tolist()
df = df[df['ID'].isin(top_p_ids)].reset_index(drop=True)
df.groupby('ID').count().sort_values(by='Trial', ascending=False)[:int(top_p * len(df['ID'].unique()))]

# %%
# n_dt = sorted((df['Interval_dt'].unique()).round(2)) 
max_window = max(window, window_prev)
dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
n_dt = [round(dt * n, 2) for n in range(dt_range)]
df['Time'] = df['Time'].round(4)

# %%
# df.groupby(['Interval', 'Trial']).size().plot.bar()
# df.groupby(['Interval', 'Trial']).agg(['nunique'])model_path
n_unique = len(df.groupby(['Interval', 'Trial']).size())
df.groupby(['Interval', 'Trial']).size().nlargest(int(0.2 * n_unique))
# df.groupby(['Interval_2', 'Trial']).size().mean()

# %%
video_stack.shape

# %%
from SpikeVidUtils import SpikeTimeVidData2


## resnet3d feats
kernel_size = (10, 8, 8)
n_embd = 256
n_embd_frames = 64
frame_feats = video_stack

frame_block_size = ((20 // kernel_size[0] * 64 * 112) // (n_embd_frames))
# frame_block_size = 5 * 14 * 14
frame_feats = video_stack.transpose(1, 2)

# frame_block_size = 560
prev_id_block_size = 30
id_block_size = 30   # 95
block_size = frame_block_size + id_block_size + prev_id_block_size # frame_block_size * 2  # small window for faster training
frame_memory = 20   # how many frames back does model see
window = window

neurons = sorted(list(set(df['ID'])))
id_stoi = { ch:i for i,ch in enumerate(neurons) }
id_itos = { i:ch for i,ch in enumerate(neurons) }

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

# %%

n = []
n_trial = [2, 8, 14, 19]
for n_stim in range(df['Trial'].max() // 20):
    # n_trial = [2, 4, 6, 8, 10, 12, 14, 18]
    for n_t in n_trial:
        trial = (n_stim + 1) * 20 - (n_t)
        n.append(trial)
train_data = df[~df['Trial'].isin(n)].reset_index(drop=True)
test_data = df[df['Trial'].isin(n)].reset_index(drop=True)
small_data = df[df['Trial'].isin([5])].reset_index(drop=True)

# %%
len(train_data['Trial'].unique()) / (len(train_data['Trial'].unique()) + len(test_data['Trial'].unique()))

# %%
frame_feats.shape

# %%
from SpikeVidUtils import SpikeTimeVidData2

# train_dat1aset = spikeTimeData(spikes, block_size, dt, stoi, itos)

train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, 
                                  prev_id_block_size, window, dt, frame_memory, 
                                  stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, 
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  intervals=intervals, start_interval=1)
test_dataset = SpikeTimeVidData2(test_data, None, block_size, id_block_size, frame_block_size, 
                                  prev_id_block_size, window, dt, frame_memory, 
                                  stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, 
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  intervals=intervals, start_interval=1)
# dataset = SpikeTimeVidData(df, frames, frame_feats, block_size, frame_block_size, prev_id_block_size, window, frame_memory, stoi, itos)
# single_batch = SpikeTimeVidData(df[df['Trial'].isin([5])], None, block_size, frame_block_size, prev_id_block_size, window, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats)
small_dataset = SpikeTimeVidData2(small_data, None, block_size, id_block_size, frame_block_size, 
                                  prev_id_block_size, window, dt, frame_memory, 
                                  stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, 
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  intervals=intervals, start_interval=1)


print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')

# %%
"""

Load Model

"""

from model_neuroformer import GPT

precision = []
recall = []
f1 = []

n_seed = 25
models = []
# for n in [0, 25, 50]:
# set_seed(25)

from model_neuroformer import GPT, GPTConfig
# initialize config class and model (holds hyperparameters)
# for is_conv in [True, False]:    
conv_layer = True
mconf = GPTConfig(train_dataset.population_size, block_size,    # frame_block_size
                        id_vocab_size=train_dataset.id_population_size,
                        frame_block_size=frame_block_size,
                        id_block_size=id_block_size,  # frame_block_size
                        prev_id_block_size=prev_id_block_size,
                        sparse_mask=False, p_sparse=0.25, sparse_topk_frame=None, sparse_topk_id=None,
                        n_dt=len(n_dt),
                        data_size=train_dataset.size,
                        class_weights=None,
                        pretrain=False,
                        n_state_layers=4, n_state_history_layers=4, n_stimulus_layers=8, self_att_layers=4,
                        n_layer=10, n_head=2, n_embd=n_embd, n_embd_frames=n_embd_frames, 
                        contrastive=True, clip_emb=1024, clip_temp=0.5,
                        conv_layer=conv_layer, kernel_size=kernel_size,
                        temp_emb=True, pos_emb=False,
                        id_drop=0.2, im_drop=0.2,
                        window=window, window_prev=window_prev, frame_window=frame_window, dt=dt,
                        neurons=neurons, stoi_dt=stoi_dt, itos_dt=itos_dt)  # 0.35
model = GPT(mconf)
# model.load_state_dict(torch.load("/home/antonis/projects/slab/git/neuroformer/models/tensorboard/V1_AL_cont/cont0+conv_emask_Cont:True_0.50.05_sparseTrue_conv_True_shuffle:True_batch:224_sparse_(None_None)_pos_emb:False_temp_emb:True_drop:0.2_dt:True_2.0_0.5_max0.05_(4, 4, 6)_2_256_nembframe64.pt", map_location='cpu'))

from trainer import Trainer, TrainerConfig
# model.load_state_dict(torch.load(parent_path +  "code/transformer_vid3/runs/models/12-01-21-14:18-e:19-b:239-l:4-h:2-ne:512-higher_order.pt"))


layers = (mconf.n_state_layers, mconf.n_state_history_layers, mconf.n_stimulus_layers)
max_epochs = 120
batch_size = 32 * 10
shuffle = True
title =  f'window:{window}'
model_path = f"""./models/tensorboard/V1_AL/{title}/sparse_f:{mconf.sparse_topk_frame}_id:{mconf.sparse_topk_id}/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(n_dt)}_max{dt}_{layers}_{mconf.n_head}_{mconf.n_embd}.pt"""# model.load_state_dict(torch.load("/Users/antonis/Downloads/[16, 17, 18, 19]_Cont_True_0.50.05_sparseFalse_conv_True_shuffle_True_batch_224_sparse_(None_None)_pos_emb_False_temp_emb_True_drop_0.2_dt_True_2.0_0.5_max0.05_(4, 0, 6)_2_256_nembframe64-2.pt", map_location='cpu'))

tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=1e-4, 
                    num_workers=4, lr_decay=False, patience=3, warmup_tokens=8e10, 
                    decay_weights=True, weight_decay=0.1, shuffle=shuffle,
                    final_tokens=len(train_dataset)*(id_block_size) * (max_epochs),
                    clip_norm=1.0, grad_norm_clip=1.0,
                    dataset='higher_order', mode='predict',
                    block_size=train_dataset.block_size,
                    id_block_size=train_dataset.id_block_size,
                    show_grads=False, plot_raster=False,
                    ckpt_path=model_path, no_pbar=False,
                    save_epoch=True)


trainer = Trainer(model, train_dataset, test_dataset, tconf, mconf)


# %%
loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
iterable = iter(train_dataset)

# %%
x, y = next(iterable)

# %%
"""

RUN SIMULATION

"""


from utils import *
from IPython.utils import io
# top_p=0.25, top_p_t=0.9, temp=2.

model_dir = "./models/tensorboard/V1_AL/window:0.05/sparse_f:None_id:None/w:0.05_wp:0.5/6_Cont:True_window:0.05_f_window:0.5_df:0.05_blocksize:30_conv_True_shuffle:True_batch:320_sparse_(None_None)_blocksz284_pos_emb:False_temp_emb:True_drop:0.2_dt:True_2.0_0.5_max0.05_(4, 4, 8)_2_256/"
model_dirs = os.path.join(model_dir, "*.pt")
model_paths = glob.glob(model_dirs)

for idx in range(0, 125):
    model_paths = glob.glob(model_dirs)
    
    if idx % 10 == 0:
        continue

    elif idx < len(model_paths):
        model_path = model_paths[idx]
        title = model_path.split("/")[-1].split(".")[0]

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"LOADED MODEL: {title}")

        trials = train_data['Trial'].unique()[::4]

        results_dict = dict()
        # for n in range(2, 20):
        df_pred = None
        df_true = None
        n_p = 0.3   # (n + 1) * 0.05
        temp = 2
        # stoi['SOS'] = 2000
        for trial in trials:    # test_data['Trial'].unique():
                print(f"Trial: {trial}")
                df_trial = df[df['Trial'] == trial]
                trial_dataset = SpikeTimeVidData2(df_trial, None, block_size, id_block_size, frame_block_size, 
                                                prev_id_block_size, window, dt, frame_memory, 
                                                stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, 
                                                pred=False, window_prev=window_prev, frame_window=frame_window,
                                                intervals=None, start_interval=1)
                trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
                results_trial = predict_raster_recursive_time_auto(model, trial_loader, window, window_prev, stoi, itos_dt, itos=itos, sample=True, top_p=0.95, top_p_t=0.95, temp=1, temp_t=1, frame_end=0, get_dt=True, gpu=False, pred_dt=True)
                df_trial_pred, df_trial_true = process_predictions(results_trial, stoi, itos, window)
                print(f"pred: {df_trial_pred.shape}, true: {df_trial_true.shape}" )
                if df_pred is None:
                    df_pred = df
                    df_true = df
                else:
                    df_pred = pd.concat([df_pred, df])
                    df_true = pd.concat([df_true, df])

        scores = compute_scores(df[df['Trial'].isin(trials)], df_pred)
        print(scores)
        print(f"pred: {len(df_pred)}, true: {len(df_true)}" )
        # results_dict[n] = (scores)


        # %%
        """

        Split data into full-stimulus trials

        """


        t_1, t_2 = 35, 36
        trial_data_1 = df[df['Trial'] == t_1]
        trial_dataset_1 = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev)
        trial_loader_1 = DataLoader(trial_dataset_1, shuffle=False, pin_memory=False)

        def loader_trial(df, n_trial):
            trial_data = df[df['Trial'] == n_trial]
            trial_dataset = SpikeTimeVidData2(trial_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev)
            trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
            return trial_loader

        trial_data_1 = loader_trial(df, t_1)
        trial_data_2 = loader_trial(df, t_2)

        iterable1 = iter(trial_data_1)
        iterable2 = iter(trial_data_2)


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

        print(f"trials: {test2_data['Trial'].unique()}")
        print(f"trials: {test3_data['Trial'].unique()}")

        from analysis import *
        from utils import *
        from SpikeVidUtils import create_full_trial

        df_pred_full = create_full_trial(df_pred)
        df_1 = create_full_trial(df, trials)
        df_2 = create_full_trial(df, n_1)
        df_3 = create_full_trial(df, n_2)

        window_pred = 1
        df_list = [df_pred_full, df_1, df_2, df_3]
        for df_ in df_list:
            df_['Interval'] = make_intervals(df_, window_pred)
            df_ = df_[df_['Interval'] > window_prev]
            df_ = df_[df_['Interval'] < window]


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

        # %%
        """

        Evaluate results

        """


        from visualize import *

        # df_2['Trial'] -= 2
        id_pred, id_true_1, id_true_2 = len(df_pred_full['ID'].unique()), len(df_1['ID'].unique()), len(df_2['ID'].unique())
        print(f"id_pred: {id_pred}, id_true_1: {id_true_1}, id_true_2: {id_true_2}")

        len_pred, len_true = len(df_pred_full), len(df_1)
        print(f"len_pred: {len_pred}, len_true: {len_true}")

        accuracy = get_accuracy(df_pred, df_2)

        scores = compute_scores(df_1, df_2)
        pred_scores = compute_scores(df_1, df_pred_full)
        print(f"real: {scores}")
        print(f"pred: {pred_scores}")

        set_plot_white()
        plt.figure(figsize=(10, 10), facecolor='white')
        plt.title(f'PSTH Correlations (V1 + AL) bin {window_pred}', fontsize=25)
        plt.ylabel('Count (n)', fontsize=25)
        plt.xlabel('Pearson r', fontsize=25)
        plt.hist(top_corr_real, label='real - real2', alpha=0.6)
        # plt.hist(top_corr_real_2, label='real - real3', alpha=0.6)
        plt.hist(top_corr_pred, label='real - simulated', alpha=0.6)
        plt.legend(fontsize=20)
        plt.show()

        # dir_name = os.path.dirname(model_path)
        dir_name = model_dir
        model_name = os.path.basename(model_path)
        save_dir = os.path.join(dir_name, title)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, F'psth_corr_{title}.svg'))
        df_pred.to_csv(os.path.join(save_dir, F'df_pred_{title}.csv'))

        plot_distribution(df_1, df_pred, save_path=os.path.join(save_dir, F'psth_dist_{title}.svg'))

        total_scores = dict()
        total_scores['real'] = scores
        total_scores['pred'] = pred_scores

        print(f"model: {title}")




# %%
