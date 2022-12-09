# %%
# !CUDA_VISIBLE_DEVICES=4,5,6,7
# !CUDA_VISIBLE_DEVICES=7,5,3

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

RESPONSE_PATH = "data/SimNeu3D/NaturalMovie/response/NaturalStim_all.csv"
STIMULUS_PATH = "data/SimNeu3D/NaturalMovie/stimulus/docuMovie"

# download data 

if not os.path.exists('data'):
    print('Downloading data...')
    import gdown
    id = 'https://drive.google.com/drive/folders/1Axmr0jNG-IxlQ3g-UOKApO-9LhCZgXL4?usp=sharing'
    gdown.download_folder(id=id, quiet=True, use_cookies=False)

# %%
# R3D: (3 x T x H x W)

from SpikeVidUtils import image_dataset

# def nearest(n, x):
#   u = n % x > x // 2
#   return n + (-1)**(1 - u) * abs(x * u - n % x)

# vid_paths = sorted(glob.glob(STIMULUS_PATH + '/*.tif'))
# vid_list = [skimage.io.imread(vid)[::3] for vid in vid_paths]
# video_stack = [torch.nan_to_num(image_dataset(vid)).transpose(1, 0) for vid in vid_list]
# torch.save({k:v for k, v in enumerate(video_stack)}, '/data5/antonis/projects/neuroformer/data/SimNeu3D/NaturalMovie/stimulus/docuMovie.pt')

vs = torch.load("data/SimNeu3D/NaturalMovie/stimulus/docuMovie.pt")
video_stack = [vs[i] for i in range(len(vs))]

# plt.imshow(video_stack[0][0, 0])

# %%
df = pd.read_csv(RESPONSE_PATH).iloc[:, 1:]
frame_window = 1
window = 0.1
window_prev = 0.1
dt = 0.05

from SpikeVidUtils import make_intervals

df['Interval'] = make_intervals(df, window)
# df['Interval_2'] = make_intervals(df, window_prev)
# df['Interval_dt'] = make_intervals(df, dt)
# df['Interval_dt'] = (df['Interval_dt'] - df['Interval'] + window).round(3)
df = df.reset_index(drop=True)
df.to_csv(f"data/SimNeu3D/NaturalMovie/response/NaturalStim_all_{window}.csv", index=False)

# from SpikeVidUtils import neuron_dict
# data_dict = neuron_dict(df)

# save_path = "data/SimNeu3D/NaturalMovie/response"
# with open(os.path.join(save_path, 'neuron_dict.pkl'), 'wb') as handle:
#     pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_dict = pickle.load(open("data/SimNeu3D/NaturalMovie/response/neuron_dict.pkl", "rb"))

# %%
# n_dt = sorted((df['Interval_dt'].unique()).round(2)) 
max_window = max(window, window_prev)
dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
n_dt = [round(dt * n, 2) for n in range(dt_range)]
df['Time'] = df['Time'].round(4)
n_unique = len(df.groupby(['Interval', 'Trial']).size())

# %%
# df.groupby(['Interval', 'Trial']).size().plot.bar()
# df.groupby(['Interval', 'Trial']).agg(['nunique'])model_path
n_unique = len(df.groupby(['Interval', 'Trial']).size())
df.groupby(['Interval', 'Trial']).size().nlargest(int(0.2 * n_unique))
# df.groupby(['Interval_2', 'Trial']).size().mean()

# %%
from SpikeVidUtils import SpikeTimeVidData2

## qv-vae feats
# frames = torch.load(parent_path + "code/data/SImNew3D/stimulus/vq-vae_code_feats-24-05-4x4x4.pt").numpy() + 2
# frame_feats = torch.load(parent_path + "code/data/SImNew3D/stimulus/vq-vae_embed_feats-24-05-4x4x4.pt").numpy()
# frame_block_size = frames.shape[-1] - 1

## resnet3d feats
kernel_size = (5, 8, 8)
n_embd = 256
n_embd_frames = 64
frame_feats = video_stack

frame_block_size = ((20 // kernel_size[0] * 64 * 112) // (n_embd_frames))
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
neurons = [i for i in range(df['ID'].min(), df['ID'].max() + 1)]
# pixels = sorted(np.unique(frames).tolist())
feat_encodings = neurons + ['SOS'] + ['EOS'] + ['PAD']  # + pixels 
stoi = { ch:i for i,ch in enumerate(feat_encodings) }
itos = { i:ch for i,ch in enumerate(feat_encodings) }
stoi_dt = { ch:i for i,ch in enumerate(n_dt) }
itos_dt = { i:ch for i,ch in enumerate(n_dt) }
max(list(itos_dt.values()))

# %%
n = []
for n_stim in range(df['Trial'].max() // 200):
    n_trial = [i for i in range(200 // 20)]
    for n_trial in n_trial:
        trial = (n_stim + 1) * 20 - n_trial
        n.append(trial)
train_data = df[~df['Trial'].isin(n)].reset_index(drop=True)
test_data = df[df['Trial'].isin(n)].reset_index(drop=True)
small_data = df[df['Trial'].isin([5])].reset_index(drop=True)

# %%
from SpikeVidUtils import SpikeTimeVidData2

# train_dat1aset = spikeTimeData(spikes, block_size, dt, stoi, itos)

train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev, frame_window=frame_window, data_dict=data_dict)
test_dataset = SpikeTimeVidData2(test_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev, frame_window=frame_window, data_dict=data_dict)
# dataset = SpikeTimeVidData(df, frames, frame_feats, block_size, frame_block_size, prev_id_block_size, window, frame_memory, stoi, itos)
# single_batch = SpikeTimeVidData(df[df['Trial'].isin([5])], None, block_size, frame_block_size, prev_id_block_size, window, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats)
small_dataset = SpikeTimeVidData2(small_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev, frame_window=frame_window, data_dict=data_dict)


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

from model_neuroformer import GPT, GPTConfig, neuralGPTConfig, Decoder
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
                        n_state_layers=4, n_state_history_layers=0, n_stimulus_layers=16, self_att_layers=4,
                        n_layer=10, n_head=2, n_embd=n_embd, n_embd_frames=n_embd_frames, 
                        contrastive=True, clip_emb=1024, clip_temp=0.5,
                        conv_layer=conv_layer, kernel_size=kernel_size,
                        temp_emb=True, pos_emb=False,
                        id_drop=0.35, im_drop=0.35,
                        window=window, window_prev=window_prev, frame_window=frame_window, dt=dt,
                        neurons=neurons, stoi_dt=stoi_dt, itos_dt=itos_dt)  # 0.35
model = GPT(mconf)
# model.load_state_dict(torch.load("/home/antonis/projects/slab/git/neuroformer/models/tensorboard/V1_AL_cont/cont0+conv_emask_Cont:True_0.50.05_sparseTrue_conv_True_shuffle:True_batch:224_sparse_(None_None)_pos_emb:False_temp_emb:True_drop:0.2_dt:True_2.0_0.5_max0.05_(4, 4, 6)_2_256_nembframe64.pt", map_location='cpu'))

from trainer import Trainer, TrainerConfig
# model.load_state_dict(torch.load(parent_path +  "code/transformer_vid3/runs/models/12-01-21-14:18-e:19-b:239-l:4-h:2-ne:512-higher_order.pt"))


layers = (mconf.n_state_layers, mconf.n_state_history_layers, mconf.n_stimulus_layers)
max_epochs = 250
batch_size = 32 * 10
shuffle = True
model_path = f"models/tensorboard/natmovie/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_sparse{mconf.sparse_mask}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(n_dt)}_max{dt}_{layers}_{mconf.n_head}_{mconf.n_embd}_nembframe{mconf.n_embd_frames}_{kernel_size}.pt"
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
# model.load_state_dict(torch.load("/Users/antonis/Downloads/[16, 17, 18, 19]_Cont_True_0.50.05_sparseFalse_conv_True_shuffle_True_batch_224_sparse_(None_None)_pos_emb_False_temp_emb_True_drop_0.2_dt_True_2.0_0.5_max0.05_(4, 0, 6)_2_256_nembframe64-2.pt", map_location='cpu'))

tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=3e-4, 
                    num_workers=4, lr_decay=True, patience=3, warmup_tokens=8e6, 
                    decay_weights=True, weight_decay=0.1, shuffle=shuffle,
                    final_tokens=len(train_dataset)*(id_block_size) * (max_epochs),
                    clip_norm=1.0, grad_norm_clip=1.0,
                    dataset='higher_order', mode='predict',
                    block_size=train_dataset.block_size,
                    id_block_size=train_dataset.id_block_size,
                    show_grads=False, plot_raster=False,
                    ckpt_path=model_path, no_pbar=False)
# f"/home/antonis/projects/slab/git/neuroformer/models/model_sim_weighted_shuffle_decay:{shuffle}_perceiver_2.0_dt:{dt}_eos_{mconf.n_layer}_{mconf.n_head}_{mconf.n_embd}.pt")


trainer = Trainer(model, train_dataset, test_dataset, tconf, mconf)
# trainer.train()  

# # %%
# """

# RUN SIMULATION

# """


# from utils import *
# from IPython.utils import io
# # top_p=0.25, top_p_t=0.9, temp=2.

# # for idx in range(len(df_pred['ID'])):
# #     df_pred['ID'][idx] = itos[df_pred['ID'][idx]]

# # model_path = "/home/antonis/projects/slab/git/neuroformer/models/tensorboard/V1_AL_modalities/cont_weighted/25_Cont:True_0.50.1_sparseFalse_conv_True_shuffle:True_batch:224_sparse_(None_None)_pos_emb:False_temp_emb:True_drop:0.2_dt:True_2.0_0.5_max0.1_(4, 4, 6)_2_256_nembframe64.pt"
# # model_path = "/Users/antonis/Downloads/[16, 17, 18, 19]_Cont_True_0.50.05_sparseFalse_conv_True_shuffle_True_batch_224_sparse_(None_None)_pos_emb_False_temp_emb_True_drop_0.2_dt_True_2.0_0.5_max0.05_(4, 0, 6)_2_256_nembframe64-2.pt"
# # model_path = "/Users/antonis/projects/slab/neuroformer/neuroformer/models/tensorboard/V1_AL/sos_clip/25_Cont:True_0.50.1_sparseFalse_conv_True_shuffle:True_batch:96_sparse_(200_4)_pos_emb:False_temp_emb:True_drop:0.2_dt:True_2.0_0.5_max0.1_(2, 2, 2)_2_256_nembframe64_(20, 8, 8).pt"
# model.load_state_dict(torch.load(model_path, map_location='cpu'))

# # trials = np.random.choice(train_data['Trial'].unique(), size=12)
# # trials = test_data['Trial'].unique()
# trials = train_data['Trial'].unique()[::4]
# # trials = train_data['Trial'].unique()
# # trials = train_data['Trial'].unique()[::4]

# # trials = [start + (i * 20) for i in range(3)]
# # trials = df['Trial'].unique()[0::20] 
# results_dict = dict()
# # for n in range(2, 20):
# df_pred = None
# df_true = None
# n_p = 0.3   # (n + 1) * 0.05
# temp = 2
# # stoi['SOS'] = 2000
# for trial in trials:    # test_data['Trial'].unique():
#     # with io.capture_output() as captured:
#         print(f"Trial: {trial}")
#         df_trial = df[df['Trial'] == trial]
#         trial_dataset = SpikeTimeVidData2(df_trial, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
#                                           window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, 
#                                           pred=False, window_prev=window_prev, frame_window=frame_window)
#         trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
#         results_trial = predict_raster_recursive_time_auto(model, trial_loader, window, window_prev, stoi, itos_dt, itos=itos, sample=True, top_p=0.75, top_p_t=0.75, temp=1.3, temp_t=1.3, frame_end=0, get_dt=True, gpu=False, pred_dt=True)
#         # results_trial = predict_raster_hungarian(model, loader, itos_dt, top_p=0.75, temp=1)
#         # print(f"MAX ID ---- {sorted(results_trial['ID'].unique()[-10])}")
#         df_trial_pred, df_trial_true = process_predictions(results_trial, stoi, itos, window)
#         print(f"pred: {df_trial_pred.shape}, true: {df_trial_true.shape}" )
#         if df_pred is None:
#             df_pred = df_trial_pred
#             df_true = df_trial_true
#         else:
#             df_pred = pd.concat([df_pred, df_trial_pred])
#             df_true = pd.concat([df_true, df_trial_true])

# # df_preds[n] = df_pred
# # print(f"--- n: {n}, n_p: {n_p}, temp: {temp} ---")
# scores = compute_scores(df[df['Trial'].isin(trials)], df_pred)
# print(scores)
# print(f"pred: {len(df_pred)}, true: {len(df_true)}" )
# # results_dict[n] = (scores)


# # %%
# """

# Split data into full-stimulus trials

# """


# t_1, t_2 = 35, 36
# trial_data_1 = df[df['Trial'] == t_1]
# trial_dataset_1 = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev)
# trial_loader_1 = DataLoader(trial_dataset_1, shuffle=False, pin_memory=False)

# def loader_trial(df, n_trial):
#     trial_data = df[df['Trial'] == n_trial]
#     trial_dataset = SpikeTimeVidData2(trial_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev)
#     trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
#     return trial_loader

# trial_data_1 = loader_trial(df, t_1)
# trial_data_2 = loader_trial(df, t_2)

# iterable1 = iter(trial_data_1)
# iterable2 = iter(trial_data_2)

# # train_len = round(len(df)*(4/5))
# # test_len = round(len(df) - train_len)

# # train_data = df[:train_len]
# # test_data = df[train_len:train_len + test_len].reset_index().drop(['index'], axis=1)
# n_trial = [2, 8, 14, 19]

# n_1 = []
# for n_stim in range(3): # range(df['Trial'].max() // 20):
#     for n_t in n_trial:
#         trial = (n_stim + 1) * 20 - (n_t - 2)
#         n_1.append(trial)
# test2_data = df[df['Trial'].isin(n_1)].reset_index(drop=True)
# small_data = df[df['Trial'].isin([5])].reset_index(drop=True)


# n_2 = []
# for n_stim in range(3): # range(df['Trial'].max() // 20):
#     for n_t in n_trial:
#         trial = (n_stim + 1) * 20 - (n_t - 1)
#         n_2.append(trial)
# test3_data = df[df['Trial'].isin(n_2)].reset_index(drop=True)
# # small_data = df[df['Trial'].isin([5])].reset_index(drop=True)

# print(f"trials: {test2_data['Trial'].unique()}")
# print(f"trials: {test3_data['Trial'].unique()}")

# from analysis import *
# from utils import *
# from SpikeVidUtils import create_full_trial

# # df_1 = df_pred[df_pred['ID'] < stoi['SO`S']].reset_index(drop=True)
# # df_2 = test_data[test_data['Trial'].isin(trials)].reset_index(drop=True)

# # df_1 = df[df['Trial'].isin(range(10))]
# # df_2 = df[df['Trial'].isin(range(11, 20))]

# df_pred_full = create_full_trial(df_pred)
# df_1 = create_full_trial(df, trials)
# df_2 = create_full_trial(df, n_1)
# df_3 = create_full_trial(df, n_2)

# # df_2['Interval'] += 0.5
# # df_pred_full['Interval'] += 0.5

# # df_1 = create_full_trial(df, t_trial=t_trial, n_start=df_pred['Trial'].min(), n_stim=3, n_step=20, n_trials=10)
# # df_2 = create_full_trial(df, t_trial=t_trial, n_start=df_pred['Trial'].min() + 1, n_stim=3, n_step=20, n_trials=10)
# # df_3 = create_full_trial(df, t_trial=t_trial, n_start=df_pred['Trial'].min() + 2, n_stim=3, n_step=20, n_trials=10)

# # df_1 = df_1[(df_1['Interval'].isin(df_pred_full['Interval'].unique()))].reset_index(drop=True)
# # df_2 = df_2[(df_2['Interval'].isin(df_pred_full['Interval'].unique()))].reset_index(drop=True)
# # df_3 = df_3[(df_3['Interval'].isin(df_pred_full['Interval'].unique()))].reset_index(drop=True)

# window_pred = None

# df_list = [df_pred_full, df_1, df_2, df_3]

# for df_ in df_list:
#     window_pred = 0.5
#     df_['Interval'] = make_intervals(df_, window_pred)

# window_pred = window if window_pred is None else window_pred
# intervals = np.array(sorted(set(df['Interval'].unique()) & set(df['Interval'].unique())))
# labels = np.array([round(window_pred + window_pred*n, 2) for n in range(0, int(max(df_pred_full['Interval']) / window_pred))])
# ids = sorted(set(df['ID'].unique()) & set(df['ID'].unique()))


# # labels = sorted(set(df_pred_full['Interval'].unique()))
# rates_pred = get_rates_trial(df_pred_full, labels)
# rates_1 = get_rates_trial(df_1, labels)
# rates_2 = get_rates_trial(df_2, labels)
# rates_3 = get_rates_trial(df_3, labels)

# top_corr_pred = calc_corr_psth(rates_pred, rates_1)
# top_corr_real = calc_corr_psth(rates_1, rates_2)
# top_corr_real_2 = calc_corr_psth(rates_1, rates_3)

# # %%
# """

# Evaluate results

# """



# from visualize import *

# # df_2['Trial'] -= 2
# id_pred, id_true_1, id_true_2 = len(df_pred_full['ID'].unique()), len(df_1['ID'].unique()), len(df_2['ID'].unique())
# print(f"id_pred: {id_pred}, id_true_1: {id_true_1}, id_true_2: {id_true_2}")

# len_pred, len_true = len(df_pred_full), len(df_1)
# print(f"len_pred: {len_pred}, len_true: {len_true}")

# accuracy = get_accuracy(df_pred, df_2)

# scores = compute_scores(df_1, df_2)
# pred_scores = compute_scores(df_1, df_pred_full)
# print(f"real: {scores}")
# print(f"pred: {pred_scores}")

# set_plot_white()
# plt.figure(figsize=(10, 10), facecolor='white')
# plt.title('PSTH Correlations (V1 + AL)', fontsize=25)
# plt.ylabel('Count (n)', fontsize=25)
# plt.xlabel('Pearson r', fontsize=25)
# plt.hist(top_corr_real, label='real - real2', alpha=0.6)
# # plt.hist(top_corr_real_2, label='real - real3', alpha=0.6)
# plt.hist(top_corr_pred, label='real - simulated', alpha=0.6)
# plt.legend(fontsize=20)

# dir_name = os.path.dirname(model_path)
# plt.savefig(os.path.join(dir_name, 'psth_corr.svg'))

# plot_distribution(df_1, df_pred)

# total_scores = dict()
# total_scores['real'] = scores
# total_scores['pred'] = pred_scores

# %%


# %%



