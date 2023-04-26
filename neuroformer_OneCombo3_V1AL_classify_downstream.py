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
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader

import math
from torch.utils.data import Dataset

from neuroformer.model_neuroformer import GPT, GPTConfig, neuralGPTConfig
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils import set_seed, update_object
from neuroformer.visualize import set_plot_params
from neuroformer.SpikeVidUtils import round_n
set_plot_params()

from scipy import io as scipyio
from scipy.special import softmax
import skimage
import skvideo.io
from scipy.ndimage import gaussian_filter, uniform_filter

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

import argparse
from neuroformer.SpikeVidUtils import round_n
import argparse



# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

set_seed(25)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--infer", action="store_true", help="Inference mode")
    parser.add_argument("--train", action="store_true", default=False, help="Train mode")
    return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
#     INFERENCE = not args.train
# else:
#     INFERENCE = True

# check if jupyter notebook
try:
    shell = get_ipython().__class__.__name__
    print("Running in Jupyter notebook")
    INFERENCE = True
except:
    print("Running in terminal")
    args = parse_args()
    INFERENCE = not args.train

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
DATASET = "Combo3_V1AL"

if not os.path.exists(RESPONSE_PATH):
    print("Downloading data...")
    import gdown
    url = "https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=share_link"
    gdown.download_folder(id=url, quiet=False, use_cookies=False)



df = pd.read_csv(RESPONSE_PATH)
video_stack = torch.load(STIMULUS_PATH)
stimulus = video_stack[:, :, 0]



# %%
print(video_stack.shape)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    ax[i].imshow(video_stack[i, 1, 0].permute(0, 1))


# %%
# load config files
import yaml

# base_path = "configs/visnav/predict_behavior"
base_path = "./models/tensorboard/V1_AL/downstream4/fixed/sparse_f:None_id:None/w:0.05_wp:0.25"

with open(os.path.join(base_path, 'mconf.yaml'), 'r') as stream:
    mconf = yaml.full_load(stream)

with open(os.path.join(base_path, 'tconf.yaml'), 'r') as stream:
    tconf = yaml.full_load(stream)

with open(os.path.join(base_path, 'dconf.yaml'), 'r') as stream:
    dconf = yaml.full_load(stream)

import omegaconf
from omegaconf import OmegaConf

# open yaml as omegacong
mconf = OmegaConf.create(mconf)
tconf = OmegaConf.create(tconf)
dconf = OmegaConf.create(dconf)


# %%

if INFERENCE:
    frame_window = mconf.frame_window
    window = mconf.window
    window_prev = mconf.window_prev
    window_behavior = mconf.window_behavior if hasattr(mconf, 'window_behavior') else None
    dt = mconf.dt
    dt_frames = mconf.dt_frames if hasattr(mconf, 'dt_frames') else 0.05
    dt_vars = mconf.dt_vars if hasattr(mconf, 'dt_vars') else 0.05
    dt_speed = mconf.dt_speed if hasattr(mconf, 'dt_speed') else 0.2
    intervals = None
else:
    frame_window = 0.25
    window = 0.05
    window_prev = 0.25
    window_behavior = window
    dt = 0.01
    dt_frames = 0.05
    dt_vars = 0.05
    dt_speed = 0.2
    intervals = None



# %%
## choose modalities ##

# behavior
behavior = False
behavior_vars = ['eyerad', 'phi', 'speed', 'th']
# behavior_vars = ['speed']
n_behavior = len(behavior_vars)
predict_behavior = False
# stimulus
visual_stim = True



# %%
# randomly permute 'id' column
df['ID'] = df['ID'].sample(frac=1, random_state=25).reset_index(drop=True)

# %%
from neuroformer.SpikeVidUtils import trial_df, get_df_visnav, make_intervals


if behavior is True:
    behavior = pd.DataFrame({k: data[k] for k in behavior_vars + ['t']})
    # rename t to time
    behavior = behavior.rename(columns={'t': 'Time'}) if behavior is not None else None
    behavior['Interval'] = make_intervals(behavior, window)
    behavior['Interval_2'] = make_intervals(behavior, window_prev)

    # prepare speed variables
    behavior['speed'] = behavior['speed'].apply(lambda x: round_n(x, dt_speed))
    dt_range_speed = behavior['speed'].min(), behavior['speed'].max()
    dt_range_speed = np.arange(dt_range_speed[0], dt_range_speed[1] + dt_speed, dt_speed)
    n_behavior = len(dt_range_speed)

    stoi_speed = { round_n(ch, dt_speed):i for i,ch in enumerate(dt_range_speed) }
    itos_speed = { i:round_n(ch, dt_speed) for i,ch in enumerate(dt_range_speed) }
    assert (window_behavior) % dt_vars < 1e-5, "window + window_prev must be divisible by dt_vars"
    samples_per_behavior = int((window + window_prev) // dt_vars)
    behavior_block_size = int((window + window_prev) // dt_vars) * (len(behavior.columns) - 1)
else:
    behavior = None
    behavior_vars = None
    behavior_block_size = 0
    samples_per_behavior = 0
    stoi_speed = None
    itos_speed = None
    dt_range_speed = None
    n_behavior = None

if predict_behavior:
    loss_bprop = ['behavior']
else:
    loss_bprop = None



# %%
from SpikeVidUtils import make_intervals

df['Interval'] = make_intervals(df, window)
df['real_interval'] = make_intervals(df, 0.05)
df['Interval_2'] = make_intervals(df, window_prev)
df = df.reset_index(drop=True)

max_window = max(window, window_prev)
dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
n_dt = [round(dt * n, 2) for n in range(dt_range)] + ['EOS'] + ['PAD']

# %%
from neuroformer.SpikeVidUtils import SpikeTimeVidData2

## resnet3d feats
n_frames = round(frame_window * 1/dt_frames)
# kernel_size = (n_frames, 4, 4)
kernel_size = [n_frames, 8, 8]
stride_size = kernel_size
padding_size = 0
n_embd = 256
n_embd_frames = mconf.n_embd_frames
frame_feats = stimulus if visual_stim else None
frame_block_size = 0
frame_feats = torch.tensor(stimulus, dtype=torch.float32)
conv_layer = True

prev_id_block_size = 55
id_block_size = 55   #
block_size = frame_block_size + id_block_size + prev_id_block_size
frame_memory = frame_window // dt_frames
window = window

neurons = sorted(list(set(df['ID'])))
id_stoi = { ch:i for i,ch in enumerate(neurons) }
id_itos = { i:ch for i,ch in enumerate(neurons) }

neurons = sorted(list(set(df['ID'].unique())))
trial_tokens = [f"Trial {n}" for n in df['Trial'].unique()]
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

# r_split_ft = np.arange(0, 1, 0.25)
r_split_ft = 0.1
finetune_trials = train_trials[:int(len(train_trials) * r_split_ft)]
finetune_data = df[df['Trial'].isin(finetune_trials)]

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
"""
image indexes

(140, 260)
(339, 424)
(500, 620)
(680, 840) 
(960, 1050)

"""
import tifffile

stim2_path = "./data/Combo3_V1AL/stimuli/Combined Stimuli 3-Movie2.tif"
stimulus_2 = tifffile.imread(stim2_path)

stim3_path = "./data/Combo3_V1AL/stimuli/Combined Stimuli 3-Movie3.tif"
stimulus_3 = tifffile.imread(stim3_path)

mouse_indexes = [(140, 260), (339, 424), (500, 620), (680, 840), (960, 1050)]
stimulus_mice = np.concatenate([stimulus_2[i[0]:i[1]] for i in mouse_indexes])
stimulus_control = np.concatenate([stimulus_3[i[0]:i[1]] for i in mouse_indexes])

# labels_mice = []

assert stimulus_mice.shape == stimulus_control.shape, "stimulus shapes must be equal"

control_labels = np.zeros(stimulus_control.shape[0])
mice_labels = np.ones(stimulus_mice.shape[0])

stimulus_task = np.concatenate([stimulus_control, stimulus_mice])
labels_task = np.concatenate([control_labels, mice_labels])

# %%
mouse_indexes_downsampled = [tuple(map(lambda x: x // 3, i)) for i in mouse_indexes]
stim_2_ds = stimulus[1]
stim_2_ds_mice = np.concatenate([stim_2_ds[i[0]:i[1]] for i in mouse_indexes_downsampled])
print(mouse_indexes_downsampled)

# %%
from neuroformer.SpikeVidUtils import get_interval_idx

mouse_indexes_intervals = [tuple(map(lambda x: get_interval_idx(x, 0.05), i)) for i in mouse_indexes_downsampled]

intervals_cls = np.concatenate([np.arange(i[0], i[1], window) for i in mouse_indexes_intervals])
train_trial_cls = range(21, 41)
test_trial_cls = range(41, 61)

train_interval_trial_cls = np.array(np.meshgrid(intervals_cls, train_trial_cls)).T.reshape(-1, 2)
test_interval_trial_cls = np.array(np.meshgrid(intervals_cls, test_trial_cls)).T.reshape(-1, 2)

# %%
# from neuroformer.SpikeVidUtils import get_interval_trials, get_interval_idx

# # get data for downstream objective
# train_trial_ds = range(20, 41)
# test_trial_ds = range(41, 61)
# mouse_indexes_intervals = [tuple(map(lambda x: get_interval_idx(x, 0.05), i)) for i in mouse_indexes_downsampled]
# print(mouse_indexes_intervals)
# # pick out intervals from df according to mouse_indexes_intervals

# def filter_by_range(x, range_tuples):
#     return any(start <= x <= end for start, end in range_tuples)

# data_cls = df[df['Interval'].apply(lambda x: filter_by_range(x, mouse_indexes_intervals))]
# train_data_cls = data_cls[(data_cls['Trial'].isin(train_trial_ds)) & (data_cls['Trial'].isin(train_data['Trial'].unique()))]
# test_data_cls = data_cls[(data_cls['Trial'].isin(test_trial_ds)) & (data_cls['Trial'].isin(test_data['Trial'].unique()))]

# train_intervals_cls = 

# %%
# n_frames = 20
# fig, ax = plt.subplots(10, 1, figsize=(200, n_frames))
# frame_idx = [i for i in range(10)]
# for i, f_idx in enumerate(frame_idx):
#     ax[i].imshow(stimulus[1, f_idx, :, :])
#     ax[i].set_title(f"Frame {f_idx}", fontsize=20)
#     ax[i].axis('off')

# stim_plot = stim_2_ds_mice
# n_frames = 10
# fig, ax = plt.subplots(1, n_frames, figsize=(200, 10))
# frame_idx = np.random.choice(range(stim_plot.shape[0]), n_frames)
# # frame_idx = [i for i in range(n_frames)]
# for i, f_idx in enumerate(frame_idx):
#     ax[i].imshow(stim_plot[f_idx, :, :])
#     ax[i].set_title(f"Frame {f_idx}", fontsize=20)
#     ax[i].axis('off')

# %%
from neuroformer.SpikeVidUtils import SpikeTimeVidData2

train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                  window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  dt_frames=dt_frames, intervals=train_interval_trial_cls, dataset=DATASET,
                                  behavior=behavior, behavior_vars=behavior_vars, dt_vars=dt_vars,
                                  behavior_block_size=behavior_block_size, samples_per_behavior=samples_per_behavior,
                                  window_behavior=window_behavior, predict_behavior=predict_behavior,
                                  stoi_speed=stoi_speed, itos_speed=itos_speed, dt_speed=dt_speed, labels=True)

test_dataset = train_dataset.copy(test_data, t=test_interval_trial_cls)

# if INFERENCE:
#     update_object(train_dataset, dconf)
#     update_object(test_dataset, dconf)
    
print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')

# %%
loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
iterable = iter(loader)
x, y = next(iterable)
for k in x.keys():
    print(k)
    print(k, x[k].shape)

# %%

layers = (mconf.n_state_layers, mconf.n_state_history_layers, mconf.n_stimulus_layers)   
max_epochs = 80
batch_size = round((32 * 2))
shuffle = True

title =  f'large_window_2'

if INFERENCE:
    model_path = glob.glob(os.path.join(base_path, '**.pt'), recursive=True)[0]
else:
    model_path = f"""./models/tensorboard/V1_AL/downstream/{title}/sparse_f:{mconf.sparse_topk_frame}_id:{mconf.sparse_topk_id}/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(stoi_dt.values())}_max{dt}_{layers}_{mconf.n_head}_{mconf.n_embd}.pt"""

model_conf = GPTConfig(train_dataset.population_size, block_size,    # frame_block_size
                        id_vocab_size=train_dataset.id_population_size,
                        frame_block_size=frame_block_size,
                        id_block_size=id_block_size,  # frame_block_size
                        prev_id_block_size=prev_id_block_size,
                        behavior_block_size=behavior_block_size,
                        sparse_mask=False, p_sparse=None, 
                        sparse_topk_frame=None, sparse_topk_id=None, sparse_topk_prev_id=None,
                        n_dt=len(n_dt),
                        class_weights=None,
                        pretrain=False,
                        n_state_layers=4, n_state_history_layers=4,
                        n_stimulus_layers=4, self_att_layers=0,
                        n_behavior_layers=0, predict_behavior=predict_behavior, n_behavior=n_behavior,
                        n_head=4, n_embd=n_embd, 
                        contrastive=False, clip_emb=1024, clip_temp=mconf.clip_temp,
                        conv_layer=conv_layer, kernel_size=kernel_size, stride_size=stride_size, padding_size=padding_size,
                        temp_emb=mconf.temp_emb, pos_emb=False,
                        id_drop=0.1, im_drop=0.1, b_drop=0.45,
                        window=window, window_prev=window_prev, frame_window=frame_window, dt=dt,
                        neurons=neurons, stoi_dt=stoi_dt, itos_dt=itos_dt, n_embd_frames=n_embd_frames,
                        ignore_index_id=stoi['PAD'], ignore_index_dt=stoi_dt['PAD'])  # 0.35


if INFERENCE:
    update_object(model_conf, mconf)
model = GPT(model_conf)
# model_weights = "models/tensorboard/visnav/behavior_predict/long_no_classification/finetune/window:0.05_prev:0.25/sparse_f:None_id:None/w:0.05_wp:0.25/6_Cont:False_window:0.05_f_window:0.2_df:0.005_blocksize:100_conv_True_shuffle:True_batch:224_sparse_(None_None)_blocksz446_pos_emb:False_temp_emb:True_drop:0.35_dt:True_2.0_52_max0.005_(8, 8, 8)_8_256.pt"
# if model_weights is not None:
#     model.load_state_dict(torch.load(model_weights), strict=False)

# %%
tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=1e-4, 
                    num_workers=4, lr_decay=True, patience=3, warmup_tokens=8e4, 
                    decay_weights=True, weight_decay=0.1, shuffle=shuffle,
                    final_tokens=len(train_dataset)*(id_block_size) * (max_epochs),
                    clip_norm=1.0, grad_norm_clip=1.0,
                    dataset='higher_order', mode='predict',
                    block_size=train_dataset.block_size,
                    id_block_size=train_dataset.id_block_size,
                    show_grads=False, plot_raster=False,
                    ckpt_path=model_path, no_pbar=False, 
                    dist=False, save_every=1000, loss_bprop=loss_bprop)

if not INFERENCE:
    trainer = Trainer(model, train_dataset, test_dataset, tconf, model_conf)
    trainer.train()
else:
    model_path = glob.glob(os.path.join(base_path, '**.pt'), recursive=True)[0]
    model.load_state_dict(torch.load(model_path), strict=False)

# %%
from neuroformer.modules import ClassifierWrapper

N_CLASSES = 2
classifier = ClassifierWrapper(model, 2)


# %%
loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
iterable = iter(loader)

# %%
# model.eval()
# # model.train()
# for i in range(1):
#     x, y = next(iterable)
#     x_1 = x['id'][0].detach().cpu().numpy()
#     x_pad = x['pad']
#     x_1 = x['id'][:len(x_1) - int(x_pad[0])]
#     preds, features, loss = model(x, y)

#     step_choices = random.sample(range(len(x_1)), min(5, len(x_1)))
#     fig, ax = plt.subplots(1, len(step_choices), figsize=(50, 5))
#     for step in step_choices:
#         step_preds = preds['id'][0, step].detach().cpu().numpy()
#         x_axis = np.arange(0, len(step_preds))
#         ax_step = ax[step]
#         ax_step.scatter(x_axis, step_preds)
#         ax_step.set_title(f"""{step}""", fontsize=20)

# %%
train_dataset.t.shape

# %%
x, y = next(iterable)
for k in x.keys():
    print(k)
    print(k, x[k].shape)

# %%
from neuroformer.utils import predict_raster_recursive_time_auto, process_predictions

PARALLEL = True
df_pred_path = None

model.load_state_dict(torch.load(model_path))

results_dict = dict()
df_pred = None if df_pred_path is None else pd.read_csv(df_pred_path)
df_true = None

top_p = 0.95
top_p_t = 0.95
temp = 1.0
temp_t = 1.0

trials = sorted(train_data['Trial'].unique())[::4]

if df_pred_path is None:
    from joblib import Parallel, delayed
    # Define a function to process each trial
    def process_trial(model, train_dataset, df, stoi, itos_dt, itos, window, window_prev, top_p, top_p_t, temp, temp_t, trial):
        print(f"-- No. {n} Trial: {trial} --")
        df_trial = df[df['Trial'] == trial]
        trial_dataset = train_dataset.copy(df_trial)
        results_trial = predict_raster_recursive_time_auto(model, trial_dataset, window, window_prev, stoi, itos_dt, itos=itos, 
                                                        sample=True, top_p=top_p, top_p_t=top_p_t, temp=temp, temp_t=temp_t, 
                                                        frame_end=0, get_dt=True, gpu=False, pred_dt=True, plot_probs=False)
        df_trial_pred, df_trial_true = process_predictions(results_trial, stoi, itos, window)
        print(f"pred: {df_trial_pred.shape}, true: {df_trial_true.shape}" )
        return df_trial_pred, df_trial_true

    if PARALLEL:
        # Process each trial in parallel
        results = Parallel(n_jobs=-1)(delayed(process_trial)(model, train_dataset, df, stoi, itos_dt, 
                                                            itos, window, window_prev, top_p, top_p_t, 
                                                            temp, temp_t, trial) for trial in trials)
    else:
        # Process each trial sequentially
        results = []
        for trial in trials:
            results.append(process_trial(model, train_dataset, df, stoi, itos_dt, 
                                            itos, window, window_prev, top_p, top_p_t, 
                                            temp, temp_t, trial))
    # Combine the results from each trial
    for n, (df_trial_pred, df_trial_true) in enumerate(results):   
        print(f"-- No. {n} Trial --")
        if df_pred is None:
            df_pred = df_trial_pred
            df_true = df_trial_true
        else:
            df_pred = pd.concat([df_pred, df_trial_pred])
            df_true = pd.concat([df_true, df_trial_true])

from neuroformer.analysis import compute_scores
df_true = df[df['Trial'].isin(trials)]
scores = compute_scores(df_true, df_pred)
print(scores)
print(f"len predL: {len(df_pred)}, len true: {len(df_true)}")

dir_name = os.path.dirname(model_path)
model_name = os.path.basename(model_path)
df_pred.to_csv(os.path.join(dir_name, F'df_pred_.csv'))

# %%
"""

Split data into full-stimulus trials

"""
dir_name = os.path.dirname(model_path)
df_pred = pd.read_csv(os.path.join(dir_name, F'df_pred_.csv'))

from neuroformer.analysis import get_rates_trial, calc_corr_psth, get_accuracy, compute_scores
from neuroformer.SpikeVidUtils import create_full_trial, set_intervals


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

n_1 = test_data['Trial'].unique()
n_2 = test_data['Trial'].unique() + 1

df_pred_full = create_full_trial(df_pred)
df_1 = create_full_trial(df, trials)
df_2 = create_full_trial(df, n_1)
df_3 = create_full_trial(df, n_2)

# sort by interval, trial
window_pred = 0.5
min_window = window_prev + window
df_pred_full = set_intervals(df_pred_full, window, window_prev, window_pred)
df_1 = set_intervals(df_1, window, window_prev, window_pred)
df_2 = set_intervals(df_2, window, window_prev, window_pred)
df_3 = set_intervals(df_3, window, window_prev, window_pred)

window_pred = window if window_pred is None else window_pred
intervals = np.array(sorted(set(df['Interval'].unique()) & set(df['Interval'].unique())))
labels = np.array([round(window_pred + window_pred*n, 2) for n in range(0, int(max(df_pred_full['Interval']) / window_pred))])
ids = sorted(set(df['ID'].unique()) & set(df['ID'].unique()))

# labels = sorted(set(df_pred_full['Interval'].unique()))
rates_pred = get_rates_trial(df_pred_full, labels)
rates_1 = get_rates_trial(df_1, labels)
rates_2 = get_rates_trial(df_2, labels)
rates_3 = get_rates_trial(df_3, labels)

neurons = df['ID'].unique()
top_corr_pred = calc_corr_psth(rates_pred, rates_1, neurons=neurons)
top_corr_real = calc_corr_psth(rates_1, rates_2, neurons=neurons)
top_corr_real_2 = calc_corr_psth(rates_1, rates_3, neurons=neurons)

# %%
"""

Evaluate results

"""


from neuroformer.visualize import set_plot_white, plot_distribution

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

# dir_name = os.path.dirname(model_path)
dir_name = model_path
model_name = os.path.basename(model_path)
save_dir = model_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df_pred.to_csv(os.path.join(save_dir, F'df_pred.csv'))

set_plot_white()
plt.figure(figsize=(10, 10), facecolor='white')
plt.title(f'PSTH Correlations (V1 + AL) bin {window_pred}', fontsize=25)
plt.ylabel('Count (n)', fontsize=25)
plt.xlabel('Pearson r', fontsize=25)
plt.hist(top_corr_real, label='real - real2', alpha=0.6)
# plt.hist(top_corr_real_2, label='real - real3', alpha=0.6)
plt.hist(top_corr_pred, label='real - simulated', alpha=0.6)
plt.legend(fontsize=20)
plt.savefig(os.path.join(save_dir, F'psth_corr_{title}.svg'))
plt.show()

plot_distribution(df_1, df_pred, save_path=os.path.join(save_dir, F'psth_dist_{title}.svg'))

total_scores = dict()
total_scores['real'] = scores
total_scores['pred'] = pred_scores

print(f"model: {title}")

# %%
loader = DataLoader(test_dataset, shuffle=False, pin_memory=False)
iterable = iter(test_dataset)


# %%
var_group = 'Interval'
int_trials = df.groupby([var_group, 'Trial']).size()
print(int_trials.mean())
# df.groupby(['Interval', 'Trial']).agg(['nunique'])
n_unique = len(df.groupby([var_group, 'Trial']).size())
df.groupby([var_group, 'Trial']).size().nlargest(int(0.2 * n_unique))
# df.groupby(['Interval_2', 'Trial']).size().mean()


# %%
# while iv < 1.95:
x, y = next(iterable)

T = len(x['id'])
P = x['pad']
T_prev = len(x['id_prev'])
P_prev = x['pad_prev'] - 4

T_y = len(y['id'])
P_y = x['pad']

iv = float(x['interval'])

xid = x['id'][: T - P]
xid = [itos[int(i)] for i in xid]
xdt = x['dt'][: T - P]

yid = y['id'][: T_y - P_y]
yid = [itos[int(i)] for i in yid]
ydt = y['dt'][: T - P]

xid_prev = x['id_prev'][: T_prev - P_prev]
xid_prev = [itos[int(i)] for i in xid_prev]

print(f"iv: {iv}, ix+window: {iv + window} pid: {x['pid']} cid: {x['cid']}")
print(f"x: {xid}")
print(f"xdt: {xdt}")
print(f"y: {yid}")
print(f"ydt: {ydt}")

print(f"xid_prev: {xid_prev}")

tdiff = 0
t_var = 'Time' # 'Interval'
int_var = 'cid'
# df[(df[t_var] >= iv - tdiff) & (df[t_var] <= iv + (window + tdiff)) & (df['Trial'] == int(x['trial']))]
# df[(df[t_var] >= float(x[int_var][0]) - tdiff) & (df[t_var] <= float(x[int_var][1] + tdiff)) & (df['Trial'] == int(x['trial']))]
df[(df[t_var] > float(x[int_var][0]) - tdiff) & (df[t_var] <= float(x['cid'][1] + tdiff)) & (df['Trial'] == int(x['trial']))]

# t_var = 'Time' # 'Interval'
# int_var = 'pid'
# df[(df[t_var] > round(float(x[int_var][0]), 2) - tdiff) & (df[t_var] <= round(float(x[int_var][1]), 2)) & (df['Trial'] == int(x['trial']))

# print(f"trial: {x['trial']}, pid: {x['pid']}, cid: {x['cid']}")

# plt.imshow(x['frames'][0, 0])

# %%
loader = DataLoader(test_dataset, shuffle=True, pin_memory=False)
iterable = iter(loader)

# %%
x, y = next(iterable)
preds, features, loss = model(x, y)

# %%
features.keys()

# %%
features['last_layer'].shape

# %%
preds, feats, loss = classifier(x)

# %%


