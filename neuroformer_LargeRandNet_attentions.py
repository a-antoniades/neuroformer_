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

from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils import set_seed


from scipy import io as scipyio
from scipy.special import softmax
import skimage
import skvideo.io
from neuroformer.utils import print_full
from scipy.ndimage import gaussian_filter, uniform_filter


import matplotlib.pyplot as plt
from neuroformer.visualize import *
set_plot_params()
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"


from neuroformer.model_neuroformer_ import GPT, GPTConfig, neuralGPTConfig
from neuroformer.trainer import Trainer, TrainerConfig

from attention.LRN_attention import *


import json
# for i in {1..10}; do python3 -m gather_atts.py; done
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--base_path', type=str, default=None)
    args.add_argument('--n_iter', type=int, default=None)
    args.add_argument('--model_path', type=str, default=None)
    return args.parse_args()

args = parse_args()

# %%
from neuroformer.prepare_data import load_LRN

df, stimulus = load_LRN()

# %%
# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# %%
from utils import set_seed
n_seed = 25
set_seed(n_seed)

# %%
# load config files

import yaml

# base_path = "./models/tensorboard/LRN/channel/2_window:0.5_prev:19.5/sparse_f:None_id:None/w:0.5_wp:19.5"
base_path = args.base_path

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
# df = pd.read_csv(parent_path + "code/data/OneCombo3/Combo3_all_stim.csv")
w_mult = 3
frame_window = dconf.frame_window
window = dconf.window
window_prev = dconf.window_prev
dt = dconf.dt
dt_frames = dconf.dt_frames
# p_window = window / (window + window_prev)
# intervals = np.load(os.path.join(base_path, "intervals.npy"))
intervals = None


from SpikeVidUtils import make_intervals

df['real_interval'] = make_intervals(df, dt)
df['Interval'] = make_intervals(df, window)
df['Interval_2'] = make_intervals(df, window_prev)
df = df.reset_index(drop=True)

# n_dt = sorted((df['Interval_dt'].unique()).round(2)) 
max_window = max(window, window_prev)
dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
n_dt = [round(dt * n, 2) for n in range(dt_range)] + ['EOS'] + ['PAD']


# %%
int_trials = df.groupby(['Interval', 'Trial']).size()
print(int_trials.mean())
# df.groupby(['Interval', 'Trial']).agg(['nunique'])model_path
# var_group = 'Interval_2'
# n_unique = len(df.groupby([var_group, 'Trial']).size())
# df.groupby([var_group, 'Trial']).size().nlargest(int(0.2 * n_unique))
# df.groupby(['Interval_2', 'Trial']).size().mean()



# %%
from SpikeVidUtils import SpikeTimeVidData2

## resnet3d feats
n_embd = mconf.n_embd
frame_feats = torch.tensor(stimulus, dtype=torch.float32).transpose(1, 0)
frame_block_size = mconf.frame_block_size  # math.ceil(frame_feats.shape[-1] * frame_window)
n_embd_frames = mconf.n_embd_frames

prev_id_block_size = mconf.prev_id_block_size    # math.ceil(frame_block_size * (1 - p_window))
id_block_size = mconf.id_block_size           # math.ceil(frame_block_size * p_window)
block_size = frame_block_size + id_block_size + prev_id_block_size # frame_block_size * 2  # small window for faster training
frame_memory = dconf.frame_memory   # how many frames back does model see

neurons = sorted(list(set(df['ID'])))
id_stoi = { ch:i for i,ch in enumerate(neurons) }
id_itos = { i:ch for i,ch in enumerate(neurons) }

# translate neural embeddings to separate them from ID embeddings
neurons = sorted(list(set(df['ID'].unique())))
trial_tokens = [f"Trial {n}" for n in df['Trial'].unique()]
feat_encodings = neurons + ['SOS'] + ['EOS'] + ['PAD']  # + pixels 
stoi = { ch:i for i,ch in enumerate(feat_encodings) }
itos = { i:ch for i,ch in enumerate(feat_encodings) }
stoi_dt = { ch:i for i,ch in enumerate(n_dt) }
itos_dt = { i:ch for i,ch in enumerate(n_dt) }

# %%
r_split = 0.8
train_trials = sorted(df['Trial'].unique())[:int(len(df['Trial'].unique()) * r_split)]
train_data = df[df['Trial'].isin(train_trials)]
test_data = df[~df['Trial'].isin(train_trials)]

# %%
from neuroformer.SpikeVidUtils import SpikeTimeVidData2
from neuroformer.utils import update_object 

# train_dat1aset = spikeTimeData(spikes, block_size, dt, stoi, itos)


train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                  window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  dt_frames=dt_frames, intervals=intervals)
test_dataset = train_dataset.copy(test_data)

update_object(train_dataset, dconf)
update_object(test_dataset, dconf)

print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')


# %%
# from utils import get_class_weights
# class_weights = get_class_weights(train_dataset, stoi, stoi_dt)


e# %%
from model_neuroformer_LRN import GPT, GPTConfig
# initialize config class and model (holds hyperparameters)
   
conv_layer = False
model_conf = GPTConfig(train_dataset.population_size, block_size,    # frame_block_size
                        id_vocab_size=train_dataset.id_population_size,
                        frame_block_size=frame_block_size,
                        id_block_size=id_block_size,  # frame_block_size
                        prev_id_block_size=prev_id_block_size,
                        sparse_mask=False, p_sparse=0.25, 
                        sparse_topk_frame=None, sparse_topk_id=None, sparse_topk_prev_id=None,
                        n_dt=len(n_dt),
                        class_weights=None,
                        pretrain=False,
                        n_state_layers=mconf.n_state_layers, n_state_history_layers=mconf.n_state_history_layers,
                        n_stimulus_layers=mconf.n_stimulus_layers, self_att_layers=mconf.self_att_layers,
                        n_head=mconf.n_head, n_embd=mconf.n_embd, 
                        contrastive=True, clip_emb=1024, clip_temp=0.5,
                        temp_emb=True, pos_emb=False,
                        id_drop=0.35, im_drop=0.35,
                        window=window, window_prev=window_prev, frame_window=frame_window, dt=dt,
                        n_embd_frames=n_embd_frames, dataset=None,
                        ignore_index_id=stoi['PAD'], ignore_index_dt=stoi_dt['PAD'])  # 0.35

update_object(model_conf, mconf)
model = GPT(mconf)

# %%
layers = (mconf.n_state_layers, mconf.n_state_history_layers, mconf.n_stimulus_layers)
max_epochs = 300
batch_size = round((14))
shuffle = True

weighted = True if mconf.class_weights is not None else False
title =  f'window:{window}_prev:{window_prev}_smooth'
model_path = f"""./models/tensorboard/LRN/ignore_index/2_{title}/sparse_f:{mconf.sparse_topk_frame}_id:{mconf.sparse_topk_id}/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(stoi_dt.values())}_max{dt}_{layers}_{mconf.n_head}_{mconf.n_embd}.pt"""

# model_path = "/data5/antonis/neuroformer/models/tensorboard/LRN/channel/window:0.5_prev:19.5_smooth/sparse_f:None_id:None/w:0.5_wp:19.5/6_Cont:True_window:0.5_f_window:20_df:0.1_blocksize:150_conv_False_shuffle:True_batch:12_sparse_(None_None)_blocksz1150_pos_emb:False_temp_emb:True_drop:0.35_dt:True_2.0_197_max0.1_(8, 8, 8)_8_256.pt"
# if os.path.exists(model_path):
#     print(f"Loading model from {model_path}")
#     model.load_state_dict(torch.load(model_path))
# else:
#     print(f"Model not found at {model_path}")
#     raise FileNotFoundError

tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=1e-4, 
                    num_workers=4, lr_decay=False, patience=3, warmup_tokens=8e7, 
                    decay_weights=True, weight_decay=0.2, shuffle=shuffle,
                    final_tokens=len(train_dataset)*(id_block_size) * (max_epochs),
                    clip_norm=1.0, grad_norm_clip=1.0,
                    dataset='higher_order', mode='predict',
                    block_size=train_dataset.block_size,
                    id_block_size=train_dataset.id_block_size,
                    show_grads=False, plot_raster=False,
                    ckpt_path=model_path, no_pbar=False, 
                    dist=True, save_every=1000)

# trainer = Trainer(model, train_dataset, test_dataset, tconf, mconf)
# trainer.train()

# %%
loader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=4, pin_memory=True)
iterable = iter(loader)

# %%
x, y = next(iterable)

# %%
"""

RUN SIMULATION

"""

from utils import *
from IPython.utils import io
# top_p=0.25, top_p_t=0.9, temp=2.

model_weights = glob.glob(os.path.join(base_path, '**/**.pt'), recursive=True)
model_weights = sorted(model_weights, key=os.path.getmtime, reverse=True)
assert len(model_weights) > 0, "No model weights found"

if args.model_path is not None:
    load_weights = args.model_path
elif model_path in model_weights:
    load_weights = model_path
else:
    load_weights = model_weights[0]

print(f'Loading weights from {load_weights}')
model.load_state_dict(torch.load(load_weights, map_location=torch.device('cpu')))

trials = test_data['Trial'].unique()[:4]

# %%
preds, features, loss = model(x, y)
# model.neural_visual_transformer.neural_state_blocks[0].attn.att.shape

# %%
features = {}

# helper function for hook
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach().cpu()
    return hook

grads = {}
def get_grads(name):
    def hook(model, input, output):
        grads[name] = output.detach().cpu()
    return hook

"""
register forward hooks for all multimodal transformer layers
so that the features are saved after every forward pass
"""

for n, mod in enumerate(model.neural_visual_transformer.neural_state_blocks):
    mod.register_forward_hook(get_features(f'neural_state_block_{n}'))

for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_blocks):
    mod.register_forward_hook(get_features(f'neural_state_history_block_{n}'))

for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_self_attention):
    mod.register_forward_hook(get_features(f'neural_state_history_self_attention_{n}'))

for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
    mod.register_forward_hook(get_features(f'neural_state_stimulus_block_{n}'))

for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
    mod.attn.attn_drop.register_full_backward_hook(get_grads(f'neural_state_stimulus_block_{n}'))

# %%
"""
do a forward pass and save the features
"""

x, y  = next(iterable)

preds = []
feats = []

features = {}

with torch.no_grad():
    logits, feats, loss = model(x, y)
    for n, mod in enumerate(model.neural_visual_transformer.neural_state_blocks):
        preds.append(features[f'neural_state_block_{n}'])
    for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_blocks):
        preds.append(features[f'neural_state_history_block_{n}'])
    for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_self_attention):
        preds.append(features[f'neural_state_history_self_attention_{n}'])
    for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
        preds.append(features[f'neural_state_stimulus_block_{n}'])

# %%
model.neural_visual_transformer.neural_state_stimulus_blocks[0].attn.attn_drop

# %%
df['Trial'].max()

# %%
att_matrix = np.zeros((len(stoi.keys()), 1000))
neurons = sorted(list(set(df['ID'].unique())))

att_data = df
att_dataset = SpikeTimeVidData2(att_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                  window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  dt_frames=dt_frames, intervals=intervals)
loader = DataLoader(att_dataset, batch_size=64, shuffle=True, num_workers=4)
# model = model.to("cuda")
model.load_state_dict(torch.load(load_weights, map_location=torch.device('cpu')))
model = model.cpu()

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"

# %%
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# device = "cpu"
model.to(device)
model.zero_grad()
n_iter = args.n_iter if args.n_iter is not None else len(loader)
pbar = tqdm(loader, total=n_iter)

model.eval()
att_matrix = np.zeros((1000, 1000))

counter = 0
grad_cond = False
for x, y in pbar:
    model.zero_grad()
    x, y = all_device((x, y), device)
    model.to(device)
    with torch.set_grad_enabled(grad_cond):
        _, _, _, = model(x, y)
    # model.cpu()
    x, y = all_device((x, y), "cpu")
    attentions = get_atts(model)
    if grad_cond:
        gradients = get_grads(model)
        attentions = gradcam(attentions, gradients)
    
    att = stack_atts(attentions, layer_key='neural_stimulus_block')
    # att = att.max(-4)[0].min(-3)[0] 
    att = att.mean(-4).mean(-3)
    # att = att[:, -1,].min(-3)[0]

    if len(att.size()) > 2:
        # flatten batch
        n_time_steps = att.shape[-1] // 1000
        att = att.view(-1, n_time_steps, 1000).mean(-2)
    x_id = x['id'].flatten()
    y_id = y['id'].flatten()

    assert len(x_id) == len(att) == len(y_id), "lengths don't match"
    
    # neurons = [int(itos[int(n)]) if not isinstance(itos[int(n)], str) else 1001 for n in x_id]
    neurons, indexes = [], []
    for n, n_id in enumerate(x_id):
        n_id = itos[int(n_id)]
        if not isinstance(n_id, str):
            neurons.append(int(n_id))
            indexes.append(n)
    if len(att) > 1:
        for neuron, index in zip(neurons, indexes):
            att_matrix[neuron] += np.array(att[index])
    # clear gpu memory
    del x, y, attentions, att, x_id, y_id, neurons
    if grad_cond:
        del gradients
    model.zero_grad()
    torch.cuda.empty_cache()
    counter += 1
    if counter >= n_iter:
        break

# save att_matrix
att_path = os.path.join(base_path, "attentions")
if not os.path.exists(att_path):
    os.mkdir(att_path)
n_files = len(glob.glob(os.path.join(att_path, "*.npy")))

save_path = os.path.join(att_path, f"{n_files}_att_matrix_gradcam_{grad_cond}_lastlayer.npy")
np.save(save_path, att_matrix)
print(f"Saved attention matrix to {save_path}")

# %%
# # os.path.join(att_path, f"{n_files}_{grad_cond}_att_matrix_gradcam.npy")

# # %%
# loader = DataLoader(att_dataset, batch_size=1, shuffle=True, num_workers=2)
# iterable = iter(loader)

# # %%
# x, y = next(iterable)
# _, _, _, = model(x, y)
# attentions = get_atts(model)
# att = accum_atts(attentions, key='neural_stimulus_block')

# # %%
# def gradcam(atts, grads):
#     common_keys = set(atts.keys()).intersection(set(grads.keys()))
#     for key in common_keys:
#         atts[key] = atts[key] * grads[key].clamp(min=0)
#     return atts

# # %%
# att_vis = accum_atts(attentions, key='neural_stimulus_block').mean(0)

# # %%
# from visualize import set_plot_white
# set_plot_white()

# loader = DataLoader(att_dataset, batch_size=1, shuffle=True, num_workers=2)
# iterable = iter(loader)

# last_layer = f"neural_stimulus_block_{n}".format(n=mconf.n_stimulus_layers-1)
# n_neurons = 10
# ncols = 5
# nrows = n_neurons // ncols

# plt.figure(figsize=(40, (20) * (n_neurons // 20)))

# model.eval()
# model.to("cpu")
# n_idx = 3
# counter = 0
# pbar = tqdm(range(n_neurons))
# while pbar:
#     x, y = next(iterable)
#     model.zero_grad()
#     with torch.set_grad_enabled(True):
#         _, _, _, = model(x, y)
#     attentions = get_atts(model)
#     gradients = get_grads(model)
#     attentions = gradcam(attentions, gradients)

#     att_vis = attentions[last_layer].min(dim=1)[0]
    
#     # # x, y = all_device((x, y), "cpu")
#     # att_vis = accum_atts(attentions, key='neural_stimulus_block').view(-1, 1000)
#     # # att_id, att_vis_grad = interpret(x, y, model)
#     # # att_vis_grad = reshape_attentions(att_vis)

#     # for n_idx in range(att_vis.shape[0]):
#         # n_idx = 1
#     try:
#         neuron_x = int(itos[int(x['id'].flatten()[n_idx])])
#         neuron_y = int(itos[int(y['id'].flatten()[n_idx])])
#         counter += 1
#     except:
#         continue

#     lw = 10
#     fs = 15
#     plt.subplot(nrows, ncols, counter)
#     plt.grid()
#     plt.title(f"x: {neuron_x}, y: {neuron_y}", fontsize=20)
#     plt.plot(att_vis[n_idx])
#     plt.axvline(x=neuron_x, color='b', label='x', linewidth=lw)
#     plt.axvline(x=neuron_y, color='g', label='y', linewidth=lw)

#     save_dir = os.path.join(base_path, "gradcam")
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     n_plots = glob.glob(os.path.join(save_dir, "**/**.png"))
#     plt.savefig(os.path.join(save_dir, f"{len(n_plots)}.png"))
#     # plt.plot(att_vis_grad[n_idx], color='purple', linestyle='--', label='grad')
#     # plt.xlabel("Channel", fontsize=fs)
#     # plt.ylabel("Attention", fontsize=fs)
#     if counter == 0:
#         plt.legend()

# # %%
# from attentionVis import interpret

# id_x = x['id'].flatten()[n_idx]
# neuron_x = int(itos[int(id_x)])

# R_id, R_id_vis = interpret(x, y, model)

# # %%
# accum_atts(R_id_vis, key='neural_stimulus_block')

# # %%
# def reshape_attentions(att_vis):
#     n_id_block, n_vis_block = att_vis.shape[-2], att_vis.shape[-1]
#     att_vis = att_vis.view(n_id_block, n_vis_block)
#     reshape_c = att_vis.shape[-1] // stimulus.shape[0]
#     assert att_vis.shape[-1] % stimulus.shape[0] == 0, "Attention shape does not match stimulus shape"
#     att_vis = att_vis.view(att_vis.shape[0], reshape_c, att_vis.shape[1] // reshape_c)
#     att_vis = att_vis.sum(-2)
#     return att_vis

# # %%
# reshape_attentions(R_id_vis).shape

# # %%

# # %%



