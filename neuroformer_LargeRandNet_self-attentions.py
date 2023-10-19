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


import json
# for i in {1..10}; do python3 -m gather_atts.py; done

from attention.LRN_attention import *


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

base_path = "./models/tensorboard/LRN/channel/2_window:0.5_prev:19.5/sparse_f:None_id:None/w:0.5_wp:19.5"

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
from SpikeVidUtils import SpikeTimeVidData2

# train_dat1aset = spikeTimeData(spikes, block_size, dt, stoi, itos)


train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                  window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  dt_frames=dt_frames, intervals=intervals)
test_dataset = SpikeTimeVidData2(test_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                 window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, 
                                 pred=False, window_prev=window_prev, frame_window=frame_window,
                                 dt_frames=dt_frames, intervals=intervals)

print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')


# %%
# from utils import get_class_weights
# class_weights = get_class_weights(train_dataset, stoi, stoi_dt)


# %%
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
                        data_size=train_dataset.size,
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

for k, v in model_conf.__dict__.items():
    if not hasattr(mconf, k):
        print(f"k: {k}, v: {v}")
        setattr(mconf, k, v)

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


if model_path in model_weights:
    load_weights = model_path
else:
    print(f'Loading weights from {os.path.basename(model_weights[0])}')
    load_weights = model_weights[0]

model.load_state_dict(torch.load(load_weights, map_location=torch.device('cpu')))

trials = test_data['Trial'].unique()[:4]

# %%
att_matrix = np.zeros((len(stoi.keys()), 1000))
neurons = sorted(list(set(df['ID'].unique())))

att_data = df[df['Trial'].isin([i for i in range(0, 20)])]
att_dataset = SpikeTimeVidData2(att_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                  window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                  pred=False, window_prev=window_prev, frame_window=frame_window,
                                  dt_frames=dt_frames, intervals=intervals)
loader = DataLoader(att_dataset, batch_size=32, shuffle=True, num_workers=1)
# model = model.to("cuda")
model.load_state_dict(torch.load(load_weights, map_location=torch.device('cpu')))

# %%
# x, y = next(iter(loader))

# _, _, _, = model(x, y)
# atts = get_atts(model)

# # %%
LAYER_KEY = "neural_state_block"

# print(atts['neural_state_block_0'].shape)
# print(atts['neural_stimulus_block_0'].shape)

# agg_atts = cat_atts(atts, LAYER_KEY)
# agg_atts = stack_atts(atts, LAYER_KEY)

# %%
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)
model.zero_grad()
pbar = tqdm(loader)

model.eval()
att_matrix = np.zeros((1003, 1003))

grad_cond = False

for x, y in pbar:
    model.zero_grad()
    x, y = all_device((x, y), device)
    model.to(device)
    with torch.set_grad_enabled(grad_cond):
        _, _, _, = model(x, y) if grad_cond else model(x)
    model.cpu()
    x, y = all_device((x, y), "cpu")
    attentions = get_atts(model)
    if grad_cond:
        gradients = get_grads(model)
        attentions = gradcam(attentions, gradients)
    # x, y = all_device((x, y), "cpu")


    # att = accum_atts(attentions, key='neural_stimulus_block').sum(0)
    # att = cat_atts(attentions, layer_key='neural_stimulus_block')
    att = stack_atts(attentions, layer_key=LAYER_KEY)
    # att = att.max(-4)[0].min(-3)[0] 
    att = att.sum(-4).sum(-3)
    # att = att[:, -1,].min(-3)[0]

    # if len(att.size()) > 2:
    #     # flatten batch
    #     att = att.view(-1, 150)
    x_id = x['id']
    y_id = y['id']
    
    # neurons = [int(itos[int(n)]) if not isinstance(itos[int(n)], str) else 1001 for n in x_id]
    for batch in range(x_id.size(0)):
        neurons_x, neurons_y, indexes = [], [], []
        for n, (xid, yid) in enumerate(zip(x_id[batch], y_id[batch])):
            xid = itos[int(xid)]
            yid = itos[int(yid)]
            if not isinstance(xid, str) and not isinstance(yid, str):
                neurons_x.append(int(xid))
                neurons_y.append(int(yid))
                indexes.append(n)
        # if len(att) > 1:
        #     for neuron, index in zip(neurons_x, neurons_y, indexes):
        #         att_matrix[neuron][neurons_] += np.array(att[index][neuron])
        att_matrix[neurons_x, neurons_x] += np.array(att[batch][indexes, indexes])

    # clear gpu memory
    del x, y, attentions, att, x_id, y_id
    if grad_cond:
        del gradients
    model.zero_grad()
    torch.cuda.empty_cache()

# save att_matrix
att_path = os.path.join(base_path, "attentions")
if not os.path.exists(att_path):
    os.mkdir(att_path)
n_files = len(glob.glob(os.path.join(att_path, "*.npy")))
 
save_path = os.path.join(att_path, f"{n_files}_self-att_{grad_cond}_lastlayer.npy")
np.save(save_path, att_matrix)
print(f"Saved attention matrix to {save_path}")

# %%



