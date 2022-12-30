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
%matplotlib inline
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

RESPONSE_PATH = "./data/Combo3_V1AL/Combo3_V1AL_response.csv"
STIMULUS_PATH = "./data/Combo3_V1AL/Combo3_V1AL_stimulus.pt"

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
window_prev = 0.2
dt = 0.05

from SpikeVidUtils import make_intervals

df['Interval'] = make_intervals(df, window)
# df['Interval_2'] = make_intervals(df, window_prev)
# df['Interval_dt'] = make_intervals(df, dt)
# df['Interval_dt'] = (df['Interval_dt'] - df['Interval'] + window).round(3)
df = df.reset_index(drop=True)

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
from SpikeVidUtils import SpikeTimeVidData2

## qv-vae feats
# frames = torch.load(parent_path + "code/data/SImNew3D/stimulus/vq-vae_code_feats-24-05-4x4x4.pt").numpy() + 2
# frame_feats = torch.load(parent_path + "code/data/SImNew3D/stimulus/vq-vae_embed_feats-24-05-4x4x4.pt").numpy()
# frame_block_size = frames.shape[-1] - 1

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
type(video_stack)

# %%
# df.groupby(['Trial', 'Interval_2']).size().nlargest(20)

# %%
# train_len = round(len(df)*(4/5))
# test_len = round(len(df) - train_len)

# train_data = df[:train_len]
# test_data = df[train_len:train_len + test_len].reset_index().drop(['index'], axis=1)

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
from SpikeVidUtils import SpikeTimeVidData2

# train_dat1aset = spikeTimeData(spikes, block_size, dt, stoi, itos)

train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev, frame_window=frame_window)
test_dataset = SpikeTimeVidData2(test_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev, frame_window=frame_window)
# dataset = SpikeTimeVidData(df, frames, frame_feats, block_size, frame_block_size, prev_id_block_size, window, frame_memory, stoi, itos)
# single_batch = SpikeTimeVidData(df[df['Trial'].isin([5])], None, block_size, frame_block_size, prev_id_block_size, window, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats)
small_dataset = SpikeTimeVidData2(small_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False, window_prev=window_prev, frame_window=frame_window)


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
                        n_state_layers=4, n_state_history_layers=4, n_stimulus_layers=8, self_att_layers=4,
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
max_epochs = 100
batch_size = 32 * 2
shuffle = True
model_path = f"models/tensorboard/V1_AL/test_2/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_sparse{mconf.sparse_mask}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(n_dt)}_max{dt}_{layers}_{mconf.n_head}_{mconf.n_embd}_nembframe{mconf.n_embd_frames}_{kernel_size}.pt"
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
# model.load_state_dict(torch.load("/Users/antonis/Downloads/[16, 17, 18, 19]_Cont_True_0.50.05_sparseFalse_conv_True_shuffle_True_batch_224_sparse_(None_None)_pos_emb_False_temp_emb_True_drop_0.2_dt_True_2.0_0.5_max0.05_(4, 0, 6)_2_256_nembframe64-2.pt", map_location='cpu'))

tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=1e-4, 
                    num_workers=4, lr_decay=True, patience=3, warmup_tokens=8e5, 
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

# %%
from omegaconf import OmegaConf

def obj_to_dict(obj):
    return {k: v for k, v in obj.__dict__.items() if not k.startswith('__')}

mconf_dict = obj_to_dict(tconf)
print(mconf_dict)
# d = OmegaConf.create(dict(a=np.float64(3.2).hex()))
# del_keys = ['stoi', 'itos', 'stoi_dt', 'itos_dt', 'neurons']
del_keys = ['neurons']
for k in del_keys:
    if k in mconf_dict:
        del mconf_dict[k]

for key in mconf_dict:
    if isinstance(mconf_dict[key], int):
        mconf_dict[key] = float(mconf_dict[key])
print(OmegaConf.to_yaml(mconf_dict))

mconf_yaml = OmegaConf.to_yaml(mconf_dict)


import tempfile
conf = mconf_dict

directory = "/local/home/antonis/neuroformer/configs"
filename = os.path.join("configs", "tconf.yaml")
with tempfile.NamedTemporaryFile() as fp:
    OmegaConf.save(config=conf, f=filename)
    loaded = OmegaConf.load(filename)
    assert conf == loaded


save_object(mconf_dict, os.path.join("configs", "mconf.json"))


# %%
mconf.kernel_size

# %%
mconf_dict
# %%
loaded.max_epochs
# %%
