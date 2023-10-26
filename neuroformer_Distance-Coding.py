# %%
import pathlib
import glob
import os
import json

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

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import DataLoader

import math

from neuroformer.model_neuroformer_ import GPT, GPTConfig
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils import set_seed, update_object, check_common_attrs, running_jupyter
from neuroformer.visualize import set_plot_params
from neuroformer.SpikeVidUtils import make_intervals, round_n, SpikeTimeVidData2
import gdown

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

import argparse
from neuroformer.SpikeVidUtils import round_n

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--infer", action="store_true", help="Inference mode")
    parser.add_argument("--train", action="store_true", default=False, help="Train mode")
    parser.add_argument("--dist", action="store_true", default=False, help="Distributed mode")
    parser.add_argument("--seed", type=int, default=25, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--rand_perm", action="store_true", default=False, help="Randomly permute the ID column")
    parser.add_argument("--mconf", type=str, default=None, help="Path to model config file")
    parser.add_argument("--eos_loss", action="store_true", default=False, help="Use EOS loss")
    parser.add_argument("--no_eos_dt", action="store_true", default=False, help="No EOS dt token")
    parser.add_argument("--downstream", action="store_true", default=False, help="Downstream task")
    parser.add_argument("--freeze_model", action="store_true", default=False, help="Freeze model")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="Distance-Coding")
    parser.add_argument("--behavior", action="store_true", default=False, help="Behavior task")
    parser.add_argument("--pred_behavior", action="store_true", default=False, help="Predict behavior")
    parser.add_argument("--past_state", action="store_true", default=False, help="Input past state")
    parser.add_argument("--visual", action="store_true", default=False, help="Visualize")
    parser.add_argument("--contrastive", action="store_true", default=False, help="Contrastive")
    parser.add_argument("--clip_loss", action="store_true", default=False, help="Clip loss")
    parser.add_argument("--clip_vars", nargs="+", default=['id','frames'], help="Clip variables")
    parser.add_argument("--class_weights", action="store_true", default=False, help="Class weights")
    parser.add_argument("--resample", action="store_true", default=False, help="Resample")
    return parser.parse_args()

# if running_jupyter(): # or __name__ == "__main__":
print("Running in Jupyter")
INFERENCE = False
DIST = False
SEED = 25
DOWNSTREAM = False
TITLE = None
RESUME = None
RAND_PERM = False
MCONF = None
EOS_LOSS = False
NO_EOS_DT = False
FREEZE_MODEL = False
TITLE = None
DATASET = "Distance-Coding"
BEHAVIOR = False
PREDICT_BEHAVIOR = False
VISUAL = True
PAST_STATE = True
CONTRASTIVE = False
CLIP_LOSS = True
CLIP_VARS = ['id','frames']
CLASS_WEIGHTS = False
RESAMPLE_DATA = False
# else:
    # print("Running in terminal")
    # args = parse_args()
    # INFERENCE = not args.train
    # DIST = args.dist
    # SEED = args.seed
    # DOWNSTREAM = args.downstream
    # TITLE = args.title
    # RESUME = args.resume
    # RAND_PERM = args.rand_perm
    # MCONF = args.mconf
    # EOS_LOSS = args.eos_loss
    # NO_EOS_DT = args.no_eos_dt
    # FREEZE_MODEL = args.freeze_model
    # DATASET = args.dataset
    # BEHAVIOR = args.behavior
    # PREDICT_BEHAVIOR = args.pred_behavior
    # VISUAL = args.visual
    # PAST_STATE = args.past_state
    # CONTRASTIVE = args.contrastive
    # CLIP_LOSS = args.clip_loss
    # CLIP_VARS = args.clip_vars
    # CLASS_WEIGHTS = args.class_weights
    # RESAMPLE_DATA = args.resample

# SET SEED - VERY IMPORTANT
set_seed(SEED)

print(f"CONTRASTIUVEEEEEEE {CONTRASTIVE}")
print(f"VISUAL: {VISUAL}")
print(f"PAST_STATE: {PAST_STATE}")


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""

from neuroformer.prepare_data import DataLinks
from DataUtils import round_n

spikes_path = "data/Distance-Coding/neuroformer/spikerates.npy"
distance_path = "data/Distance-Coding/neuroformer/distance_array.npy"

spikes = np.load(spikes_path)
distance = np.round(np.load(distance_path), 3)

frame_feats = None

# %%
cell_numbers_path = "./data/Distance-Coding/iscell.npy"
cell_numbers = np.load(cell_numbers_path)

# %%
# load config files
import yaml

# base_path = "configs/visnav/predict_behavior"
if MCONF is not None:
    base_path = os.path.dirname(MCONF)
elif RESUME is not None:
    base_path = os.path.dirname(RESUME)
else:
    # base_path = "./configs/Combo3_V1AL/kernel_size/wave_emb/01second-noselfatt/01second-noselfatt_small/"
    base_path = None
    
if base_path is not None:
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

    # set attrs that are not equal
    common_attrs = check_common_attrs(mconf, tconf, dconf)
    print(f"Common attributes: {common_attrs}")

else:
    mconf = False

# %%
frame_window = 0.5
window = 0.01
window_prev = 0.01
window_behavior = window
dt = 0.01
dt_frames = 0.01
dt_vars = 0.01
intervals = None

# randomly permute 'id' column
if RAND_PERM:
    df['ID'] = df['ID'].sample(frac=1, random_state=25).reset_index(drop=True)

# %%
## resnet3d feats
from neuroformer.DataUtils import split_data_by_interval

intervals = np.arange(0, spikes.shape[1] * dt, window)
train_intervals, test_intervals, finetune_intervals = split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.1)

from neuroformer.DataUtils import Tokenizer

# intervals = np.array([round_n(t, window) for t in np.arange(0)])
id_block_size = 4000
prev_id_block_size = 4000
frame_block_size = 0

# make sure intervals = same size as distance

min_shape = min(intervals.shape[0], distance.shape[0])
intervals = intervals[:min_shape]
distance = distance[:min_shape]

print(f"intervals.shape: {intervals.shape}")
print(f"distance.shape: {distance.shape}")

# -------- #


spikes_dict = {
    "ID": spikes,
    "Interval": intervals,
    "dt": dt,
    "id_block_size": id_block_size,
    "prev_id_block_size": prev_id_block_size,
    "frame_block_size": frame_window,
    "window": window,
    "window_prev": window_prev,
}


max_window = max(window, window_prev)
dt_range = math.ceil(max_window / dt) + 1
n_dt = [round(dt * n, 2) for n in range(dt_range)]

token_types = {
    'ID': {'tokens': list(np.arange(0, spikes.shape[0]))},
    'dt': {'tokens': n_dt, 'resolution': dt},
    'distance': {'tokens': list(set(distance)), 'resolution': 0.001},
}

tokenizer = Tokenizer(token_types, max_window, dt)

# %%
dt_var_distance = 0.01

""" structure:
{
    type_of_modality:
        {name of modality: {'data':data, 'dt': dt, 'predict': True/False},
        ...
        }
    ...
}
"""
modalities = {
    'all': 
            {'distance': 
                {'data': distance, 'dt': dt_var_distance, 'predict': True}
            },
}

# %%
for modality_type, modality in modalities.items():
    for variable_type, variable in modality.items():
        print(variable_type, variable)

# %%
# # %%
# var_group = 'Interval'
# int_trials = df.groupby([var_group, 'Trial']).size()
# print(int_trials.mean())
# # df.groupby(['Interval', 'Trial']).agg(['nunique'])
# n_unique = len(df.groupby([var_group, 'Trial']).size())
# df.groupby([var_group, 'Trial']).size().nlargest(int(0.1 * n_unique))
# # df.groupby(['Interval_2', 'Trial']).size().mean()

# %%
from neuroformer.DataUtils import NFDataloader

train_dataset = NFDataloader(spikes_dict, tokenizer, frame_feats,
                             dataset=DATASET, intervals=train_intervals, modalities=modalities)
# update_object(train_dataset, dconf)
# train_dataset = train_dataset.copy(spikes, train_intervals, resample_data=False)
test_dataset = train_dataset.copy(spikes, test_intervals, resample_data=False)
# finetune_dataset = train_dataset.copy(spikes, finetune_intervals, resample_data=False)
    
# print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
iterable = iter(train_dataset)
x, y = next(iterable)

# %%
def recursive_print(x, keys=None):
    if keys is None:
        keys = []
    if isinstance(x, dict):
        for key, value in x.items():
            recursive_print(value, keys + [key])
    elif isinstance(x, torch.Tensor):
        print("_".join(keys), x.shape, x.dtype)

# suppose iterable is your iterable object
x, y = next(iterable)

recursive_print(x)
recursive_print(y)


# %%
import yaml

# Function to load YAML configuration file
def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# Use the function
config_path = "./configs/NF_1.5/mconf.yaml"
config = load_config(config_path)  # replace 'config.yaml' with your file path

from neuroformer.utils import update_config

# update config
updated_config = update_config(config, modalities, tokenizer, x, y, 2)

from types import SimpleNamespace

def dict_to_object(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_object(v) for k, v in d.items()})
    else:
        return d

updated_dict_object = dict_to_object(updated_config)

config = updated_dict_object

# %%
from neuroformer.model_neuroformer import GPT, GPTConfig

config = updated_dict_object
config.id_vocab_size = tokenizer.ID_vocab_size


model = GPT(config, tokenizer)

# %%
loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)

iterable = iter(loader)

# %%
x, y = next(iterable)

recursive_print(y)

# %%
preds, features, loss = model(x, y)

# %%
MAX_EPOCHS = 2500
BATCH_SIZE = 8
SHUFFLE = True
CKPT_PATH = './models/Distance-Coding/first_run/'

from neuroformer.trainer import TrainerConfig, Trainer

tconf = TrainerConfig(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=7e-5, 
                    num_workers=4, lr_decay=True, patience=3, warmup_tokens=8e7, 
                    decay_weights=True, weight_decay=1.0, shuffle=SHUFFLE,
                    final_tokens=len(train_dataset)*(config.block_size.id) * (MAX_EPOCHS),
                    clip_norm=1.0, grad_norm_clip=1.0,
                    show_grads=False,
                    ckpt_path=CKPT_PATH, no_pbar=False, 
                    dist=DIST, save_every=0)

trainer = Trainer(model, train_dataset, test_dataset, tconf, config)


# %%
trainer.train()

# %%
distance.shape

# %%
intervals.shape

# %%
from tqdm.notebook import tqdm
from torch.utils.data.dataloader import default_collate

def my_collate_fn(batch):
    try:
        return default_collate(batch)
    except RuntimeError as e:
        print(f"There was an error with collating the batch: {str(e)}")
        for idx, item in enumerate(batch):
            print(f"Item {idx}: {item}")  # Or print whatever specific information you need
        raise e  # Re-raise the exception to stop the training


loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=my_collate_fn)
pbar = tqdm(loader, total=len(loader))
for x, y in pbar:
    continue

# %%



