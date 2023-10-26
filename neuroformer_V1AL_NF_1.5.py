# %%

import glob
import os

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

from neuroformer.model_neuroformer import Neuroformer, NeuroformerConfig, get_attr
from neuroformer.utils import get_attr
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils_2 import (set_seed, update_object, running_jupyter, 
                                 all_device, load_config, 
                                 dict_to_object, object_to_dict, recursive_print,
                                 create_modalities_dict)
from neuroformer.visualize import set_plot_params
from neuroformer.SpikeVidUtils import make_intervals, round_n, SpikeTimeVidData2
from neuroformer.DataUtils import round_n, split_data_by_interval, Tokenizer, combo3_V1AL_callback
from neuroformer.datasets import load_V1AL

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

import argparse
import wandb

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
    parser.add_argument("--loss_bprop", type=str, default=None, help="Loss type to backpropagate")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID")
    return parser.parse_args()

if running_jupyter(): # or __name__ == "__main__":
    print("Running in Jupyter")
    INFERENCE = False
    DIST = False
    SEED = 69
    DOWNSTREAM = False
    TITLE = None
    RESUME = None
    RAND_PERM = False
    MCONF = None
    EOS_LOSS = False
    NO_EOS_DT = False
    FREEZE_MODEL = False
    TITLE = None
    DATASET = "lateral"
    BEHAVIOR = False
    PREDICT_BEHAVIOR = False
    VISUAL = True
    PAST_STATE = True
    CONTRASTIVE = False
    CLIP_LOSS = True
    CLIP_VARS = ['id','frames']
    CLASS_WEIGHTS = False
    RESAMPLE_DATA = False
    LOSS_BPROP = None
    CONFIG = None
else:
    print("Running in terminal")
    args = parse_args()
    INFERENCE = not args.train
    DIST = args.dist
    SEED = args.seed
    DOWNSTREAM = args.downstream
    TITLE = args.title
    RESUME = args.resume
    RAND_PERM = args.rand_perm
    MCONF = args.mconf
    EOS_LOSS = args.eos_loss
    NO_EOS_DT = args.no_eos_dt
    FREEZE_MODEL = args.freeze_model
    DATASET = args.dataset
    BEHAVIOR = args.behavior
    PREDICT_BEHAVIOR = args.pred_behavior
    VISUAL = args.visual
    PAST_STATE = args.past_state
    CONTRASTIVE = args.contrastive
    CLIP_LOSS = args.clip_loss
    CLIP_VARS = args.clip_vars
    CLASS_WEIGHTS = args.class_weights
    RESAMPLE_DATA = args.resample
    LOSS_BPROP = args.loss_bprop
    CONFIG = args.config

# SET SEED - VERY IMPORTANT
set_seed(SEED)

print(f"CONTRASTIUVEEEEEEE {CONTRASTIVE}")
print(f"VISUAL: {VISUAL}")
print(f"PAST_STATE: {PAST_STATE}")

# %%
# Use the function
if CONFIG is None:
    # config_path = "./configs/NF_1.5/mconf.yaml"
    # config_path = "./configs/NF_1.5/VisNav_VR_Expt/gru2_only/mconf.yaml"
    # config_path = "./configs/NF_1.5/VisNav_VR_Expt/mlp_only/mconf.yaml"
    # config_path = "./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml"
    config_path = "./configs/Combo3_V1AL/NF_1.5/mconf.yaml"

else:
    config_path = CONFIG
config = load_config(config_path)  # replace 'config.yaml' with your file path

# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""

data, intervals, train_intervals, test_intervals, finetune_intervals = load_V1AL()
spikes = data['spikes']
stimulus = data['stimulus']
speed = None

# %%
window = config.window.curr
window_prev = config.window.prev
dt = config.resolution.dt


# from neuroformer.DataUtils import make_intervals
# df = pd.read_csv("./data/VisNav_VR_Expt/MedialVRDataset/df.csv")
# selection_1 = np.array(pd.read_csv("./data/VisNav_VR_Expt/MedialVRDataset/sel1.csv")).flatten()
# df = df[df['ID'].isin(selection_1)]
# df['Interval'] = make_intervals(df, window)
# df['Interval_2'] = make_intervals(df, window_prev)
# # df.groupby(['Interval', 'Trial']).size().plot.bar()
# # df.groupby(['Interval', 'Trial']).agg(['nunique'])model_path
# n_unique = len(df.groupby(['Interval', 'Trial']).size())
# print(df.groupby(['Interval', 'Trial']).size().nlargest(int(0.7 * n_unique)))
# print(df.groupby(['Interval_2', 'Trial']).size().nlargest(int(0.2 * n_unique)))


print(f"intervals.shape: {intervals.shape}")

# -------- #

spikes_dict = {
    "ID": data['spikes'],
    "Frames": data['stimulus'],
    "Interval": intervals,
    "dt": config.resolution.dt,
    "id_block_size": config.block_size.id,
    "prev_id_block_size": config.block_size.prev_id,
    "frame_block_size": config.block_size.frame,
    "window": config.window.curr,
    "window_prev": config.window.prev,
    "frame_window": config.window.frame,
}

""" structure:
{
    type_of_modality:
        {name of modality: {'data':data, 'dt': dt, 'predict': True/False},
        ...
        }
    ...
}
"""

frames = {'feats': stimulus, 'callback': combo3_V1AL_callback, 'window': config.window.frame, 'dt': config.resolution.dt}
modalities = create_modalities_dict(data, config.modalities) if get_attr(config, 'modalities', None) else None

max_window = max(config.window.curr, config.window.prev)
dt_range = math.ceil(max_window / dt) + 1
n_dt = [round_n(x, dt) for x in np.arange(0, max_window + dt, dt)]

token_types = {
    'ID': {'tokens': list(np.arange(0, data['spikes'][1].shape[0]))},
    'dt': {'tokens': n_dt, 'resolution': dt},
}
tokenizer = Tokenizer(token_types, max_window, dt)



# %%
from neuroformer.DataUtils import NFDataloader

train_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=DATASET, 
                             frames=frames, intervals=train_intervals, modalities=modalities)
test_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=DATASET, 
                            frames=frames, intervals=test_intervals, modalities=modalities)
finetune_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=DATASET, 
                                frames=frames, intervals=finetune_intervals, modalities=modalities)

    
# print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
iterable = iter(train_dataset)
x, y = next(iterable)
print(x['id'])
print(x['dt'])
recursive_print(x)

# update config
# updated_config = update_config(config, modalities, tokenizer, x, y, 2)
# updated_dict_object = dict_to_object(updated_config)
# config = updated_dict_object

# Update the config
config.id_vocab_size = tokenizer.ID_vocab_size
model = Neuroformer(config, tokenizer)

# Create a DataLoader
loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
iterable = iter(loader)
x, y = next(iterable)
recursive_print(y)
preds, features, loss = model(x, y)

# %%
# Set training parameters
MAX_EPOCHS = 200
BATCH_SIZE = 32 * 5
SHUFFLE = True

if config.gru_only:
    model_name = "GRU"
elif config.mlp_only:
    model_name = "MLP"
elif config.gru2_only:
    model_name = "GRU_2.0"
else:
    model_name = "Neuroformer"

CKPT_PATH = f"/share/edc/home/antonis/neuroformer/models/NF.15/Visnav_VR_Expt/{DATASET}/{model_name}/{TITLE}/{str(config.layers)}/{SEED}"
CKPT_PATH = CKPT_PATH.replace("namespace", "").replace(" ", "_")

if args.sweep_id is not None:
    from neuroformer.hparam_sweep import train_sweep
    print(f"-- SWEEP_ID -- {args.sweep_id}")
    wandb.agent(args.sweep_id, function=train_sweep)
else:
    # Create a TrainerConfig and Trainer
    tconf = TrainerConfig(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=1e-4, 
                          num_workers=16, lr_decay=True, patience=3, warmup_tokens=8e7, 
                          decay_weights=True, weight_decay=1.0, shuffle=SHUFFLE,
                          final_tokens=len(train_dataset)*(config.block_size.id) * (MAX_EPOCHS),
                          clip_norm=1.0, grad_norm_clip=1.0,
                          show_grads=False,
                          ckpt_path=CKPT_PATH, no_pbar=False, 
                          dist=DIST, save_every=0, eval_every=5, min_eval_epoch=50,
                          use_wandb=True, wandb_project="V1AL", wandb_group=f"1.5_V1AL")

    trainer = Trainer(model, train_dataset, test_dataset, tconf, config)
    trainer.train()

# %%



