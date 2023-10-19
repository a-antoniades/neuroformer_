# %%

import glob
import os

import sys
import glob
import pickle
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

from neuroformer.model_neuroformer_2 import Neuroformer, NeuroformerConfig, load_model_and_tokenizer
from neuroformer.utils import get_attr
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils_2 import (set_seed, update_object, running_jupyter, 
                                 all_devicmodele, load_config, 
                                 dict_to_object, object_to_dict, recursive_print,
                                 create_modalities_dict, generate_spikes)
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
    parser.add_argument("--dataset", type=str, default="V1AL")
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

    ## inference
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint")
    parser.add_argument("--true_past", action="store_true", default=False, help="True past")
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



config, tokenizer, model = load_model_and_tokenizer(args.ckpt_path)

# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""

data = load_V1AL(config)
spikes = data['spikes']
stimulus = data['stimulus']

# %%
window = config.window.curr
window_prev = config.window.prev
dt = config.resolution.dt

# %%

import itertools

intervals = np.arange(0, 31, config.window.curr)
trials = list(set(data['spikes'].keys()))
combinations = np.array(list(itertools.product(intervals, trials)))
train_intervals, test_intervals, finetune_intervals = split_data_by_interval(combinations, r_split=0.8, r_split_ft=0.01)

test_intervals = test_intervals[:100]

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

# %%

# %%
from neuroformer.DataUtils import NFDataloader

frames = {'feats': stimulus, 'callback': combo3_V1AL_callback, 'window': config.window.frame, 'dt': config.resolution.dt}
modalities = create_modalities_dict(data, config.modalities) if get_attr(config, 'modalities', None) else None

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

loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
iterable = iter(loader)
x, y = next(iterable)
recursive_print(y)
preds, features, loss = model(x, y)

# Set training parameters
MAX_EPOCHS = 300
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

# CKPT_PATH = f"/share/edc/home/antonis/neuroformer/models/NF.15/Visnav_VR_Expt/{DATASET}/{model_name}/1_new/{str(config.layers)}/{str(config.window)}/{SEED}"
# CKPT_PATH = CKPT_PATH.replace("namespace", "").replace(" ", "_")
CKPT_PATH = args.ckpt_path

# Define the parameters
sample = True
top_p = 0.95
top_p_t = 0.95
temp = 0.95
temp_t = 1.
frame_end = 0
true_past = args.true_past
get_dt = True
gpu = False
pred_dt = True

# Run the prediction function
results_trial = generate_spikes(model, test_dataset, window, 
                                window_prev, tokenizer, 
                                sample=sample, top_p=top_p, top_p_t=top_p_t, 
                                temp=temp, temp_t=temp_t, frame_end=frame_end, 
                                true_past=true_past,
                                get_dt=get_dt, gpu=gpu, pred_dt=pred_dt,
                                plot_probs=False)

# Create a filename string with the parameters
filename = f"results_trial_sample-{sample}_top_p-{top_p}_top_p_t-{top_p_t}_temp-{temp}_temp_t-{temp_t}_frame_end-{frame_end}_true_past-{true_past}_get_dt-{get_dt}_gpu-{gpu}_pred_dt-{pred_dt}.pkl"

# Save the results in a pickle file
save_inference_path = os.path.join(CKPT_PATH, "inference")
if not os.path.exists(save_inference_path):
    os.makedirs(save_inference_path)

print(f"Saving inference results in {os.path.join(save_inference_path, filename)}")

with open(os.path.join(save_inference_path, filename), "wb") as f:
    pickle.dump(results_trial, f)

# %%
def check_model_on_gpu(model):
    return all(param.device.type == 'cuda' for param in model.parameters())

if check_model_on_gpu(model):
    print("All model parameters are on the GPU")
else:
    print("Not all model parameters are on the GPU")

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
x = all_device(x, device)
y = all_device(y, device)

features, preds, loss = model(x, y)
