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

from neuroformer.model_neuroformer_2 import GPT, GPTConfig, load_model_and_tokenizer
from neuroformer.utils import get_attr
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils_2 import (set_seed, update_object, running_jupyter, 
                                 all_device, load_config, 
                                 dict_to_object, object_to_dict, recursive_print,
                                 create_modalities_dict, generate_spikes)
from neuroformer.visualize import set_plot_params
from neuroformer.SpikeVidUtils import make_intervals, round_n, SpikeTimeVidData2
from neuroformer.DataUtils import round_n, split_data_by_interval, Tokenizer
from neuroformer.datasets import load_visnav, load_V1AL

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

from neuroformer.default_args import DefaultArgs, parse_args

if running_jupyter(): # or __name__ == "__main__":
    print("Running in Jupyter")
    args = DefaultArgs()
    args.config = "./configs/NF_1.5/mconf.yaml"
    args.dataset = "medial"
else:
    print("Running in terminal")
    args = parse_args()

# SET SEED - VERY IMPORTANT
set_seed(args.seed)

print(f"CONTRASTIUVEEEEEEE {args.contrastive}")
print(f"VISUAL: {args.visual}")
print(f"PAST_STATE: {args.past_state}")


config, tokenizer, model = load_model_and_tokenizer(args.ckpt_path)

# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""

if args.dataset in ["lateral", "medial"]:
    data, intervals, train_intervals, \
    test_intervals, finetune_intervals, \
    callback = load_visnav(args.dataset, config, 
                           selection=config.selection if hasattr(config, "selection") else None)
elif args.dataset == "V1AL":
    data, intervals, train_intervals, \
    test_intervals, finetune_intervals, \
    callback = load_V1AL(config)

spikes = data['spikes']
stimulus = data['stimulus']

# %%
window = config.window.curr
window_prev = config.window.prev
dt = config.resolution.dt


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
import pickle

# %%
from neuroformer.DataUtils import NFDataloader

modalities = create_modalities_dict(data, config.modalities)
frames = {'feats': stimulus, 'callback': callback, 'window': config.window.frame, 'dt': config.resolution.dt}

train_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                             frames=frames, intervals=train_intervals, modalities=modalities)
test_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                            frames=frames, intervals=test_intervals, modalities=modalities)
finetune_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                                frames=frames, intervals=finetune_intervals, modalities=modalities)

    
# print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
iterable = iter(train_dataset)
x, y = next(iterable)
recursive_print(x)

# Update the config
config.id_vocab_size = tokenizer.ID_vocab_size

# Create a DataLoader
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
temp = 1.
temp_t = 1.
frame_end = 0
true_past = args.true_past
get_dt = True
gpu = True
pred_dt = True

# # Run the prediction function
# results_trial = generate_spikes(model, test_dataset, window, 
#                                 window_prev, tokenizer, 
#                                 sample=sample, top_p=top_p, top_p_t=top_p_t, 
#                                 temp=temp, temp_t=temp_t, frame_end=frame_end, 
#                                 true_past=true_past,
#                                 get_dt=get_dt, gpu=gpu, pred_dt=pred_dt,
#                                 plot_probs=False)

# # Create a filename string with the parameters
# filename = f"results_trial_sample-{sample}_top_p-{top_p}_top_p_t-{top_p_t}_temp-{temp}_temp_t-{temp_t}_frame_end-{frame_end}_true_past-{true_past}_get_dt-{get_dt}_gpu-{gpu}_pred_dt-{pred_dt}.pkl"

# # Save the results in a pickle file
# save_inference_path = os.path.join(CKPT_PATH, "inference")
# if not os.path.exists(save_inference_path):
#     os.makedirs(save_inference_path)

# print(f"Saving inference results in {os.path.join(save_inference_path, filename)}")

# with open(os.path.join(save_inference_path, filename), "wb") as f:
#     pickle.dump(results_trial, f)


# predict other modality
from neuroformer.utils_2 import predict_modality
# model.load_state_dict(torch.load(os.path.join(CKPT_PATH, f"_epoch_{speed}.pt")))

if args.predict_modes is not None:
    block_type = 'behavior'
    block_config = get_attr(config.modalities, block_type).variables
    for mode in args.predict_modes:
        mode_config = get_attr(block_config, mode)
        behavior_preds = predict_modality(model, finetune_dataset, modality=mode, 
                                          block_type=block_type, objective=get_attr(mode_config, 'objective'))
        # filename = f"behavior_preds_{mode}.csv"
        # save_inference_path = os.path.join(CKPT_PATH, "inference")
        # if not os.path.exists(save_inference_path):
        #     os.makedirs(save_inference_path)
        # print(f"Saving inference results in {os.path.join(save_inference_path, filename)}")
        # behavior_preds.to_csv(os.path.join(save_inference_path, filename))
