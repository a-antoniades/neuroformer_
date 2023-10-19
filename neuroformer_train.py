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

from neuroformer.model_neuroformer_2 import GPT, GPTConfig
from neuroformer.utils import get_attr
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils_2 import (set_seed, update_object, running_jupyter, 
                                 all_device, load_config, 
                                 dict_to_object, object_to_dict, recursive_print,
                                 create_modalities_dict)
from neuroformer.visualize import set_plot_params
from neuroformer.SpikeVidUtils import make_intervals, round_n, SpikeTimeVidData2
from neuroformer.DataUtils import round_n, Tokenizer
from neuroformer.datasets import load_visnav, load_V1AL

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"
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
else:
    print("Running in terminal")
    args = parse_args()

# SET SEED - VERY IMPORTANT
set_seed(args.seed)

print(f"CONTRASTIUVEEEEEEE {args.contrastive}")
print(f"VISUAL: {args.visual}")
print(f"PAST_STATE: {args.past_state}")

# Use the function
if args.config is None:
    config_path = "./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml"
else:
    config_path = args.config
config = load_config(config_path)  # replace 'config.yaml' with your file path


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

""" 
 - see mconf.yaml "modalities" structure:

modalities:
  behavior:
    n_layers: 4
    window: 0.05
    variables:
      speed:
        data: speed
        dt: 0.05
        predict: true
        objective: regression
      phi:
        data: phi
        dt: 0.05
        predict: true
        objective: regression
      th:
        data: th
        dt: 0.05
        predict: true
        objective: regression


Modalities: any additional modalities other than spikes and frames
    Behavior: the name of the <modality type>
        Variables: the name of the <modality>
            Data: the data of the <modality> in shape (n_samples, n_features)
            dt: the time resolution of the <modality>, used to index n_samples
            Predict: whether to predict this modality or not.
                     If you set predict to false, then it will 
                     not be used as an input in the model,
                     but rather to be predicted as an output. 
            Objective: regression or classification

"""

frames = {'feats': stimulus, 'callback': callback, 'window': config.window.frame, 'dt': config.resolution.dt}


"""
See neuroformer.


"""
modalities = create_modalities_dict(data, config.modalities) if get_attr(config, 'modalities', None) else None


def configure_token_types(config, modalities):
    max_window = max(config.window.curr, config.window.prev)
    dt_range = math.ceil(max_window / dt) + 1
    n_dt = [round_n(x, dt) for x in np.arange(0, max_window + dt, dt)]

    token_types = {
        'ID': {'tokens': list(np.arange(0, data['spikes'].shape[0] if isinstance(data['spikes'], np.ndarray) \
                                    else data['spikes'][1].shape[0]))},
        'dt': {'tokens': n_dt, 'resolution': dt},
        **({
            modality: {
                'tokens': sorted(list(set(eval(modality)))),
                'resolution': details.get('resolution')
            }
            # if we have to classify the modality, 
            # then we need to tokenize it
            for modality, details in modalities.items()
            if details.get('predict', False) and details.get('objective', '') == 'classification'
        } if modalities is not None else {})
    }
    return 
tokenizer = Tokenizer(token_types, max_window, dt)


# %%
if modalities is not None:
    for modality_type, modality in modalities.items():
        for variable_type, variable in modality.items():
            print(variable_type, variable)


# %%
from neuroformer.DataUtils import NFDataloader

train_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                             frames=frames, intervals=train_intervals, modalities=modalities)
test_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                            frames=frames, intervals=test_intervals, modalities=modalities)
finetune_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                                frames=frames, intervals=finetune_intervals, modalities=modalities)

    
# print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
iterable = iter(train_dataset)
x, y = next(iterable)
print(x['id'])
print(x['dt'])
recursive_print(x)

# Update the config
config.id_vocab_size = tokenizer.ID_vocab_size
model = GPT(config, tokenizer)

# Create a DataLoader
loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
iterable = iter(loader)
x, y = next(iterable)
recursive_print(y)
preds, features, loss = model(x, y)

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

CKPT_PATH = f"/share/edc/home/antonis/neuroformer/models/NF.15/Visnav_VR_Expt/{args.dataset}/{model_name}/{args.title}/{str(config.layers)}/{args.seed}"
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
                          dist=args.dist, save_every=0, eval_every=5, min_eval_epoch=50,
                          use_wandb=True, wandb_project="neuroformer", 
                          wandb_group=f"1.5.1_visnav_{args.dataset}", wandb_name=args.title)

    trainer = Trainer(model, train_dataset, test_dataset, tconf, config)
    trainer.train()


# # %%
# from neuroformer.utils_2 import predict_modality
# import pickle

# # CKPT_PATH = "/share/edc/home/antonis/neuroformer/models/NF.15/Visnav_VR_Expt/medial/speed_regression/69/"
# CKPT_PATH = "./models/NF.15/Visnav_VR_Expt/lateral/speed_regression/69/"
# model.load_state_dict(torch.load(os.path.join(CKPT_PATH, "_epoch_speed.pt"), map_location=torch.device('cpu')))
# tokenizer = pickle.load(open(os.path.join(CKPT_PATH, "tokenizer.pkl"), 'rb'))

# # %%
# loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
# iterable = iter(loader)

# # %%
# # x, y = next(iterable)
# # recursive_print(y)
# # preds, features, loss = model(x, y)

# # print("Loss: ", loss['speed'])
# # print("Preds: ", preds['speed'])
# # print("True: ", y['modalities']['speed']['value'])



# # %%
# from neuroformer.utils_2 import predict_modality


# modality = 'speed'
# objective = modalities['all'][modality]['objective']
# behavior_preds = predict_modality(model, finetune_dataset, modality=modality, 
#                                   block_type='modalities', objective=objective)

# # save predictions
# behavior_preds.to_csv(f"{CKPT_PATH}/behavior_preds.csv")



# # %%
# from scipy.stats import pearsonr
# from neuroformer.visualize import set_plot_params
# from neuroformer.visualize import set_research_params

# # set_research_params()

# save_path = f"./_rebuttal/behavior/regression/{DATASET}/{model_name}"
# if not os.path.exists(save2_path):
#     os.makedirs(save_path)
# behavior_preds.to_csv(os.path.join(save_path, 'behavior_preds.csv'), index=False)

# x_true, y_true = behavior_preds['cum_interval'], behavior_preds['true']
# x_pred, y_pred = behavior_preds['cum_interval'], behavior_preds['modalities_speed_value']

# # pearson r
# r, p = pearsonr([float(y) for y in y_pred], [float(y) for y in y_true])

# # plot
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.scatter(y_true, y_pred, s=100, c='k', alpha=0.5)

# # get the current axis limits after plotting your data
# xlims = ax.get_xlim()
# ylims = ax.get_ylim()
# s_f = 0.8
# # the line of perfect prediction should span the minimum to the maximum of the current x and y limits
# combined_limits = [min(xlims[0], ylims[0]) * s_f, max(xlims[1], ylims[1]) * s_f]
# ax.plot(combined_limits, combined_limits, 'k--', color='red')

# ax.set_xlabel('True speed', fontsize=20)
# ax.set_ylabel('Predicted speed', fontsize=20)
# ax.set_title(f'{model_name}, Regression', fontsize=20)
# # add pearson r to figure
# ax.text(0.05, 0.9, 'r = {:.2f}'.format(r), fontsize=20, transform=ax.transAxes)
# # add p to figure
# ax.text(0.05, 0.8, 'p < 0.001'.format(p), fontsize=20, transform=ax.transAxes)

# # axis limits = [-1.5, 1.5]
# # ax.set_xlim(axis_limits)
# # ax.set_ylim(axis_limits)
# plt.savefig(os.path.join(save_path, 'regression_2.pdf'), dpi=300, bbox_inches='tight')


# # plot
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# ax.scatter(y_true, y_pred, c='k', alpha=0.5)

# # get the current axis limits after plotting your data
# xlims = ax.get_xlim()
# ylims = ax.get_ylim()
# s_f = 0.8
# # the line of perfect prediction should span the minimum to the maximum of the current x and y limits
# combined_limits = [min(xlims[0], ylims[0]) * s_f, max(xlims[1], ylims[1]) * s_f]
# ax.plot(combined_limits, combined_limits, 'k--', color='red')

# ax.set_xlabel('True speed',)
# ax.set_ylabel('Predicted speed',)
# ax.set_title(f'{model_name}, Regression',)
# # add pearson r to figure
# ax.text(0.05, 0.9, 'r = {:.2f}'.format(r), transform=ax.transAxes)
# # add p to figure
# ax.text(0.05, 0.8, 'p < 0.001'.format(p), transform=ax.transAxes)

# # axis limits = [-1.5, 1.5]
# # ax.set_xlim(axis_limits)
# # ax.set_ylim(axis_limits)
# plt.savefig(os.path.join(save_path, 'regression_2.pdf'), dpi=300, bbox_inches='tight')


# # %%
# plt.figure(figsize=(5, 2.5))
# x = np.arange(len(behavior_preds))
# plt.title(f'Speed Predictions, {model_name} Regression vs. True')
# plt.plot(x, y_true, c='r', label='True')
# plt.plot(x, y_pred, c='b', label='Regression')
# plt.xlabel('Time (0.05s)')
# plt.ylabel('Speed (z-scored)')
# plt.legend(loc='upper left', framealpha=0.9)
# plt.savefig(os.path.join(save_path, 'speed_preds.pdf'), bbox_inches='tight')


# plt.figure(figsize=(10, 5))
# x = np.arange(len(behavior_preds))
# plt.title(f'Speed Predictions, {model_name} Regression vs. True')
# plt.plot(x, y_true, c='r', label='True')
# plt.plot(x, y_pred, c='b', label='Regression')
# plt.xlabel('Time (0.05s)')
# plt.ylabel('Speed (z-scored)')
# plt.legend(loc='upper left', framealpha=0.9)
# plt.savefig(os.path.join(save_path, 'speed_preds_2.pdf'), bbox_inches='tight')

# # %%