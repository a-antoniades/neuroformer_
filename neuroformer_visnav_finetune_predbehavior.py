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

from trainer import Trainer, TrainerConfig
from utils import set_seed

from scipy import io as scipyio
from scipy.special import softmax
import skimage
import skvideo.io
from utils import print_full
from scipy.ndimage import gaussian_filter, uniform_filter

import matplotlib.pyplot as plt
from utils import *
from visualize import *
set_plot_params()
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

from model_neuroformer import GPT, GPTConfig, neuralGPTConfig
from trainer import Trainer, TrainerConfig

from SpikeVidUtils import round_n
from neuroformer.visualize import *
import argparse

set_seed(25)


# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", type=str, default="neuroformer", help="Model to use")
    parser.add_argument("--finetune", action="store_true", default=False, help="Finetune model")
    parser.add_argument("--ft_prop", type=float, default=0.1, help="Proportion of data to finetune on")
    parser.add_argument("--ft_base_path", type=str, default=None, help="Path to model to finetune")
    parser.add_argument("--base_path", type=str, default=None, help="Path to model config")

    # training args
    parser.add_argument("--train", action="store_true", default=False, help="Train model")

    # inference args
    parser.add_argument("--infer_response", action="store_true", default=False, help="Infer response")
    parser.add_argument("--infer_behavior", action="store_true", default=False, help="Infer behavior")

    return parser.parse_args()


def main(args):

    # %%
    data_dir = "./data/VisNav_VR_Expt"

    if not os.path.exists(data_dir):
        print("Downloading data...")
        import gdown
        url = "https://drive.google.com/drive/folders/117S-7NmbgrqjmjZ4QTNgoa-mx8R_yUso?usp=sharing"
        gdown.download_folder(id=url, quiet=False, use_cookies=False, output="data/")

    # %%
    # load config files
    import yaml

    # base_path = "configs/visnav/predict_behavior"
    if args.base_path is None:
        base_path = "models/tensorboard/visnav/behavior_predict/long_no_classification/window:0.05_prev:0.25/sparse_f:None_id:None/w:0.05_wp:0.25"
    else:
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
    import mat73
    import scipy

    data_path = "data/VisNav_VR_Expt/experiment_data.mat"
    data = mat73.loadmat(data_path)['neuroformer']
    print(data.keys())

    # %%
    frame_window = 0.2
    window = 0.05
    window_prev = 0.25
    window_behavior = window
    dt = 0.005
    dt_frames = 0.05
    dt_vars = 0.05
    dt_speed = 0.2
    intervals = None


    # %%
    ## choose modalities ##

    # behavior
    behavior = True
    # behavior_vars = ['t', 'eyerad', 'phi', 'speed', 'th']
    behavior_vars = ['speed']
    n_behavior = len(behavior_vars)
    predict_behavior = False
    # stimulus
    visual_stim = True

    # %%
    from neuroformer.SpikeVidUtils import trial_df, get_df_visnav, make_intervals

    stimulus = data['vid_sm']
    response = data['spiketimes']['spks']
    trial_data = data['trialsummary']
    # response = data_response['spiketime_sel2']['spks']

    print(data.keys())

    df = get_df_visnav(response, trial_data, dt_vars)
    # df = df[df['ID'].isin(neurons_sel1)].reset_index(drop=True)

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
        # loss_bprop = ['behavior', 'id']
        loss_bprop = None
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
    kernel_size = (n_frames, 5, 5)
    n_embd = 256
    n_embd_frames = 64
    frame_feats = stimulus if visual_stim else None
    frame_block_size = ((n_frames // kernel_size[0] * 30 * 100) // (n_embd_frames))
    frame_feats = torch.tensor(stimulus, dtype=torch.float32)
    conv_layer = True

    prev_id_block_size = 300
    id_block_size = 100   #
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
    finetune_trials = np.random.shuffle(test_data['Trial'].unique())
    finetune_trials = train_trials[:int(len(train_trials) * r_split_ft)]
    finetune_data = df[df['Trial'].isin(finetune_trials)]

    # %%
    from neuroformer.SpikeVidUtils import SpikeTimeVidData2


    train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                    window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                    pred=False, window_prev=window_prev, frame_window=frame_window,
                                    dt_frames=dt_frames, intervals=None, dataset='visnav',
                                    behavior=behavior, behavior_vars=behavior_vars, dt_vars=dt_vars,
                                    behavior_block_size=behavior_block_size, samples_per_behavior=samples_per_behavior,
                                    window_behavior=window_behavior, predict_behavior=predict_behavior,
                                    stoi_speed=stoi_speed, itos_speed=itos_speed, dt_speed=dt_speed)

    test_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                    window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                    pred=False, window_prev=window_prev, frame_window=frame_window,
                                    dt_frames=dt_frames, intervals=None, dataset='visnav',
                                    behavior=behavior, behavior_vars=behavior_vars, dt_vars=dt_vars,
                                    behavior_block_size=behavior_block_size, samples_per_behavior=samples_per_behavior,
                                    window_behavior=window_behavior, predict_behavior=predict_behavior,
                                    stoi_speed=stoi_speed, itos_speed=itos_speed, dt_speed=dt_speed)

    finetune_dataset = SpikeTimeVidData2(finetune_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                    window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                    pred=False, window_prev=window_prev, frame_window=frame_window,
                                    dt_frames=dt_frames, intervals=None, dataset='visnav',
                                    behavior=behavior, behavior_vars=behavior_vars, dt_vars=dt_vars,
                                    behavior_block_size=behavior_block_size, samples_per_behavior=samples_per_behavior,
                                    window_behavior=window_behavior, predict_behavior=predict_behavior,
                                    stoi_speed=stoi_speed, itos_speed=itos_speed, dt_speed=dt_speed)


    print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')

    # %%
    from neuroformer.utils import update_object

    layers = (mconf.n_state_layers, mconf.n_state_history_layers, mconf.n_stimulus_layers)   
    max_epochs = 500
    batch_size = round((16))
    shuffle = True

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
                            n_state_layers=mconf.n_state_layers, n_state_history_layers=mconf.n_state_history_layers,
                            n_stimulus_layers=mconf.n_stimulus_layers, self_att_layers=mconf.self_att_layers,
                            n_behavior_layers=mconf.n_behavior_layers, predict_behavior=predict_behavior, n_behavior=n_behavior,
                            n_head=mconf.n_head, n_embd=mconf.n_embd, 
                            contrastive=mconf.contrastive, clip_emb=1024, clip_temp=mconf.clip_temp,
                            conv_layer=conv_layer, kernel_size=kernel_size,
                            temp_emb=mconf.temp_emb, pos_emb=False,
                            id_drop=0.35, im_drop=0.35, b_drop=0.45,
                            window=window, window_prev=window_prev, frame_window=frame_window, dt=dt,
                            neurons=neurons, stoi_dt=stoi_dt, itos_dt=itos_dt, n_embd_frames=n_embd_frames,
                            ignore_index_id=stoi['PAD'], ignore_index_dt=stoi_dt['PAD'])  # 0.35
    

    update_object(model_conf, mconf)
    model = GPT(model_conf)

    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    iterable = iter(loader)

    x, y = next(iterable)
    # print(x['behavior'].shape, x['behavior_dt'].shape)
    for k in x.keys():
        print(k, x[k].shape)
    for k in y.keys():
        print(f"y: {k}, {y[k].shape}")

    preds, features, loss = model(x, y)

    if args.finetune:
        model_weights = "models/tensorboard/visnav/behavior_predict/long_no_classification/finetune/window:0.05_prev:0.25/sparse_f:None_id:None/w:0.05_wp:0.25/6_Cont:False_window:0.05_f_window:0.2_df:0.005_blocksize:100_conv_True_shuffle:True_batch:224_sparse_(None_None)_blocksz446_pos_emb:False_temp_emb:True_drop:0.35_dt:True_2.0_52_max0.005_(8, 8, 8)_8_256.pt"
        if model_weights is not None:
            model.load_state_dict(torch.load(model_weights), strict=False)
    elif args.ft_base_path:
        # find .pt file inside base_path
        model_weights = glob.glob(os.path.join(args.ft_base_path, '**.pt'))
        model_path = model_weights
        if len(model_weights) > 0:
            print("// Loading Inference Weights //")
            print(model_weights)
            model_weights = model_weights[0]
            model.load_state_dict(torch.load(os.path.join(args.ft_base_path, model_weights)), strict=False)
        else:
            raise ValueError(f'No .pt file found in {args.ft_base_path}')
    elif args.train: 
        print(f"// Pretraining, no loaded weights //")
        title =  f'window:{window}_prev:{window_prev}'
        model_weights = f"""./models/tensorboard/visnav/multi-contrastive/{model_conf.contrastive_vars}/no_pretraining/{title}/sparse_f:{mconf.sparse_topk_frame}_id:{mconf.sparse_topk_id}/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(stoi_dt.values())}_max{dt}_{layers}_{mconf.n_head}_{mconf.n_embd}.pt"""

        # %%
        tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=1e-4, 
                            num_workers=4, lr_decay=True, patience=3, warmup_tokens=8e7, 
                            decay_weights=True, weight_decay=1.0, shuffle=shuffle,
                            final_tokens=len(train_dataset)*(id_block_size) * (max_epochs),
                            clip_norm=1.0, grad_norm_clip=1.0,
                            dataset='higher_order', mode='predict',
                            block_size=train_dataset.block_size,
                            id_block_size=train_dataset.id_block_size,
                            show_grads=False, plot_raster=False,
                            ckpt_path=model_weights, no_pbar=False, 
                            dist=False, save_every=1000, loss_bprop=loss_bprop)
        
        trainer = Trainer(model, train_dataset, test_dataset, tconf, model_conf)
        trainer.train()
    else:
        model_path = glob.glob(os.path.join(args.base_path, '**.pt'))
        model.load_state_dict(torch.load(model_path[0]), strict=False)


# %%
    if args.infer_behavior:

        from neuroformer.utils import predict_behavior

        n_trials = 10
        no_ = 1
        # chosen_trials = test_data['Trial'].unique()[no_ * n_trials: (no_ + 1) * n_trials]
        chosen_trials = np.random.choice(test_data['Trial'].unique(), n_trials)
        trial_data = test_data[test_data['Trial'].isin(chosen_trials)]
        trial_dataset = SpikeTimeVidData2(trial_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                        window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                        pred=False, window_prev=window_prev, frame_window=frame_window,
                                        dt_frames=dt_frames, intervals=None, dataset='visnav',
                                        behavior=behavior, behavior_vars=behavior_vars, dt_vars=dt_vars,
                                        behavior_block_size=behavior_block_size, samples_per_behavior=samples_per_behavior,
                                        window_behavior=window_behavior, predict_behavior=predict_behavior,
                                        stoi_speed=stoi_speed, itos_speed=itos_speed, dt_speed=dt_speed)

        behavior_preds = predict_behavior(model, trial_dataset, itos_speed, sample=False)

        model_weights_path = model_weights.split('/')
        behavior_preds.to_csv(os.path.join(os.path.dirname(model_weights), 'behavior_preds.csv'), index=False)

        from neuroformer.visualize import set_plot_white
        set_plot_white()

        plt.figure(figsize=(20, 10))
        plt.grid()
        ms = 10 * 2
        plt.scatter(behavior_preds['cum_interval'], behavior_preds['behavior'], s=ms, label='pred', marker='x')
        plt.scatter(behavior_preds['cum_interval'], behavior_preds['true'], s=ms, label='true')
        plt.legend()

        plt.title("Finetuned model (2 trials)", fontsize=20)
        save_path = args.ft_base_path if args.ft_base_path else os.path.dirname(model_weights)
        plt.savefig(os.path.join(save_path, f"preds_{model_weights.split('/')[-1].split('.')[0]}.png"))
        plt.savefig(os.path.join(save_path, f"preds_{model_weights.split('/')[-1].split('.')[0]}.svg"))

    if args.infer_response:

        from neuroformer.utils import predict_raster_recursive_time_auto, process_predictions

        results_dict = dict()
        df_pred = None
        df_true = None

        top_p = 0.75
        top_p_t = 0.75
        temp = 1.25
        temp_t = 1.25

        # trials = test_data['Trial'].unique()[:8]
        trials = random.sample(list(test_data['Trial'].unique()), 8)
        for trial in trials:   
                print(f"Trial: {trial}")
                df_trial = df[df['Trial'] == trial]
                trial_dataset = SpikeTimeVidData2(df_trial, None, block_size, id_block_size, frame_block_size, prev_id_block_size, 
                                        window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats,
                                        pred=False, window_prev=window_prev, frame_window=frame_window,
                                        dt_frames=dt_frames, intervals=None, dataset='visnav',
                                        behavior=behavior, behavior_vars=behavior_vars, dt_vars=dt_vars,
                                        behavior_block_size=behavior_block_size, samples_per_behavior=samples_per_behavior,
                                        window_behavior=window_behavior, predict_behavior=predict_behavior,
                                        stoi_speed=stoi_speed, itos_speed=itos_speed, dt_speed=dt_speed)
                results_trial = predict_raster_recursive_time_auto(model, trial_dataset, window, window_prev, stoi, itos_dt, itos=itos, 
                                                                sample=True, top_p=top_p, top_p_t=top_p_t, temp=temp, temp_t=temp_t, 
                                                                frame_end=0, get_dt=True, gpu=False, pred_dt=True)
                # results_trial = predict_raster_hungarian(model, loader, itos_dt, top_p=0.75, temp=1)
                # print(f"MAX ID ---- {sorted(results_trial['ID'].unique()[-10])}")
                df_trial_pred, df_trial_true = process_predictions(results_trial, stoi, itos, window)
                print(f"pred: {df_trial_pred.shape}, true: {df_trial_true.shape}" )
                if df_pred is None:
                    df_pred = df_trial_pred
                    df_true = df_trial_true
                else:
                    df_pred = pd.concat([df_pred, df_trial_pred])
                    df_true = pd.concat([df_true, df_trial_true])

        from neuroformer.analysis import get_rates_trial, calc_corr_psth

        df_1 = df_trial
        df_pred_full = df_pred

        window_pred = 0.5
        window_pred = window if window_pred is None else window_pred
        intervals = np.array(sorted(set(df['Interval'].unique()) & set(df['Interval'].unique())))
        labels = np.array([round(window_pred + window_pred*n, 2) for n in range(0, int(max(df_pred_full['Interval']) / window_pred))])
        ids = sorted(set(df['ID'].unique()) & set(df['ID'].unique()))

        rates_pred = get_rates_trial(df_pred_full, labels)
        rates_1 = get_rates_trial(df_1, labels)

        top_corr_pred = calc_corr_psth(rates_pred, rates_1)

        """

        Evaluate results

        """

        from neuroformer.analysis import get_accuracy, compute_scores

        len_pred, len_true = len(df_pred_full), len(df_1)
        print(f"len_pred: {len_pred}, len_true: {len_true}")

        accuracy = get_accuracy(df_pred, df_1)
        pred_scores = compute_scores(df_1, df_pred_full)

        print(f"pred: {pred_scores}")

        n_bins = 30
        set_plot_white()
        plt.figure(figsize=(10, 10), facecolor='white')
        plt.title(f'PSTH Correlations (V1 + AL) {title}', fontsize=25)
        plt.ylabel('Count (n)', fontsize=25)
        plt.xlabel('Pearson r', fontsize=25)
        # plt.hist(top_corr_real_2, label='real - real3', alpha=0.6)
        plt.hist(top_corr_pred, label='real - simulated', alpha=0.6, bins=30)
        plt.legend(fontsize=20)

        dir_name = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)

        top_p = 0
        save_title = f'_top_p{top_p}'
        plt.savefig(os.path.join(dir_name, F'psth_corr_{save_title}_.svg'))
        df_pred.to_csv(os.path.join(dir_name, F'df_pred_{save_title}_.csv'))

        plot_distribution(df_1, df_pred, save_path=os.path.join(dir_name, F'psth_dist_.svg'))

        total_scores = dict()
        total_scores['pred'] = pred_scores

        print(f"model: {title}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # base_path = "/data5/antonis/neuroformer/models/tensorboard/visnav/behavior_predict"
    # model_weights = glob.glob(os.path.join(base_path, "**/**.pt"), recursive=True)
    # print(f"model_weights: {model_weights}")
    # for model_weight in model_weights:
    #     run_path = os.path.dirname(model_weight)
    #     print(f"run_path: {run_path}")
    #     args.base_path = run_path 
    #     try:
    #         main(args)
    #     except:
    #         print(f" -- Error in {model_weight} -- ")
    #         continue
# %%


"""
command:

CUDA_VISIBLE_DEVICES=0 python neuroformer_visnav_finetune_predbehavior.py \
    --base_path /data5/antonis/neuroformer/models/tensorboard/visnav/behavior_pred_exp/classification/window:0.05_prev:0.25/sparse_f:None_id:None/w:0.05_wp:0.25 \
    --infer_response 

"""