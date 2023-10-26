import sys
sys.path.append('./neuroformer')

import itertools
import torch
import numpy as np
import pandas as pd
import pickle


def split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.1):
    chosen_idx = np.random.choice(len(intervals), int(len(intervals) * r_split))
    train_intervals = intervals[chosen_idx]
    test_intervals = intervals[~chosen_idx]
    finetune_intervals = np.array(train_intervals[:int(len(train_intervals) * r_split_ft)])
    return train_intervals, test_intervals, finetune_intervals

def combo3_V1AL_callback(frames, frame_idx, n_frames, **kwargs):
    """
    Shape of frames: [3, 640, 64, 112]
                     (3 = number of stimuli)
                     (0-20 = n_stim 0,
                      20-40 = n_stim 1,
                      40-60 = n_stim 2)
    frame_idx: the frame_idx in question
    n_frames: the number of frames to be returned
    """
    trial = kwargs['trial']
    if trial <= 20: n_stim = 0
    elif trial <= 40: n_stim = 1
    elif trial <= 60: n_stim = 2
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    f_idx_0 = max(0, frame_idx - n_frames)
    f_idx_1 = f_idx_0 + n_frames
    chosen_frames = frames[n_stim, f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
    return chosen_frames

def visnav_callback(frames, frame_idx, n_frames, **kwargs):
    """
    frames: [n_frames, 1, 64, 112]
    frame_idx: the frame_idx in question
    n_frames: the number of frames to be returned
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    f_idx_0 = max(0, frame_idx - n_frames)
    f_idx_1 = f_idx_0 + n_frames
    chosen_frames = frames[f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
    return chosen_frames

def load_V1AL(config, stimulus_path=None, response_path=None, top_p_ids=None):
    if stimulus_path is None:
        # stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
        stimulus_path = "data/Combo3_V1AL/Combo3_V1AL_stimulus.pt"
    if response_path is None:
        response_path = "data/Combo3_V1AL/NF_1.5/Combo3_V1AL.pkl"
    
    data = {}
    data['spikes'] = pickle.load(open(response_path, "rb"))
    data['stimulus'] = torch.load(stimulus_path).transpose(1, 2).squeeze(1)

    intervals = np.arange(0, 31, config.window.curr)
    trials = list(set(data['spikes'].keys()))
    combinations = np.array(list(itertools.product(intervals, trials)))
    train_intervals, test_intervals, finetune_intervals = split_data_by_interval(combinations, r_split=0.8, r_split_ft=0.01)

    return (data, intervals,
           train_intervals, test_intervals, 
           finetune_intervals, combo3_V1AL_callback)

def load_natural_movie(stimulus_path=None, response_path=None, top_p_ids=None):
    if stimulus_path is None:
        stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
    if response_path is None:
        response_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/neural/NatureMoviePart1-A/20-NatureMovie_part1-A_spikes(1).mat"
    # vid_paths = sorted(glob.glob(stimulus_path + '/*.tif'))
    # vid_list = [skimage.io.imread(vid)[::3] for vid in vid_paths]
    # video_stack = [torch.nan_to_num(image_dataset(vid)).transpose(1, 0) for vid in vid_list]

    vs = torch.load(stimulus_path)
    video_stack = [vs[i] for i in range(len(vs))]

    df = pd.read_csv(response_path)

    df['Time'] = df['Time'] * 0.1499

    if top_p_ids is not None:
        df = df[df['ID'].isin(top_p_ids)]

    n = []
    for n_stim in range(df['Trial'].max() // 200):
        n_trial = [i for i in range(200 // 20)]
        for n_trial in n_trial:
            trial = (n_stim + 1) * 20 - n_trial
            n.append(trial)
    n_trials = n

    return video_stack, df, n_trials

def load_visnav(version, config, selection=None):
    if version not in ["medial", "lateral"]:
        raise ValueError("version must be either 'medial' or 'lateral'")
    
    if version == "medial":
        data_path = "./data/VisNav_VR_Expt/MedialVRDataset/"
    elif version == "lateral":
        data_path = "./data/VisNav_VR_Expt/LateralVRDataset/"

    spikes_path = f"{data_path}/NF_1.5/spikerates_dt_0.01.npy"
    speed_path = f"{data_path}/NF_1.5/behavior_speed_dt_0.05.npy"
    stim_path = f"{data_path}/NF_1.5/stimulus.npy"
    phi_path = f"{data_path}/NF_1.5/phi_dt_0.05.npy"
    th_path = f"{data_path}/NF_1.5/th_dt_0.05.npy"

    data = dict()
    data['spikes'] = np.load(spikes_path)
    data['speed'] = np.load(speed_path)
    data['stimulus'] = np.load(stim_path)
    data['phi'] = np.load(phi_path)
    data['th'] = np.load(th_path)

    if selection is not None:
        selection = np.array(pd.read_csv(os.path.join(data_path, f"{selection}.csv"), header=None)).flatten()
        data['spikes'] = data['spikes'][selection - 1]

    spikes = data['spikes']
    intervals = np.arange(0, spikes.shape[1] * config.resolution.dt, config.window.curr)
    train_intervals, test_intervals, finetune_intervals = split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.01)

    return data, intervals, train_intervals, test_intervals, finetune_intervals, visnav_callback
