import numpy as np
import collections
from torch.utils import data

import torch
import torch.nn as nn

import pandas as pd

import math
from torch.utils.data import Dataset

from scipy import io as scipyio
from skimage import io
from PIL import Image
from torchvision import transforms
from skimage import io


def trial_df(data):
    """
    # data: Number of neurons (N,)
    # returns: Time x ID x Trial dataframe
    """
    # gather data from all the different neurons/trials
    df_dict = collections.defaultdict(dict)
    for ID in range(len(data)):
        df = data[ID]
        trial_dict = collections.defaultdict(list)
        for trial in range(df.shape[-1]):
            for row in range(df.shape[0]):
                if df[row, trial] <= 0 or df[row, trial] is None:
                    continue
                trial_dict[trial].append(df[row, trial])
        if len(trial_dict) == 0:
            continue
        df_dict[ID] = trial_dict
    
    # organize in one single dataframe
    df_list = []
    for ID in df_dict.keys():
        df_list.append(pd.concat(pd.DataFrame({'Time':v, 'ID':ID, 'Trial':k,}) for k, v in df_dict[ID].items()))
    
    df = pd.concat(df_list).sort_values(by=['Trial', 'Time'])
    return df

def trial_df_real(data):
    """
    # data: Number of neurons (N,)
    # returns: Time x ID x Trial dataframe
    """
    # gather data from all the different neurons/trials
    df_dict = collections.defaultdict(dict)
    for ID in range(len(data)):
        df = data[ID]
        if type(df) == np.void:
            df = df[0][:, 0]
        trial_dict = collections.defaultdict(list)
        for trial in range(df.shape[-1]):
            for row in range(df.shape[0]):
                if df[row, trial] is None:
                    continue
                trial_dict[trial] = df[row, trial].flatten().tolist()
        if len(trial_dict) == 0:
            continue
        df_dict[ID] = trial_dict
    
    # organize in one single dataframe
    df_list = []
    for ID in df_dict.keys():
        df_list.append(pd.concat(pd.DataFrame({'Time':v, 'ID':ID, 'Trial':k,}) for k, v in df_dict[ID].items()))
    
    df = pd.concat(df_list).sort_values(by=['Trial', 'Time'])
    return df

def trial_df_combo3(data, n_stim=0):
    """
    # data: Number of neurons (N,)
    # returns: Time x ID x Trial dataframe
    """
    # gather data from all the different neurons/trials
    df_dict = collections.defaultdict(dict)
    for ID in range(len(data)):
        df = data[ID]
        if type(df) == np.void:
            df = df[0][:, n_stim]
        trial_dict = collections.defaultdict(list)
        for trial_no in range(df.shape[-1]):
            trial = df[trial_no].flatten()
            if len(trial) == 0:
                continue
            trial_dict[trial_no + (20) * n_stim] = trial.tolist()
        if len(trial_dict) == 0:
            continue
        df_dict[ID] = trial_dict
    
    # organize in one single dataframe
    df_list = []
    for ID in df_dict.keys():
        df_list.append(pd.concat(pd.DataFrame({'Time':v, 'ID':ID, 'Trial':k,}) for k, v in df_dict[ID].items()))
    
    df = pd.concat(df_list).sort_values(by=['Trial', 'Time'])
    return df


def split_intervals(df, interval=1):
    new_df = df.sort_values(['Trial', 'Time']).reset_index().T.iloc[1:]
    prev_t = 0
    n = 0
    for column in range(new_df.shape[-1]):
        t = new_df.iloc[0, column]
        if t < prev_t:
            prev_t = 0
        dt = t - prev_t
        if dt >= interval:
            idx = column
            new_column = [new_df.iloc[0, idx - 1],
                          '.', new_df.iloc[2, idx]]    
            col_name = 'n_ %i' % (idx)
            new_df.insert(idx, idx, new_column, allow_duplicates=True) 
            n += 1
            prev_t = t

    return new_df.T.reset_index().iloc[:, 1:]


def make_intervals(data, window):
    intervals = []
    for trial in data['Trial'].unique():
        df = data[data['Trial'] == trial]
        rn = 0
        while True:
            rn += window
            interval = df[(df['Time'] < rn) & (df['Time'] >= rn - window)]
            intervals += [rn] * len(interval)
            if rn > max(df['Time']):
                break
    return intervals



def group_intervals(df, dt):
    '''group data into intervals'''
    bins = int(max(df['Time'])/dt)
    intervals = pd.cut(df['Time'], bins=int(max(df['Time'])/dt))
    labels = [dt + dt*n for n in range(0, int(max(df['Time'])/dt))]
    df['intervals'] = pd.cut(df['Time'], bins=int(max(df['Time'])/dt), labels=labels).astype('float')
    df['intervals'].round(decimals=1)
    return df

def split_idx(df, block_size):
    '''assign indexer to intervals for DataLoader class'''
    new_row = []
    current_i = -1
    seq_l = 0
    prev_dt = 0 
    for row in df.iterrows():
        dt = row[-1]['intervals']
        if dt == prev_dt:
            if seq_l > block_size // 2:
                current_i = current_i + 1
                seq_l = 0
        else:
            current_i = current_i + 1
            seq_l = 0
        new_row.append(current_i)
        prev_dt = dt
    df['idx'] = new_row
    return df
    
# Tf combo = 1/60
# Tf = 1/20   # preriod
def get_frame_idx(t, Tf):
    """ 
    Get the raw index of frame at certain neuron firing time 
        
    # Tf:
    video 1/f (s)
    """
    idx = math.ceil(t / Tf)
    return idx if idx > 0 else 0

# Tf combo = 1/60
def dt_frames_idx(t, Tf, dt_frames=0.25):
    """     
    Get the sequence of frames index at certain time 
        
    # dt_frames:
    # dt / Tf
    """
    return int(t // Tf) // dt_frames


def image_dataset(frame_stack):
    """ Convert frames into images tensor compatible with Resnet"""
    preprocess = transforms.Compose([
    transforms.Resize((64, 112)),
    transforms.CenterCrop((64, 112)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    # frame_stack = io.imread(path)
    im_list = []
    for idx in range(len(frame_stack)):
        image = Image.fromarray(frame_stack[idx])   # .convert('RGB')
        image = preprocess(image).unsqueeze(0)
        image = (image / image.max()) - 0.5
        im_list.append(image)

    im_stack = torch.cat(im_list)
    print("im_sack size: {}".format(im_stack.size()))
    return im_stack

def video_dataset(frame_stack):
    """ 
    Scale, Normalize, and Convert to format (C, T, H, W) for ResNet 3D
    """
    frame_stack = frame_stack / frame_stack.max()
    mean = frame_stack.mean()
    std = frame_stack.std()
    frame_stack = (frame_stack - mean) / std
    frame_stack = frame_stack / frame_stack.max()
    video = np.repeat(frame_stack[None, ...], 3, axis=0)
    return video


# dataloader class
class SpikeTimeVidData(Dataset):
        """
        # data: 
        0 col: Time
        1 col: Neuron ID

        # block_size:
        Transformer Window

        # dt
        Time intervals from data col 0

        # stoi, itos: dictionaries mapping neuron ID to transformer vocab
        and vice-versa.
        """

        def __init__(self, data, frames, block_size, frame_block_size, window, frame_memory, stoi, itos, neurons, stoi_dt=None, itos_dt=None, frame_feats=None, pred=False):
                
                pixels = [i for i in range(frames.min(), frames.max() + 1)] if frames is not None else []
                feat_encodings = neurons + ['EOS'] + ['PAD'] + pixels                 
                # stoi = { ch:i for i,ch in enumerate(feat_encodings) }
                # itos = { i:ch for i,ch in enumerate(feat_encodings) }
                self.stoi = stoi
                self.itos = itos
                self.stoi_dt = stoi_dt
                self.itos_dt = itos_dt
                self.dt_max = max(list(stoi_dt.values()))
                                
                self.frame_block_size = frame_block_size
                self.block_size = block_size
                self.id_block_size = block_size - frame_block_size 
                
                assert self.id_block_size > 0
                assert self.frame_block_size + self.id_block_size == self.block_size
                
                self.frame_memory = frame_memory
                
                self.data = data.reset_index().drop(['index'], axis=1)
                self.frame_feats = frame_feats
                self.frames = frames

                data_size, id_population_size, pixels_population_size = len(data), len(neurons + ['SOS'] + ['EOS'] + ['PAD']), len(pixels)
                print('Length: %d Neurons: %d Pixels: %d.' % (data_size, id_population_size, pixels_population_size))
                print(f'id block size: {self.id_block_size}')
                print(f'frames: {frame_block_size}, id: {self.id_block_size}')
                self.population_size = len([*stoi.keys()])
                self.id_population_size = id_population_size

                # keep track of which interval we are and how many neurons remain
                # (i.e did not fit in block)
                # self.t = self.data['Interval'].unique()
                self.t = self.data.drop_duplicates(subset=['Interval', 'Trial'])[['Interval', 'Trial']] # .sample(frac=1).reset_index(drop=True) # interval-trial unique pairs
                self.idx = 0
                self.window = window        # interval window (prediction)
                self.interval = 0           # our current interval                
                self.pred = pred
                self.size = len(data) + 2 * len(self.t) # 2x = EOS , PAD

        def __len__(self):
                return len(self.t)

        def __getitem__(self, idx):
                """
                Using an odd Block_Size, in order to be able to 
                appropriately mask joint image and id encodings.
                
                Example for block_size = n:

                x = [frame_token_1... frame_token_n ..., id_1, id_n,]    
                y = [frame_token_2... frame_token_n + 1 ..., id_2, id_n + 1,]

                """

                # grab a chunk of (block_size + 1) characters from the data
                t = self.t.iloc[idx]
                t_idx = self.data.index[(self.data['Interval'] == t['Interval']) & (self.data['Trial'] == t['Trial'])]
                start_idx, stop_idx = t_idx[0], (t_idx[-1] + 1)  # add 1 so last idx will be included in iloc

                x = collections.defaultdict(list)
                y = collections.defaultdict(list)

                data_current = self.data[(self.data['Interval'] == t['Interval']) & (self.data['Trial'] == t['Trial'])]

                # encode every character to an integer
                chunk = data_current['ID']
                current_dix = [self.stoi[s] for s in chunk]              
                all_dix = current_dix

                # pad x values if needed
                pad_n = self.id_block_size - (len(all_dix) + 1)   # len chunk is 1 unit bigger than x, y
                x['pad'] = pad_n if pad_n > 0 else 0 
                # # remove one pad in order to add EOS token
                # x['pad'] = x['pad'] - 1
                id_tokens = [self.stoi['SOS']] + all_dix + [self.stoi['EOS']] + [self.stoi['PAD']] * x['pad']    # [:self.id_block_size]
                x['id'] = torch.tensor(id_tokens[:-1], dtype=torch.long)

                # add temporal signal
                dt_chunk = data_current['Interval_dt']
                # zero-out time for sequence
                all_dt_chunk = [self.stoi_dt[dt] for dt in dt_chunk] # prev_dt_chunk.append(dt_chunk)
                # all_dt_chunk_min = all_dt_chunk.min()
                # all_dt_chunk = all_dt_chunk - all_dt_chunk.min()
                all_dt_chunk_padded = [0] + all_dt_chunk + [self.dt_max] * (x['pad'] + 1) # 0 = SOS, max = EOS
                
                x['dt'] = torch.tensor(all_dt_chunk_padded[:-1], dtype=torch.long)
                
                # video encodings
                frame_idx = math.ceil((t['Interval'] / self.window) - 1)    # get last 1 second of frames
                frame_idx = len(self.frame_feats) - 1 if frame_idx >= len(self.frame_feats) else frame_idx
                if self.frames is not None:
                    frames = self.frames[frame_idx]
                    fdix = [self.stoi[s] for s in frames]
                    y_frames = fdix[1:] + [self.stoi['SOS']]
                    y['frames'] = torch.tensor(y_frames, dtype=torch.long)
                
                y['id'] = torch.tensor(id_tokens[1:], dtype=torch.long)

                if self.frame_feats is not None:
                    x['frames'] = torch.tensor(self.frame_feats[frame_idx], dtype=torch.float32)
                else:
                    x['frames'] = torch.tensor(fdix, dtype=torch.long)
                
                if self.frames is not None:
                    x['frame_codes'] = torch.tensor(fdix, dtype=torch.long)
                
                y['dt'] = torch.tensor(all_dt_chunk_padded[1:], dtype=torch.long)
                
                if self.pred:
                    y['time'] = torch.tensor(all_dt_chunk_padded[1:], dtype=torch.long)

                y['indexes'] = torch.linspace(1, len(y['id']), len(y['id'])).long() + self.idx
                self.idx += len(y['id']) - x['pad']

                # x['pad'] += 1   # if +1, EOS is not attended
                # x['pad'] = 0    # if 0, EOS is attended
                x['interval'] = torch.tensor(t)

                return x, y


# dataloader class
class SpikeTimeVidData2(Dataset):
        """
        # data: 
        0 col: Time
        1 col: Neuron ID

        # block_size:
        Transformer Window

        # dt
        Time intervals from data col 0

        # stoi, itos: dictionaries mapping neuron ID to transformer vocab
        and vice-versa.
        """

        def __init__(self, data, frames, block_size, id_block_size, frame_block_size, id_prev_block_size, window, frame_memory, 
                     stoi, itos, neurons, stoi_dt=None, itos_dt=None, frame_feats=None, pred=False):
                
                pixels = [i for i in range(frames.min(), frames.max() + 1)] if frames is not None else []
                feat_encodings = neurons + ['EOS'] + ['PAD'] + pixels                 
                # stoi = { ch:i for i,ch in enumerate(feat_encodings) }
                # itos = { i:ch for i,ch in enumerate(feat_encodings) }
                self.stoi = stoi
                self.itos = itos
                self.stoi_dt = stoi_dt
                self.itos_dt = itos_dt
                self.dt_max = max(list(stoi_dt.values()))
                                
                self.frame_block_size = frame_block_size
                self.block_size = block_size
                self.id_prev_block_size = id_prev_block_size
                self.id_block_size = id_block_size
                
                assert self.id_block_size > 0
                # assert self.frame_block_size + self.id_block_size + self.prev_id_block_size == self.block_size
                
                self.frame_memory = frame_memory
                
                self.data = data.reset_index().drop(['index'], axis=1)
                self.frame_feats = frame_feats
                self.frames = frames

                data_size, id_population_size, pixels_population_size = len(data), len(neurons + ['SOS'] + ['EOS'] + ['PAD']), len(pixels)
                print('Length: %d Neurons: %d Pixels: %d.' % (data_size, id_population_size, pixels_population_size))
                print(f'id block size: {self.id_block_size}')
                print(f'frames: {frame_block_size}, id: {self.id_block_size}')
                self.population_size = len([*stoi.keys()])
                self.id_population_size = id_population_size

                # keep track of which interval we are and how many neurons remain
                # (i.e did not fit in block)
                # self.t = self.data['Interval'].unique()
                self.t = self.data.drop_duplicates(subset=['Interval', 'Trial'])[['Interval', 'Trial']] # .sample(frac=1).reset_index(drop=True) # interval-trial unique pairs
                self.idx = 0
                self.window = window        # interval window (prediction)
                self.interval = 0           # our current interval                
                self.pred = pred
                self.size = len(data) + 2 * len(self.t) # 2x = EOS , PAD

        def __len__(self):
                return len(self.t)

        def get_interval(self, interval, trial):
                """
                Returns interval[0] >= data < interval[1].
                chunk = ID
                dt_chunk = dt
                pad_n
                """
                # data = self.data[(self.data['Interval'] > interval[0]) & 
                #                  (self.data['Interval'] <= interval[1]) &
                #                  (self.data['Trial'] == trial)][-(self.id_prev_block_size - 2):]

                data = self.data[(self.data['Interval'] == interval) & 
                                 (self.data['Trial'] == trial)][-(self.id_prev_block_size - 2):]                
                
                chunk = data['id']
                dix = [self.stoi[s] for s in chunk]
                dix = [self.stoi['SOS']] + dix + [self.stoi['EOS']]
                pad_n = self.id_block_size - (len(dix) + 1 - 2) # len chunk is 1 unit bigger than x, y
                dix = dix + [self.stoi['PAD']] * pad_n

                dt_chunk = data['Time'] - data['Time'].min() + 1
                dt_chunk = [1] + dt_chunk.tolist() + [1 + 0.2] * (pad_n + 1) # 0 = SOS, max = EOS
            
                return chunk, dt_chunk, pad_n

        def __getitem__(self, idx):
                """
                Using an odd Block_Size, in order to be able to 
                appropriately mask joint image and id encodings.
                
                Example for block_size = n:

                x = [frame_token_1... frame_token_n ..., id_1, id_n,]    
                y = [frame_token_2... frame_token_n + 1 ..., id_2, id_n + 1,]

                """

                # grab a chunk of (block_size + 1) characters from the data
                t = self.t.iloc[idx]
                t_idx = self.data.index[(self.data['Interval'] == t['Interval']) & (self.data['Trial'] == t['Trial'])]
                start_idx, stop_idx = t_idx[0], (t_idx[-1] + 1)  # add 1 so last idx will be included in iloc

                x = collections.defaultdict(list)
                y = collections.defaultdict(list)

                ## PREV ##
                # get state history + dt (last 30 seconds)
                prev_int = t['Interval'] - 0.5
                prev_int = prev_int if prev_int > 0 else 0  
                prev_id_interval = prev_int, t['Interval']
                x['id_prev'], x['dt_prev'], pad_n = self.get_interval(prev_int, t['Trial'])

                
                ## CURRENT ##
                # data_current = self.data[(self.data['Interval'] == t['Interval']) & (self.data['Trial'] == t['Trial'])][-(self.id_block_size - 2):]
                current_int = t['Interval']
                x['id'], x['dt'], x['pad'] = self.get_interval(current_int, t['Trial'])

                x['id'] = torch.tensor(current_dix[:-1], dtype=torch.long)

                # add temporal signal
                dt_chunk = data_current['Time'] - data_current['Time'].min() + 1
                x['dt'] = torch.tensor(dt_chunk[:-1], dtype=torch.float)
                
                # for backbone:
                if t['Trial'] <= 20: n_stim = 0
                elif t['Trial'] <= 40: n_stim = 1
                elif t['Trial'] <= 60: n_stim = 2                
                frame_idx = get_frame_idx(t['Interval'] + self.window, 1/20)     # get last 1 second of frames
                frame_idx = frame_idx if frame_idx >= 20 else 20
                frame_idx = frame_idx if frame_idx < self.frame_feats.shape[2] else self.frame_feats.shape[2]
                # x['idx'] = torch.tensor([frame_idx, n_stim], dtype=torch.float16)
                if self.frame_feats is not None:
                    x['frames'] = self.frame_feats[n_stim, :, frame_idx - 20:frame_idx].type(torch.float16)
                
                y['id'] = torch.tensor(current_dix[1:], dtype=torch.long)
                y['dt'] = torch.tensor(dt_chunk[1:], dtype=torch.float)
                
                if self.pred:
                    dt_real = np.array(dt_chunk[1:]) + data_current['Time'].min()
                    y['time'] = torch.tensor(dt_real, dtype=torch.float)

                y['indexes'] = torch.linspace(1, len(y['id']), len(y['id'])).long() + self.idx
                self.idx += len(y['id']) - x['pad']

                # x['pad'] += 1   # if +1, EOS is not attended
                # x['pad'] = 0    # if 0, EOS is attended
                # x['interval'] = torch.tensor(t)
                # x['frame_token'] = torch.tensor([self.stoi['EOS']], dtype=torch.long)
                x['prev_int'] = torch.tensor(len(data_prev), dtype=torch.long)
                
                return x, y


# # video encodings
# frame_idx = math.ceil((t['Interval'] / self.window) - 1)    # get last 1 second of frames
# frame_idx = len(self.frame_feats) - 1 if frame_idx >= len(self.frame_feats) else frame_idx
# if self.frames is not None:
#     frames = self.frames[frame_idx]
#     fdix = [self.stoi[s] for s in frames]
#     y_frames = fdix[1:] + [self.stoi['SOS']]
#     y['frames'] = torch.tensor(y_frames, dtype=torch.long)

# if self.frame_feats is not None:
#     x['frames'] = torch.tensor(self.frame_feats[frame_idx], dtype=torch.float32)
# else:
#     x['frames'] = torch.tensor(fdix, dtype=torch.long)

# if self.frames is not None:
#     x['frame_codes'] = torch.tensor(fdix, dtype=torch.long)

# def top_k_logits(logits, k):
#     v, ix = torch.topk(logits, k)
#     out = logits.clone()
#     out[out < v[:, [-1]]] = -float('inf')
#     return out


# @torch.no_grad()
# def sample(model, loader, temperature=1/0, sample=False, top_k=None):
#     block_size = model.get_block_size()
#     model.eval()
#     for x, y in loader:
#         for key, value in x.items():
#             x[key] = x[key].to(self.device)
#         y = y.to(self.device)
