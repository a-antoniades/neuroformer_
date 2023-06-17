import glob
import os
import logging
import pickle
import collections
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from beam_search import beam_decode
from SpikeVidUtils import get_interval, round_n
from SpikeVidUtils import SpikeTimeVidData2 as SP
from model_neuroformer import GPT

import yaml

logger = logging.getLogger(__name__)




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def all_device(data, device):
    device = torch.device(device)
    if isinstance(data, dict):
        return {k: all_device(v, device) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [all_device(v, device) for v in data if v is not None]
    elif isinstance(data, tuple):
        return tuple(all_device(v, device) for v in data if v is not None)
    else:
        return data.to(device)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('inf')
    return out

def top_k_arr(a, k):
    idx = np.argpartition(-a.ravel(),k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))

def get_interval_dist(df, plot_dist=False):
    dist = df.groupby(['Interval', 'Trial']).size().reset_index(name='Count')
    # dist = dist.groupby('Interval').agg({'Count': ['mean', 'std']})
    if plot_dist:
        dist.plot.bar()
        plt.show()
    return dist

def get_best_ckpt(base_path, model_path=None):
    model_weights = glob.glob(os.path.join(base_path, '**/**.pt'), recursive=True)
    model_weights = sorted(model_weights, key=os.path.getmtime, reverse=True)
    assert len(model_weights) > 0, "No model weights found"

    if model_path in model_weights:
        load_weights = model_path
    else:
        load_weights = model_weights[0]
    
    return load_weights

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def update_object(obj1, obj2):
    """Update the attributes of obj1 with the attributes of obj2"""
    for attr_name in dir(obj2):
        # Ignore special attributes
        if not attr_name.startswith('__'):
            attr_value = getattr(obj2, attr_name)

            # Set the attribute if it doesn't exist in obj1
            # if not hasattr(obj1, attr_name):
            # print(f"Setting {attr_name} to {attr_value}")
            setattr(obj1, attr_name, attr_value)

def df_to_dict(df):
    d = {k: f.groupby('Interval').apply(lambda x: {'Time': np.array(x['Time']), 'ID': np.array(x['ID'])}).to_dict()
        for k, f in df.groupby('Trial')}
    return d
    
def get_model_attr(mconf, tconf):
  n_head = mconf.n_head
  n_block = mconf.n_layer
  nembd = mconf.n_embd
  data = tconf.dataset[-20:-4]
  model_attr =  f"Head:{n_head}_Block{n_block}_nembd:{nembd}_data:{data}"
  return model_attr

def print_full(df, length=None):
    length = len(df) if length is None else len(df)
    print(length)
    pd.set_option('display.max_rows', length)
    torch.set_printoptions(threshold=1e3)
    print(df)
    pd.reset_option('display.max_rows')
    torch.set_printoptions(threshold=1e3)

def object_to_dict(obj):
    d = dict()
    d_types = [int, float, str, bool, tuple, type(None)]
    for a in dir(obj):
        if not a.startswith('__'):
            if type(getattr(obj, a)) in d_types:
                d[a] = getattr(obj, a)
    return d

    # return {a: getattr(obj, a) for a in dir(obj) if not a.startswith('__')}

def save_yaml(obj, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(obj, outfile)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def set_model_attributes(mconf):
    for a in dir(mconf):
        if not a.startswith('__'):
            globals()[a] = getattr(mconf, a)


class NestedDefaultDict(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


def varname(variable):
    for name in list(globals().keys()):
        expression = f'id({name})'
        if id(variable) == eval(expression):
            return name

def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def set_model_attr(mconf):
    for a in dir(mconf):
        if not a.startswith('__'):
            globals()[a] = value = getattr(mconf, a)


def load_model(model_dir):
    model_path = glob.glob(os.path.join(model_dir, "*.pt"))[0]
    mconf_path = glob.glob(os.path.join(model_dir, "*_mconf.pkl"))[0]
    tconf_path = glob.glob(os.path.join(model_dir, "*_tconf.pkl"))[0]

    with open(mconf_path, 'rb') as handle:
        mconf = pickle.load(handle)
    with open(tconf_path, 'rb') as handle:
        tconf = pickle.load(handle)
    
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path))
    return model, mconf, tconf

def running_jupyter():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # 'ZMQInteractiveShell' is the class name for the Jupyter Notebook shell
            return True
        else:
            # Probably in IPython or other interactive shell
            return False
    except (NameError, ImportError):
        # Probably in a standard Python shell
        pass
    return False

def process_predictions(results, stoi, itos, window):
    pred_keys = ['ID', 'dt', 'Trial', 'Interval']
    predicted_dict = {k: results[k] for k in results if k in pred_keys}

    forbidden_tokens = ['SOS', 'EOS', 'PAD']
    df_pred = pd.DataFrame(predicted_dict)
    df_pred = df_pred[(df_pred != 'SOS').all(axis=1)].reset_index(drop=True)
    df_pred = df_pred[(df_pred != 'EOS').all(axis=1)].reset_index(drop=True)
    df_pred = df_pred[(df_pred != 'PAD').all(axis=1)].reset_index(drop=True)
    
    df_pred['Time'] = df_pred['dt'] + df_pred['Interval']
    df_pred = df_pred[df_pred['Interval'] > 0]
    # df_pred = df_pred[(df_pred['ID'] <= stoi['SOS']) & (df_pred['dt'] <= window) & (df_pred['Time'] >= 0)]
    true_keys = ['true', 'time']
    true_dict = {k: results[k] for k in results if k in true_keys}
    df_true = pd.DataFrame(true_dict)
    # if 'SOS' in stoi:
    #     # sos_id = list(itos.keys())[list(itos.values()).index('SOS')]
    #     sos_id = stoi['SOS']
    #     n_sos = len(df_true[df_true['true'] == sos_id])
    #     print(f'SOS fouuuund: {n_sos}')
    #     df_true = df_true[df_true['true'] != sos_id]
    # if 'EOS' in stoi:
    #     # eos_id = list(itos.keys())[list(itos.values()).index('EOS')]
    #     eos_id = stoi['EOS']
    #     n_eos = len(df_true[df_true['true'] == eos_id])
    #     print(f'EOS fouuuund: {n_eos}')
    #     df_true = df_true[df_true['true'] != eos_id]
    df_true.rename({'true':'ID', 'time':'dt'}, axis=1, inplace=True)
    # df_true['time'] = df_true['dt'] + df_true['interval'] - 0.5

    return df_pred.reset_index(drop=True), df_true.reset_index(drop=True)

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _, _ = model(x)
        # pluch the logits at the final step and scale by temperature
        logits = logits['id'][:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x['id'] = torch.cat((x['id'], ix), dim=1)

        return x

# @torch.no_grad()
# def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1e10):
#     """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#         Args:
#             logits: logits distribution shape (batch size, vocabulary size)
#             top_k >0: keep only top k tokens with highest probability (top-k filtering).
#             top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
#                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
#     """
#     top_k = min(top_k, logits.size(-1))  # Safety check
#     if top_k > 0:
#         # Remove all tokens with a probability less than the last token of the top-k
#         indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
#         logits[indices_to_remove] = filter_value

#     if top_p > 0:
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
#         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#         # Remove tokens with cumulative probability above the threshold
#         sorted_indices_to_remove = cumulative_probs > top_p
#         # Shift the indices to the right to keep also the first token above the threshold
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0

#         # batch_indices is an addition in the modified function. This creates an array holding the batch indices
#         batch_indices = torch.arange(logits.size(0))[:, None].expand_as(sorted_indices_to_remove)

#         # Use torch.where to handle higher-dimensional tensors for indexing
#         indices_to_remove = torch.where(sorted_indices_to_remove, sorted_indices, sorted_indices)
#         logits[batch_indices, indices_to_remove] = filter_value

#     return logits

@torch.no_grad()
def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def model_ensemble(models, x):
    """
    Ensemble of models
    """
    logits_total = dict()
    for model in models:
        model.eval()
        logits, _, _ = model(x)
        logits_total['id'] = logits['id'] + logits_total['id'] if 'id' in logits_total else logits['id']
        logits_total['dt'] = logits['dt'] + logits_total['dt'] if 'dt' in logits_total else logits['dt']
    return logits 

@torch.no_grad()
def predict_raster_recursive_time_auto(model, dataset, window, window_prev, stoi, itos_dt, itos=None, 
                                      get_dt=False, sample=False, top_k=0, top_p=0, top_p_t=0, temp=1, temp_t=1, 
                                      frame_end=0, gpu=False, pred_dt=True, true_past=False,
                                      p_bar=False, plot_probs=False):    
    """
    predict both ID and dt recursively
    """
    
    def pad_x(x, length, pad_token):
        """
        pad x with pad_token to length
        """
        if torch.is_tensor(x):
            x = x.tolist()
            
        pad_n = length - len(x)
        if pad_n < 0:
            x = x[-(length + 1):]
        if pad_n > 0:
            x = x + [pad_token] * pad_n
        x = torch.tensor(x, dtype=torch.long, device=device)
        return x
    

    def aggregate_dt(dt):
        agg_dt = []
        for i in range(len(dt)):
            # curr_dt = agg_dt[i] if i > 0 else 0 
            prev_dt = agg_dt[i - 1] if i > 1 else 0
            # agg_dt.append(curr_dt + dt[i])
            if i==0:
                agg_dt.append(dt[i])
            elif dt[i] == 0:
                if prev_dt == 0:
                    agg_dt.append(0)
                elif prev_dt > 0:
                    agg_dt.append(prev_dt + 1)
            elif dt[i] == dt[i - 1]:
                agg_dt.append(agg_dt[i - 1])
            elif dt[i] < dt[i - 1]:
                tot = agg_dt[i - 1] + dt[i] + 1
                agg_dt.append(tot)
            elif dt[i] > dt[i - 1]:
                diff = agg_dt[i - 1] + (dt[i] - dt[i - 1])
                agg_dt.append(dt[i - 1] + diff)
            else:
                return ValueError 
            # assert agg_dt[i] >= 0, f"agg_dt[{i}] = {agg_dt[i]}, dt[{i}] = {dt[i]}, dt[{i - 1}] = {dt[i - 1]}"

        assert len(agg_dt) == len(dt)
        # assert max(agg_dt) <= max(list(itos_dt.keys()))
        return agg_dt

    
    def add_sos_eos(x, sos_token=None, eos_token=None, idx_excl=None):
        """
        add sos and eos tokens to x
        """
     
        if sos_token is not None:
            idx_excl = []
            x_clean = []
            for n, i in enumerate(x):
                if i not in (eos_token, sos_token):
                    x_clean.append(i)
                else:
                    idx_excl.append(n)
        else:
            x_clean = [i for n, i in enumerate(x) if n not in idx_excl]
            sos_token, eos_token = min(list(itos_dt.keys())), max(list(itos_dt.keys()))
        x = torch.tensor([sos_token] + x_clean + [eos_token], dtype=torch.long, device=device)
        return x, idx_excl
    
    def sort_ids(id, dt):
        """
        sort ids and dt by dt
        (on last axis)
        """
        dt, idx = torch.sort(dt, dim=-1, descending=True)
        id = torch.gather(id, dim=-1, index=idx)
        return id, dt

    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = [model_n.to(device) for model_n in model] if isinstance(model, list) else model.to(device) 
    model = [model_n.eval() for model_n in model] if isinstance(model, list) else model.eval()
    tf = 0
    mconf = model[0].config if isinstance(model, list) else model.config
    T_id = mconf.id_block_size
    T_id_prev = mconf.prev_id_block_size
    
    context = [] # torch.tensor(0, device=device).unsqueeze(0)
    data = dict()
    data['true'] = []
    data['ID'] = context
    data['time'] = []
    data['dt'] = context
    data['Trial'] = context
    data['Interval'] = context

    loader = DataLoader(dataset, shuffle=False, pin_memory=False)
    pbar = tqdm(enumerate(loader), total=len(loader), disable=p_bar)
    for it, (x, y) in pbar:
        xid_prev_real = x['id_prev'].flatten()
        pad_real = x['pad_prev'].flatten() - 2

        xdt_prev_real = x['dt_prev'].flatten()
        # if it > 2:
        #     break
        # print(f"it = {it}, interval: {x['interval']}, window_prev: {window_prev}, window: {window}")

        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        
        # feed predicted IDs from buffer into past state
        # count how many steps 
        if true_past is False:
            if it > (window_prev / window):
                df = {k: v for k, v in data.items() if k in ('ID', 'dt', 'Trial', 'Interval', 'Time')}
                df = pd.DataFrame(df)

                # filter all instances of ['SOS', 'EOS' and 'PAD'] from all columns of pandas dataframe:
                df = df[(df != 'SOS').all(axis=1)].reset_index(drop=True)
                df = df[(df != 'EOS').all(axis=1)].reset_index(drop=True)
                df = df[(df != 'PAD').all(axis=1)].reset_index(drop=True)
                df['Time'] = df['dt'] + df['Interval'] - window

                # store results in dict
                prev_id_interval, current_id_interval = dataset.calc_intervals(x['interval'])
                x['id_prev'], x['dt_prev'], pad_prev = dataset.get_interval(prev_id_interval, float(x['trial']), T_id_prev, data=df)
                x['id_prev'] = torch.tensor(x['id_prev'][:-1], dtype=x['id'].dtype).unsqueeze(0).to(device)
                x['dt_prev'] = torch.tensor(x['dt_prev'][:-1], dtype=x['id'].dtype).unsqueeze(0).to(device)
                x['pad_prev'] = torch.tensor(pad_prev, dtype=x['id_prev'].dtype).unsqueeze(0).to(device)

                # sort ids according to dt
                x['id_prev'], x['dt_prev'] = sort_ids(x['id_prev'], x['dt_prev'])

                x_id_prev = x['id_prev'].flatten()
                x_dt_prev = x['dt_prev'].flatten()
                x_pad_prev = x['pad_prev'].flatten() - 2
                # print(f"real: {xid_prev_real[:len(xid_prev_real) - pad_real]}")
                # print(f"pred: {x_id_prev[:len(x_id_prev) - x_pad_prev]}")
                # print(f"real_dt: {xdt_prev_real[:len(xdt_prev_real) - pad_real]}")
                # print(f"pred_dt: {x_dt_prev[:len(x_dt_prev) - x_pad_prev]}")
                # print("---------------------------------")

            
        pad = x['pad'] if 'pad' in x else 0
        x['id_full'] = x['id'][:, 0]
        # x['id'] = x['id'][:, 0]
        x['dt_full'] = x['dt'][:, 0] if pred_dt else x['dt']
        # x['dt'] = x['dt'][:, 0] if pred_dt else x['dt']

        current_id_stoi = torch.empty(0, device=device)
        current_dt_stoi = torch.empty(0, device=device)
        for i in range(T_id - 1):   # 1st token is SOS (already there)
            t_pad = torch.tensor([stoi['PAD']] * (T_id - x['id_full'].shape[-1]), device=device)
            t_pad_dt = torch.tensor([0] * (T_id - x['dt_full'].shape[-1]), device=device)

            # forward model, if list of models, then ensemble
            if isinstance(model, list):
                logits = model_ensemble(model, x)
            else:
                logits, features, _ = model(x, y)
            
            logits['id'] = logits['id'][:, i]
            logits['dt'] = logits['dt'][:, i]
            # optionally crop probabilities to only the top k / p options
            if top_k or top_p != 0:
                logits['id'] = top_k_top_p_filtering(logits['id'], top_k=top_k, top_p=top_p)
                logits['dt'] = top_k_top_p_filtering(logits['dt'], top_k=top_k, top_p=top_p_t)
            
            logits['id'] = logits['id'] / temp
            logits['dt'] = logits['dt'] / temp_t

            # apply softmax to logits
            probs = F.softmax(logits['id'], dim=-1)
            probs_dt = F.softmax(logits['dt'], dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                ix_dt = torch.multinomial(probs_dt, num_samples=1)
                # ix = torch.poisson(torch.exp(logits), num_samples=1)
            else:
                # choose highest topk (1) sample
                _, ix = torch.topk(probs, k=1, dim=-1)
                _, ix_dt = torch.topk(probs_dt, k=1, dim=-1)

            if plot_probs:
                probs_n = np.array(probs)[0]
                xaxis = np.arange(len(probs_n))
                topk=5
                topk_indices = np.argpartition(probs_n, -topk)[-topk:]
                topk_probs = probs_n[topk_indices]
                plt.figure()
                plt.title(f"ID t={i}, indices: {topk_indices}")
                plt.axvline(x=ix, color='b', linestyle='--')
                plt.bar(xaxis, probs_n)
                plt.show()
                print(x['id'])
                print(f"Step {it}, i: {i} ix: {ix}, x_true: {y['id'][0, i]}")
                print(f"topk_ix: {topk_indices}")
                print(f"topk_probs: {topk_probs}")

                # plot dt probs
                probs_dt_n = np.array(probs_dt)[0]
                xaxis = np.arange(len(probs_dt_n))
                topk=5
                topk_indices = np.argpartition(probs_dt_n, -topk)[-topk:]
                topk_probs = probs_dt_n[topk_indices]
                plt.figure()
                plt.title(f"DT t={i}, indices: {topk_indices}")
                plt.axvline(x=ix_dt, color='b', linestyle='--')
                plt.bar(xaxis, probs_dt_n)
                plt.show()
                print(x['dt'])
                print(f"Step {it}, i: {i} ix_dt: {ix_dt}, dt_true: {y['dt'][0, i]}")
                print(f"topk_ix_dt: {topk_indices} \
                        topk_dt_probs: {topk_probs}")
            
            # convert ix_dt to dt and add to current time
            # print(f"ix: {ix}, x_true: {y['id'][0, i]} ")
            current_id_stoi = torch.cat((current_id_stoi, ix.flatten()))
            current_dt_stoi = torch.cat((current_dt_stoi, ix_dt.flatten()))
            dtx_itos = [itos_dt[int(ix_dt.flatten())]]

            x['id'][:, i + 1] = ix.flatten()
            x['dt'][:, i + 1] = ix_dt.flatten() if pred_dt else x['dt']
           
            if ix >= stoi['EOS']:  # ix >= stoi['EOS']:   # or i > T_id - int(x['pad']):   # or len(current_id_stoi) == T_id: # and dtx == 0.5:    # dtx >= window:   # ix == stoi['EOS']:
                # print(f"n_regres_block: {i}")
                break
                        
            try:
                ix_itos = torch.tensor(itos[int(ix.flatten())]).unsqueeze(0)
            except:
                TypeError(f"ix: {ix}, itos: {itos}")
            
            data['ID'] = data['ID'] + ix_itos.tolist()    # torch.cat((data['ID'], ix_itos))
            data['dt'] = data['dt'] + dtx_itos          # torch.cat((data['dt'], dtx_itos))
            data['Trial'] = data['Trial'] + x['trial'].tolist()            # torch.cat((data['Trial'], x['trial']))
            data['Interval'] = data['Interval'] + x['interval'].tolist() # torch.cat((data['Interval'], x['interval']))

            # x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
            # x['dt_full'] = torch.cat((x['dt_full'], ix_dt.flatten())) if pred_dt else x['dt']

        dty_itos = [itos_dt[int(dt)] for dt in y['dt'][:, :T_id - pad].flatten()]
        data['time'] = data['time'] + dty_itos
        # data['true'] = torch.cat((data['true'], y['id'][:, :T_id - pad].flatten()))
        data['true'] = data['true'] + list(y['id'][:, :T_id - pad].flatten())
        pbar.set_description(f"len pred: {len(data['ID'])}, len true: {len(data['true'])}")

        
    return data
    

@torch.no_grad()
def predict_beam_search(model, loader, stoi, frame_end=0):
    device = 'cpu' # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
    t_id = model.config.id_block_size
    true_raster = []
    predicted_raster = []
    true_timing = []
    context = torch.tensor(0).unsqueeze(0)
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:
        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        # set context vector if beginning
        if it == 0:
            # context = x[:, :, 0].flatten()
            true_raster = context
            predicted_raster = context
            true_timing = context
        true_raster = torch.cat((true_raster, y['id'][:, :t_id - x['pad']].flatten()))
        
        ix = beam_decode(model, stoi, x, frame_end)
        predicted_raster = torch.cat((predicted_raster, ix))
    return true_raster[1:], predicted_raster[1:]


# @torch.no_grad()
# def predict_behavior(model, dataset, itos, sample=False, top_k=0, top_p=0):
#     model.eval()
#     loader = DataLoader(dataset, shuffle=False, pin_memory=False)
#     pbar = tqdm(enumerate(loader), total=len(loader))
#     model.cpu()
#     model.eval()
#     behavior_preds = []
#     intervals = []
#     trials = []
#     behavior_true = []
#     for it, (x, y) in pbar:
#         x = all_device(x, 'cpu')
#         y = all_device(y, 'cpu')
#         logits, features, loss = model(x, y)
#         if top_k or top_p != 0:
#             logits['behavior'] = top_k_top_p_filtering(logits['behavior'], top_k=top_k, top_p=top_p)
#         probs = F.softmax(logits['behavior'], dim=-1)
#         if sample:
#             ix =  torch.multinomial(probs, 1).flatten()
#         else:
#             ix = torch.argmax(probs, dim=-1).flatten()
#         ix_itos = itos[int(ix)]
#         interval = float(x['interval'].flatten())
#         trial = float(x['trial'].flatten())
#         y_behavior = itos[int(y['behavior'])]
#         behavior_preds.append(ix_itos)
#         intervals.append(interval)
#         trials.append(trial)
#         behavior_true.append(y_behavior)
    
#     # make behavior preds, intervals etc into dataframe
#     behavior_preds = pd.DataFrame(behavior_preds)
#     behavior_preds.columns = ['behavior']
#     behavior_preds['interval'] = intervals
#     behavior_preds['trial'] = trials
#     behavior_preds['true'] = behavior_true

#     # make cum interval
#     behavior_preds['cum_interval'] = behavior_preds['interval'].copy()
#     prev_trial = None
#     for trial in behavior_preds['trial'].unique():
#         if prev_trial is None:
#             prev_trial = trial
#             continue
#         else:
#             max_interval = behavior_preds[behavior_preds['trial'] == prev_trial]['interval'].max()
#             behavior_preds.loc[behavior_preds['trial'] >= trial, 'cum_interval'] += max_interval

#     return behavior_preds

@torch.no_grad()
def predict_behavior(model, dataset, itos, sample=False, top_k=0, top_p=0):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=False)
    pbar = tqdm(enumerate(loader), total=len(loader))
    
    behavior_preds = []
    intervals = []
    trials = []
    behavior_true = []
    for it, (x, y) in pbar:
        x = all_device(x, device)
        y = all_device(y, device)
        logits, features, loss = model(x, y)
        # take everything back to cpu
        logits = all_device(logits, 'cpu')
        features = all_device(features, 'cpu')
        loss = all_device(loss, 'cpu')
        
        if top_k or top_p != 0:
            logits['behavior'] = top_k_top_p_filtering(logits['behavior'], top_k=top_k, top_p=top_p)
        probs = F.softmax(logits['behavior'], dim=-1)
        if sample:
            ix = torch.multinomial(probs, 1).flatten()
        else:
            ix = torch.argmax(probs, dim=-1).flatten()
        
        ix_itos = [itos[int(i)] for i in ix]
        interval = x['interval'].flatten().tolist()
        trial = x['trial'].flatten().tolist()
        y_behavior = [itos[int(i)] for i in y['behavior']]
        behavior_preds.extend(ix_itos)
        intervals.extend(interval)
        trials.extend(trial)
        behavior_true.extend(y_behavior)
    
    # make behavior preds, intervals etc into dataframe
    behavior_preds = pd.DataFrame(behavior_preds)
    behavior_preds.columns = ['behavior']
    behavior_preds['interval'] = intervals
    behavior_preds['trial'] = trials
    behavior_preds['true'] = behavior_true

    # make cum interval
    behavior_preds['cum_interval'] = behavior_preds['interval'].copy()
    prev_trial = None
    for trial in behavior_preds['trial'].unique():
        if prev_trial is None:
            prev_trial = trial
            continue
        else:
            max_interval = behavior_preds[behavior_preds['trial'] == prev_trial]['interval'].max()
            behavior_preds.loc[behavior_preds['trial'] >= trial, 'cum_interval'] += max_interval

    return behavior_preds

@torch.no_grad()
def extract_latents(model, dataset):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)

    
    loader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=False)
    pbar = tqdm(enumerate(loader), total=len(loader))
    
    latents = collections.defaultdict(list)
    feats = collections.defaultdict(list)
    intervals = []
    trials = []
    behavior_true = []
    for it, (x, y) in pbar:
        x = all_device(x, device)
        y = all_device(y, device)
        logits, features, loss = model(x, y)
        # take everything back to cpu
        logits = all_device(logits, 'cpu')
        features = all_device(features, 'cpu')
        loss = all_device(loss, 'cpu')
        x = all_device(x, 'cpu')
    
        for modality in features['clip'].keys():
            for idx, (behavior, feat) in enumerate(zip(x['behavior'], features['clip'][modality])):
                # latents[modality][round_n(float(behavior), res)].append(feats)
                latents[modality].append(feat)
        for key in x.keys():
            for idx, _ in enumerate(x[key]):
                feats[key].append(x[key][idx].numpy())

    return feats, latents

@torch.no_grad()
def predict_beam_search_time(model, loader, stoi, itos_dt, frame_end=0):
    device = 'cpu' # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
    t_id = model.config.id_block_size
    true_raster = []
    predicted_raster = []
    true_timing = []
    predicted_timing = []
    context = torch.tensor(0).unsqueeze(0)
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:
        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        # set context vector if beginning
        if it == 0:
            # context = x[:, :, 0].flatten()
            true_raster = context
            predicted_raster = context
            true_timing = context
            predicted_timing = context
        true_raster = torch.cat((true_raster, y['id'][:, :t_id - x['pad']].flatten()))
        true_timing = torch.cat((true_timing, y['dt'][:, :t_id - x['pad']].flatten()))
        
        ix, dt = beam_decode(model, stoi, x)
        predicted_raster = torch.cat((predicted_raster, ix))
        predicted_timing = torch.cat((predicted_timing, dt))
    return true_raster[1:], predicted_raster[1:], true_timing[1:], predicted_timing[1:]
        

# def create_full_trial(df, n_step, n_stim, t_trial, n_start=None, n_trials=1):
#     """
    
#     n_stim: how many stimuli
#     n_step: how many trials per stimulus does dataset have
#     n_start: min trial to start from
#     n_trials: how many trials PER STIMULUS do you want to keep

#     """
#     n_start = df['Trial'].min() if n_start is None else n_start
#     trials = []
#     for n in range(n_trials):
#         df_trial = None
#         n_start += n
#         for i in range(n_stim):
#             t_now =  n_start + (i * n_step)
#             df_t = df[df['Trial'] == t_now]
#             if df_trial is None:
#                 df_trial = df_t
#             else:
#                 t_start = df['Interval'].max()
#                 df_t['Interval'] += t_trial
#                 df_t['Time'] += t_trial
#                 df_trial = pd.concat([df_trial, df_t], ignore_index=True)
#         df_trial['Trial'] = n
#         trials.append(df_trial)
#     return pd.concat(trials, ignore_index=True).sort_values(by=['Trial', 'Time'])

# from utils import *

def get_class_weights(dataset, stoi, stoi_dt):
    dt = []
    id = []
    for x, y in dataset:
        id.extend([stoi['SOS']] + y['id'][:len(y['id']) - x['pad']].flatten().tolist())    # *x['pad']) # -1 in pad to include PAD token
        dt.extend([stoi_dt[0]] + y['dt'][:len(y['dt']) - x['pad']].flatten().tolist())   #*x['pad']) # -1 in pad to include PAD token

    n_samples = len(id)
    n_classes = len(stoi.keys()) - 1

    id = pd.DataFrame(id)
    dt = pd.DataFrame(dt)

    id_freq = id.groupby([0]).size()
    dt_freq = dt.groupby([0]).size()

    id_ones = np.ones(dataset.id_population_size)
    dt_ones = np.ones(dataset.dt_population_size)
    id_freq_max = id_freq[:-1].max()
    dt_freq_max = dt_freq[:-1].max()

    id_ones[id_freq.index] = n_samples / (n_classes *  id_freq)
    dt_ones[dt_freq.index] = n_samples / (n_classes *  dt_freq)
    
    class_freq = dict()
    class_freq['id'] = torch.tensor(id_ones, dtype=torch.float32)
    class_freq['dt'] = torch.tensor(dt_ones, dtype=torch.float32)

    cw_mean = 1 # c_weights.mean()
    cw_shrink = 3/4
    class_freq['id'] = cw_mean + cw_shrink * (class_freq['id'] - cw_mean)
    
    return class_freq 

    class_weights = get_class_weights(train_dataset)

    cmax_weight = class_weights['id'].mean() + (class_weights['id'].std())

    c_weights = class_weights['id']

    cw_mean = 1 # c_weights.mean()
    cw_shrink = 3/4
    c_weights = cw_mean + cw_shrink * (c_weights - cw_mean)

    class_weights['id'] = c_weights
    class_weights['id'] = class_weights['id'].clamp(min=0.5, max=6)

    plt.bar(np.arange(len(class_weights['id'])), class_weights['id'])
    # plt.bar(np.arange(len(c_weights)), c_weights)

def check_common_attrs(*objects):
    if not objects:
        raise ValueError("At least one object must be provided")

    first, *others = objects

    common_attrs = {}
    for attr in dir(first):
        if attr.startswith("__"):  # skip Python internal (dunder) attributes
            continue

        try:
            first_value = getattr(first, attr)
        except AttributeError:
            continue

        common_objects = []
        for obj in others:
            if hasattr(obj, attr) and getattr(obj, attr) != first_value:
                setattr(obj, attr, first_value)
                common_objects.append(obj)

        if common_objects:
            common_attrs[attr] = (first_value, common_objects)

    return common_attrs

# # precision_score = collections.defaultdict(list)
# # recall_score = collections.defaultdict(list)
# # f1_score = collections.defaultdict(list)
# device = 'cuda'
# width = 1
# trials = test_data['Trial'].unique()

# precision = []
# recall = []
# f1 = []
# df_1 = []
# df_2 = []
# for n, trial in enumerate(trials):
#     trial_2 = int(20 * (trial // 20) + np.random.choice([i for i in range(1, 21)], 1))
#     if trial_2 == trial:
#         trial_2 = trial + 1
#     df_data_trial = df[df['Trial'] == trial]
#     df_data_2_trial = df[df['Trial'] == trial_2]
#     df_1.append(df_data_trial)
#     df_2.append(df_data_2_trial)
#     if n > 0 and n % 4 == 0:
#         df_1 = pd.concat(df_1).sort_values(by=['Trial', 'Time'])
#         df_2 = pd.concat(df_2).sort_values(by=['Trial', 'Time'])
#         for n_id in df_data_trial['ID'].unique():
#             spikes_true = df_1['Time'][df_1['ID'] == n_id]
#             spikes_pred = df_2['Time'][df_2['ID'] == n_id]
#             if len(spikes_pred) > 0:
#                 [cos_score, cos_prec, cos_call, y, y_hat, t_y] = compute_score(width, spikes_true, spikes_pred)
#             else:
#                 continue
#             # scores = compute_scores(df_trial_true, df_trial_pred)
            
#             precision.append(cos_prec)
#             recall.append(cos_call)
#             f1.append(cos_score)
#         df_1 = []
#         df_2 = []
#     # for n_id in df_data_trial['ID'].unique():
#     #     # spikes_true = np.array(df_true_trial[df_true_trial['ID'] == n_id]['Time'])
#     #     # spikes_pred = np.array(df_pred_trial[df_pred_trial['ID'] == n_id]['Time'])
#     #     spikes_true = df_data_trial['Time'][df_data_trial['ID'] == n_id]
#     #     spikes_pred = df_data_2_trial['Time'][df_data_2_trial['ID'] == n_id]
#     #     if len(spikes_pred) > 0:
#     #         [cos_score, cos_prec, cos_call, y, y_hat, t_y] = compute_score(width, spikes_true, spikes_pred)
#     #     else:
#     #         cos_score = 0
#     #         cos_prec = 0
#     #         cos_call = 0
#         # precision.append(cos_prec)
#         # recall.append(cos_call)
#         # f1.append(cos_score)
# # precision_score[len(precision_score.keys())].append(np.mean(np.nan_to_num(precision)))
# # recall_score[len(recall_score.keys())].append(np.mean(np.nan_to_num(recall)))
# # f1_score[len(f1_score.keys())].append(np.mean(np.nan_to_num(f1)))

# precision_score['Ground Truth'] = np.mean(np.nan_to_num(precision))
# recall_score['Ground Truth'] = np.mean(np.nan_to_num(recall))
# f1_score['Ground Truth'] = np.mean(np.nan_to_num(f1))

# import cv2
# import numpy as np

# def draw_grid(image, n, m, color=(0, 255, 0), thickness=1):
#     """
#     Draw an n x m grid on an image.

#     Args:
#         image (np.array): Input image. (height, width, channels)
#         n (int): Number of horizontal grid lines.
#         m (int): Number of vertical grid lines.
#         color (tuple, optional): Color of the grid lines. Default is green (0, 255, 0).
#         thickness (int, optional): Thickness of the grid lines. Default is 1.

#     Returns:
#         np.array: Image with the grid drawn.
#     """
#     height, width, _ = image.shape

#     # Calculate the step size for the grid lines
#     step_x = width // (m + 1)
#     step_y = height // (n + 1)

#     # Draw the vertical grid lines
#     for i in range(1, m + 1):
#         cv2.line(image, (i * step_x, 0), (i * step_x, height), color, thickness)

#     # Draw the horizontal grid lines
#     for j in range(1, n + 1):
#         cv2.line(image, (0, j * step_y), (width, j * step_y), color, thickness)

#     return image