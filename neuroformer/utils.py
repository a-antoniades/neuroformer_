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
from tqdm import tqdm
from beam_search import beam_decode
logger = logging.getLogger(__name__)

from model_neuroformer import GPT



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


# results = predict_raster_recursive_time_auto(model, loader, window, stoi, itos_dt, sample=True, top_p=0.95, top_p_t=0.95, frame_end=0, get_dt=True, gpu=False)

def process_predictions(results, stoi, itos, window):
    pred_keys = ['ID', 'dt', 'Trial', 'Interval']
    predicted_dict = {k: results[k] for k in results if k in pred_keys}
    df_pred = pd.DataFrame(predicted_dict)
    df_pred['Time'] = df_pred['dt'] + df_pred['Interval']
    df_pred = df_pred[df_pred['Interval'] > 0]
    # df_pred = df_pred[(df_pred['ID'] <= stoi['SOS']) & (df_pred['dt'] <= window) & (df_pred['Time'] >= 0)]
    true_keys = ['true', 'time']
    true_dict = {k: results[k] for k in results if k in true_keys}
    df_true = pd.DataFrame(true_dict)
    if 'SOS' in stoi:
        # sos_id = list(itos.keys())[list(itos.values()).index('SOS')]
        sos_id = stoi['SOS']
        n_sos = len(df_true[df_true['true'] == sos_id])
        print(f'SOS fouuuund: {n_sos}')
        df_true = df_true[df_true['true'] != sos_id]
    if 'EOS' in stoi:
        # eos_id = list(itos.keys())[list(itos.values()).index('EOS')]
        eos_id = stoi['EOS']
        n_eos = len(df_true[df_true['true'] == eos_id])
        print(f'EOS fouuuund: {n_eos}')
        df_true = df_true[df_true['true'] != eos_id]
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
def predict_raster(model, loader, frame_end=0, get_dt=False, gpu=False):
    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
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
        preds, features, _ = model(x)
        PAD = x['pad']
        b, t = x['id'].size()
        logits = preds['logits'][:, frame_end:frame_end + t - PAD, :]    # get last unpadded token (-x['pad'])
        # take logits of final step and apply softmax
        probs = F.softmax(logits, dim=-1)
        # choose highest topk (1) sample
        _, ix = torch.topk(probs, k=1, dim=-1)
        # append true and predicted in lists
        true_raster = torch.cat((true_raster.to('cpu'), y['id'][:, :t - PAD].flatten().to('cpu')))   # get last unpadded token
        predicted_raster = torch.cat((predicted_raster.to('cpu'), ix.flatten().to('cpu')))
        if get_dt:
            true_timing = torch.cat((true_timing.to('cpu'), y['time'][:, :t - PAD].flatten().to('cpu')))
    return true_raster[1:], predicted_raster[1:], true_timing[1:]

@torch.no_grad()
def predict_raster_recursive(model, loader, stoi, get_dt=False, sample=False, top_k=0, top_p=0, frame_end=0, id_prev=0, gpu=False):
    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    tf = frame_end
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
        
        t = x['id'].shape[-1]
        pad = x['pad'].to(device)
        x['id_full'] = x['id'][:, 0].to(device)
        x['id'] = x['id'][:, 0].to(device)
        for i in range(t - pad):
            t_pad = torch.tensor([stoi['PAD']] * (t - x['id_full'].shape[-1])).to(device)
            x['id'] = torch.cat((x['id_full'], t_pad)).unsqueeze(0)
            preds, features, _ = model(x)
            logits = preds['logits'][:, tf + i]
            # optionally crop probabilities to only the top k options
            if top_k or top_p != 0:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            # apply softmax to logits
            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                # ix = torch.poisson(torch.exp(logits), num_samples=1)
            else:
                # choose highest topk (1) sample
                _, ix = torch.topk(probs, k=1, dim=-1)
            if ix > stoi['PAD']:
                ix = torch.tensor([513])
            if 'prev_int' in x and i <= x['prev_int'] + 1:
                continue
            # append true and predicted in lists
            true_raster = torch.cat((true_raster.to('cpu'), y['id'][:, i].flatten().to('cpu')))   # get last unpadded token
            predicted_raster = torch.cat((predicted_raster.to('cpu'), ix.flatten().to('cpu')))
            x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
            # if get_dt:
            #     true_timing = torch.cat((true_timing.to('cpu'), y['dt'][:, i].flatten().to('cpu')))
        if get_dt:
            true_timing = torch.cat((true_timing.to('cpu'), y['time'][:, :t - pad].flatten().to('cpu')))
    return true_raster[1:], predicted_raster[1:], true_timing[1:]

@torch.no_grad()
def predict_raster_recursive_time(model, loader, stoi, itos_dt, get_dt=False, sample=False, top_k=0, top_p=0, frame_end=0, gpu=False):
    """
    predict both ID and dt recursively
    """
    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    tf = frame_end
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
    t_id = model.config.id_block_size
    context = torch.tensor(0).unsqueeze(0)
    data = dict()
    data['true'] = context
    data['pred'] = context
    data['time'] = context
    data['time_pred'] = context
    data['trial'] = context
    data['interval'] = context
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:
        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        # set context vector if beginning
        
        t = x['id'].shape[-1]
        pad = x['pad'] if 'pad' in x else 0
        x['id_full'] = x['id'][:, 0]
        x['id'] = x['id'][:, 0]
        x['dt_full'] = x['dt'][:, 0]
        x['dt'] = x['dt'][:, 0]
        for i in range(t - pad):
            t_pad = torch.tensor([stoi['PAD']] * (t - x['id_full'].shape[-1]))
            t_pad_dt = torch.tensor([0] * (t - x['dt_full'].shape[-1]))
            x['id'] = torch.cat((x['id_full'], t_pad)).unsqueeze(0)
            x['dt'] = torch.cat((x['dt_full'], t_pad_dt)).unsqueeze(0)
            logits, features, _ = model(x)
            logits['id'] = logits['id'][:, tf + i]
            logits['dt'] = logits['dt'][:, tf + i]
            trial = x['trial'][:, tf + i]
            interval = x['interval'][:, tf + i]
            
            # optionally crop probabilities to only the top k / p options
            if top_k or top_p != 0:
                logits['id'] = top_k_top_p_filtering(logits['id'], top_k=top_k, top_p=top_p)
                logits['dt'] = top_k_top_p_filtering(logits['dt'], top_k=top_k, top_p=0.4)

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
            
            if ix > stoi['PAD']:
                ix = torch.tensor([513])
            # append true and predicted in lists
            
            data['true'] = torch.cat((data['true'].to('cpu'), y['id'][:, i].flatten().to('cpu')))   # get last unpadded token
            data['pred'] = torch.cat((data['pred'].to('cpu'), ix.flatten().to('cpu')))
            data['time_pred'] = torch.cat((data['time_pred'], ix_dt.flatten()))
            data['trial'] = torch.cat((data['trial'].to('cpu'), trial.flatten().to('cpu')))
            data['interval'] = torch.cat((data['interval'].to('cpu'), interval.flatten().to('cpu')))
            x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
            x['dt_full'] = torch.cat((x['dt_full'], ix_dt.flatten()))
        
            if get_dt:
                ix_dt_y = torch.tensor(itos_dt[y['dt'][:, i].item()]).unsqueeze(0)
                data['time'] = torch.cat((data['time'].to('cpu'), ix_dt_y)) 
    return data

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


from torch.utils.data.dataloader import DataLoader
from SpikeVidUtils import get_interval, round_n
from SpikeVidUtils import SpikeTimeVidData2 as SP
@torch.no_grad()
def predict_raster_recursive_time_auto(model, dataset, window, window_prev, stoi, itos_dt, itos=None, 
                                      get_dt=False, sample=False, top_k=0, top_p=0, top_p_t=0, temp=1, temp_t=1, 
                                      frame_end=0, gpu=False, pred_dt=True, p_bar=False, plot_probs=False):    
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

    stoi_dt = {v: k for k, v in itos_dt.items()}
    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = [model_n.to(device) for model_n in model] if isinstance(model, list) else model.to(device) 
    model = [model_n.eval() for model_n in model] if isinstance(model, list) else model.eval()
    tf = 0
    mconf = model[0].config if isinstance(model, list) else model.config
    T_id = mconf.id_block_size
    T_id_prev = mconf.prev_id_block_size
    
    context = torch.tensor(0, device=device).unsqueeze(0)
    data = dict()
    data['true'] = context
    data['ID'] = context
    data['time'] = context
    data['dt'] = context
    data['Trial'] = context
    data['Interval'] = context

    id_prev_stoi_buffer = [stoi['SOS']]
    dt_prev_stoi_buffer = [float(context)]
    loader = DataLoader(dataset, shuffle=False, pin_memory=False)
    pbar = tqdm(enumerate(loader), total=len(loader), disable=p_bar)
    for it, (x, y) in pbar:
        # if it > 2:
        #     break
        # print(f"it = {it}, interval: {x['interval']}, window_prev: {window_prev}, window: {window}")

        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        
        # if x['interval'] > window_prev + 2:
        #     data['Time'] = data['dt'] + data['Interval']
        #     df = {k: v for k, v in data.items() if k in ('ID', 'dt', 'Trial', 'Interval', 'Time')}
        #     df = pd.DataFrame(df)
        #     prev_id_interval, current_id_interval = dataset.calc_intervals(x['interval'])
        #     x['id_prev'], x['dt_prev'], pad_prev = dataset.get_interval(prev_id_interval, float(x['trial']), T_id_prev)
        #     x['id_prev'] = torch.tensor(x['id_prev'], dtype=torch.long).unsqueeze(0).to(device)
        #     x['dt_prev'] = torch.tensor(x['dt_prev'], dtype=torch.long).unsqueeze(0).to(device)
            
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
            # x['id'] = torch.cat((x['id_full'], t_pad)).unsqueeze(0).long()
            # x['dt'] = torch.cat((x['dt_full'], t_pad_dt)).unsqueeze(0).long() if pred_dt else x['dt']

            # print(x['id'], x['dt'])
            # forward model, if list of models, then ensemble
            if isinstance(model, list):
                logits = model_ensemble(model, x)
            else:
                logits, features, _ = model(x)
            
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
                print(x['id'])
                print(f"i: {i} ix: {ix}, x_true: {y['id'][0, i]}")
                probs_n = np.array(probs)[0]
                xaxis = np.arange(len(probs_n))
                topk=5
                topk_indices = np.argpartition(probs_n, -topk)[-topk:]
                topk_probs = probs_n[topk_indices]
                plt.figure()
                plt.title(f"t={i}, indices: {topk_indices}")
                plt.bar(xaxis, probs_n)
                plt.show()
            
            # convert ix_dt to dt and add to current time
            # print(f"ix: {ix}, x_true: {y['id'][0, i]} ")
            current_id_stoi = torch.cat((current_id_stoi, ix.flatten()))
            current_dt_stoi = torch.cat((current_dt_stoi, ix_dt.flatten()))
            dtx_itos = torch.tensor(itos_dt[int(ix_dt.flatten())], device=device).unsqueeze(0)

            x['id'][:, i + 1] = ix.flatten()
            x['dt'][:, i + 1] = ix_dt.flatten() if pred_dt else x['dt']
           
            if ix >= stoi['EOS']:    # T_id - int(x['pad']):   # or len(current_id_stoi) == T_id: # and dtx == 0.5:    # dtx >= window:   # ix == stoi['EOS']:
                # print(f"n_regres_block: {i}")
                break
            
            n_ix = sum([1 for i in current_id_stoi if (i == stoi['EOS']) or (i == stoi['PAD'])])
            id_prev_stoi_buffer.extend(ix.flatten())
            dt_prev_stoi_buffer.extend(ix_dt.flatten())
            try:
                ix_itos = torch.tensor(itos[int(ix.flatten())]).unsqueeze(0)
            except:
                TypeError(f"ix: {ix}, itos: {itos}")
            data['ID'] = torch.cat((data['ID'], ix_itos))
            data['dt'] = torch.cat((data['dt'], dtx_itos))
            data['Trial'] = torch.cat((data['Trial'], x['trial']))
            data['Interval'] = torch.cat((data['Interval'], x['interval']))

            # x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
            # x['dt_full'] = torch.cat((x['dt_full'], ix_dt.flatten())) if pred_dt else x['dt']


        dty_itos = torch.tensor([itos_dt[int(dt)] for dt in y['dt'][:, :T_id - pad].flatten()], device=device)
        data['time'] = torch.cat((data['time'], dty_itos))   
        data['true'] = torch.cat((data['true'], y['id'][:, :T_id - pad].flatten()))
        pbar.set_description(f"len pred: {len(data['ID'])}, len true: {len(data['true'])}")

    for key, value in data.items():
        data[key] = data[key].to("cpu")
        
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
        

# @torch.no_grad()
# def predict_raster_enc_dec(model, loader, frame_end=0, get_dt=False):
#     device = 'cpu' # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     model.eval()
#     T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
#     t_id = model.config.id_block_size
#     true_raster = []
#     predicted_raster = []
#     true_timing = []
#     context = torch.tensor(0).unsqueeze(0)
#     pbar = tqdm(enumerate(loader), total=len(loader))
#     for it, (x, y) in pbar:
#         for key, value in x.items():
#             x[key] = x[key].to(device)
#         for key, value in y.items():
#             y[key] = y[key].to(device)
#         # set context vector if beginning
#         if it == 0:
#             # context = x[:, :, 0].flatten()
#             true_raster = context
#             predicted_raster = context
#             true_timing = context
#         preds, features, _ = model(x)
#         PAD = x['pad']
#         logits = preds['logits'][:, 0:T - PAD, :]    # get last unpadded token (-x['pad'])
#         # take logits of final step and apply softmax
#         probs = F.softmax(logits, dim=-1)
#         # choose highest topk (1) sample
#         _, ix = torch.topk(probs, k=1, dim=-1)
#         # append true and predicted in lists
#         true_raster = torch.cat((true_raster.to('cpu'), y['id'][:, :t_id - PAD].flatten().to('cpu')))   # get last unpadded token
#         predicted_raster = torch.cat((predicted_raster.to('cpu'), ix[:, :t_id - PAD].flatten().to('cpu')))
#         if get_dt:
#             true_timing = torch.cat((true_timing.to('cpu'), y['time'][:, :t_id - PAD].flatten().to('cpu')))
#     return true_raster[1:], predicted_raster[1:], true_timing[1:]


@torch.no_grad()
def predict_raster_enc_dec(model, loader, frame_end=0, get_dt=False):
    device = 'cpu' # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
    t_id = model.config.id_block_size
    tf = frame_end
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
        preds, features, _ = model(x)
        PAD = x['pad']
        logits = preds['logits'] # get last unpadded token (-x['pad'])
        # take logits of final step and apply softmax
        probs = F.softmax(logits, dim=-1)
        # choose highest topk (1) sample
        _, ix = torch.topk(probs, k=1, dim=-1)
        # append true and predicted in lists
        true_raster = torch.cat((true_raster.to('cpu'), y['id'][:, :t_id - PAD].flatten().to('cpu')))   # get last unpadded token
        predicted_raster = torch.cat((predicted_raster.to('cpu'), ix[:, :t_id - PAD].flatten().to('cpu')))
        if get_dt:
            true_timing = torch.cat((true_timing.to('cpu'), y['time'][:, :t_id - PAD].flatten().to('cpu')))
    return true_raster[1:], predicted_raster[1:], true_timing[1:]


@torch.no_grad()
def predict_raster_resnet(model, loader, frame_end=0, get_dt=False):
    device = 'cpu' # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
    true_raster = []
    predicted_raster = []
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
            timing = context
        preds, features, _ = model(x)
        PAD = x['pad']
        b, t = x['id'].size()
        logits = preds['logits'][:, frame_end:t - PAD, :]    # get last unpadded token (-x['pad'])
        # take logits of final step and apply softmax
        probs = F.softmax(logits, dim=-1)
        # choose highest topk (1) sample
        _, ix = torch.topk(probs, k=1, dim=-1)
        # append true and predicted in lists
        true_raster = torch.cat((true_raster.to('cpu'), y['modes'][:, frame_end:t - PAD].flatten().to('cpu')))   # get last unpadded token
        predicted_raster = torch.cat((predicted_raster.to('cpu'), ix.flatten().to('cpu')))
    return true_raster[1:], predicted_raster[1:]


@torch.no_grad()
def predict_raster_hungarian(model, loader, itos_dt, top_k=0, top_p=0, temp=1, device='cpu'):
    model = model.to(device)
    model.eval()
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
    context = torch.tensor(0, device=device).unsqueeze(0)
    data = dict()
    data['true'] = context
    data['ID'] = context
    data['time'] = context
    data['dt'] = context
    data['Trial'] = context
    data['Interval'] = context
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:
        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        # set context vector if beginning
        logits, features, _ = model(x)
        logits['id'] = logits['id'] / temp
        logits['dt'] = logits['dt'] / temp
        # take logits of final step and apply softmax
        ix = None
        ix_dt = None
        if top_k or top_p != 0:
            for step in range(logits['id'].shape[1]):
                logits_step = logits['id'][:, step]
                logits_step_dt = logits['dt'][:, step]
                logits_step = top_k_top_p_filtering(logits_step, top_k=top_k, top_p=top_p)
                logits_Step_dt = top_k_top_p_filtering(logits_step_dt, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits_step, dim=-1)
                probs_dt = F.softmax(logits_Step_dt, dim=-1)
                ix_step = torch.multinomial(probs, num_samples=1)
                ix_step_dt = torch.multinomial(probs_dt, num_samples=1)
                ix = ix_step if ix is None else torch.cat((ix, ix_step))
                dtx_step_dt = torch.tensor(itos_dt[int(ix_step_dt.flatten())], device=device).unsqueeze(0)
                ix_dt = dtx_step_dt if ix_dt is None else torch.cat((ix_dt, dtx_step_dt))

        # choose highest topk (1) sample
        # append true and predicted in lists
        data['true'] = torch.cat((data['true'], y['id'].flatten()))   # get last unpadded token
        data['ID'] = torch.cat((data['ID'], ix.flatten()))
        data['dt'] = torch.cat((data['dt'], ix_dt))
        data['Trial'] = torch.cat((data['Trial'], x['trial'].flatten().repeat(len(ix.flatten()))))
        data['Interval'] = torch.cat((data['Interval'], x['interval'].flatten().repeat(len(ix.flatten()))))
        data['time'] = torch.cat((data['time'], y['dt'].flatten())) 
        pbar.set_description(f"len pred: {len(data['ID'])}, len true: {len(data['true'])}")

    for key, value in data.items():
        data[key] = data[key].to("cpu")

    return data

@torch.no_grad()
def predict_time_raster(model, loader, f_block_sz, id_block_sz, block_size=None, get_dt=False, gpu=False):
    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    T = model.get_block_size() if block_size is None else block_size # model.config.id_block_size # model.get_block_size()
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
        preds, features, _ = model(x)
        PAD = x['pad']
        logits = preds['logits'][:, f_block_sz:T - PAD, :]    # get last unpadded token (-x['pad'])
        # take logits of final step and apply softmax
        probs = F.softmax(logits, dim=-1)
        # choose highest topk (1) sample
        _, ix = torch.topk(probs, k=1, dim=-1)
        # append true and predicted in lists
        true_raster = torch.cat((true_raster.to('cpu'), y['modes'][:, f_block_sz:T - PAD].flatten().to('cpu')))   # get last unpadded token
        predicted_raster = torch.cat((predicted_raster.to('cpu'), ix.flatten().to('cpu')))
        true_timing = torch.cat((true_timing.to('cpu'), y['dt'][:, :id_block_sz - PAD].flatten().to('cpu')))
        predicted_timing = torch.cat((predicted_timing.to('cpu'), preds['dt'][:, :id_block_sz - PAD].flatten().to('cpu')))
    return true_raster[1:], predicted_raster[1:], true_timing[1:], predicted_timing[1:]

@torch.no_grad()
def predict_intervals(model, loader, steps):
    true_trace = []
    predicted_trace = []
    model.eval()
    context = 0
    for it, (x, y) in enumerate(loader):
        # set context vector if beginning
        if it == 0:
            context = x.flatten()
            true_trace = context
            predicted_trace = context
        logits, _ = model(x)
        # take logits of final step
        logits = logits[:, -1, :]
        # apply softmax
        probs = F.softmax(logits, dim=-1)
        # choose highest topk (1) sample
        _, ix = torch.topk(probs, k=1, dim=-1)
        # append true and predicted in lists
        true_trace = torch.cat((true_trace, y[:, -1]))
        predicted_trace = torch.cat((predicted_trace, ix.flatten()))
        if it > steps:
            return true_trace[model.config.block_size:], predicted_trace[model.config.block_size:]

@torch.no_grad()
def predict_and_plot_time(model, loader, config):

    true, predicted, true_timing, predicted_timing = predict_time_raster(model, loader, 
                                                                        config.frame_block_size, config.frame_block_size,
                                                                        gpu=True)

    def build_time_seq(time_list):
        times = []
        current_time = 0
        for dt in time_list:
            if dt == 0:
                dt = current_time
            times.append(dt)
        return times

    predicted_time = build_time_seq(predicted_timing)
    true_time = build_time_seq(true_timing)

    def get_ids(data):
        arr = np.where(data <= config.id_vocab_size, data, np.nan)
        arr_nan = np.isnan(arr)
        arr_not_nan = ~ arr_nan
        return arr[arr_not_nan]

    id_true = get_ids(true.flatten().numpy())
    id_predicted = get_ids(predicted.flatten().numpy())
    len_pred = len(true)
    plt.figure(figsize=(20,20))
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.title(f'Test ID Raster', size=20)
    plt.xlabel('Time', size=20)
    plt.ylabel('Response', size=20)
    plt.scatter(true_time[:len(id_true)], id_true, alpha=0.7, label='true', s=75)
    plt.scatter(predicted_time[:len(id_predicted)], id_predicted, alpha=0.6, label='predicted', marker='x', s=75)
    plt.legend()
    plt.tight_layout()
    plt.show()


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