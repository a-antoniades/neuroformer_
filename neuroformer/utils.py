import logging
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

def get_model_attr(mconf, tconf):
  n_head = mconf.n_head
  n_block = mconf.n_layer
  nembd = mconf.n_embd
  data = tconf.dataset[-20:-4]
  model_attr =  f"Head:{n_head}_Block{n_block}_nembd:{nembd}_data:{data}"
  return model_attr

def set_plot_params():
    ## fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu mono'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    # # font sizes
    # plt.rcParams['font.size'] = 16
    # plt.rcParams['axes.labelsize'] = 12
    # plt.rcParams['xtick.labelsize'] = 10
    # plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams['legend.fontsize'] = 14
    # plt.rcParams['figure.titlesize'] = 16

    ## colors
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams["figure.facecolor"] = '202020'
    plt.rcParams['axes.facecolor']= '202020'
    plt.rcParams['savefig.facecolor']= '202020'

def set_plot_white():
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

    # Set the font used for MathJax - more on this later
    plt.rc('mathtext',**{'default':'regular'})

    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams["figure.facecolor"] = 'white'
    plt.rcParams['axes.facecolor']= 'white'
    plt.rcParams['savefig.facecolor']= 'white'

def set_plot_black():
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams["figure.facecolor"] = '202020'
    plt.rcParams['axes.facecolor']= '202020'
    plt.rcParams['savefig.facecolor']= '202020'

def plot_losses(trainer): 
    plt.figure(figsize=(20,5))
    
    # plotting train losses
    plt.subplot(1,2,1)
    plt.title('%s training losses' % str(trainer)[1:8])
    for i, losses in enumerate(trainer.train_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")
    
    # plotting testing losses
    plt.subplot(1,2,2)
    plt.title('%s testing losses' % str(trainer)[1:8])
    for i, losses in enumerate(trainer.test_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")

    plt.show()

def plot_losses_wattr(trainer, model_attr): 
    plt.figure(figsize=(20,5))
    
    # plotting train losses
    plt.subplot(1,2,1)
    plt.title('%s training losses' % model_attr)
    for i, losses in enumerate(trainer.train_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")
    
    # plotting testing losses
    plt.subplot(1,2,2)
    plt.title('%s testing losses' % model_attr)
    for i, losses in enumerate(trainer.test_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")

    plt.show()

def print_full(df, length=None):
    length = len(df) if length is None else len(df)
    print(length)
    pd.set_option('display.max_rows', length)
    torch.set_printoptions(threshold=1e3)
    print(df)
    pd.reset_option('display.max_rows')
    torch.set_printoptions(threshold=1e3)

# results = predict_raster_recursive_time_auto(model, loader, window, stoi, itos_dt, sample=True, top_p=0.95, top_p_t=0.95, frame_end=0, get_dt=True, gpu=False)

def process_predictions(results, stoi, window):
    pred_keys = ['ID', 'dt', 'Trial', 'Interval']
    predicted_dict = {k: results[k] for k in results if k in pred_keys}
    df_pred = pd.DataFrame(predicted_dict)
    df_pred['Time'] = df_pred['dt'] + df_pred['Interval'] - 0.5
    df_pred = df_pred[(df_pred['ID'] <= stoi['SOS']) & (df_pred['dt'] <= window) & (df_pred['Time'] >= 0)]
    true_keys = ['true', 'time']
    true_dict = {k: results[k] for k in results if k in true_keys}
    df_true = pd.DataFrame(true_dict)
    df_true = df_true[df_true['true'] <= stoi['SOS']]
    df_true.rename({'true':'ID', 'time':'dt'}, axis=1, inplace=True)
    # df_true['time'] = df_true['dt'] + df_true['interval'] - 0.5

    return df_pred, df_true

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
def predict_raster_recursive_time_auto(model, loader, window, stoi, itos_dt, get_dt=False, sample=False, top_k=0, top_p=0, top_p_t=0, temp=1, temp_t=1, frame_end=0, gpu=False, pred_dt=True):
    """
    predict both ID and dt recursively
    """

    def pad_x(x, length, pad_token):
        """
        pad x with pad_token to length
        """
        pad_n = length - len(x)
        if pad > 0:
            x = x + [pad_token] * pad_n
            x = torch.tensor(x, dtype=torch.long, device=device)
        return x.unsqueeze(0)


    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    tf = frame_end
    T = model.get_block_size() # model.config.id_block_size # model.get_block_size()
    T_id = model.config.id_block_size
    T_id_prev = model.config.prev_id_block_size
    context = torch.tensor(0, device=device).unsqueeze(0)
    data = dict()
    data['true'] = context
    data['ID'] = context
    data['time'] = context
    data['dt'] = context
    data['Trial'] = context
    data['Interval'] = context

    id_prev_stoi = context
    dt_prev_stoi = context
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:

        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        
        if it > 12:
            x['id_prev'] = [stoi['SOS']] + id_prev_stoi[-(T_id_prev - 2):].tolist()     # + [stoi['EOS']]
            x['id_prev'] = pad_x(x['id_prev'], T_id_prev, stoi['PAD'])
            if pred_dt:
                x['dt_prev'] = [0] + dt_prev_stoi[-(T_id_prev - 2):].tolist()           # + [max(list(itos_dt.keys()))]
                x['dt_prev'] = pad_x(x['dt_prev'], T_id_prev, max(list(itos_dt.keys())))
        
        pad = x['pad'] if 'pad' in x else 0
        x['id_full'] = x['id'][:, 0]
        x['id'] = x['id'][:, 0]
        x['dt_full'] = x['dt'][:, 0] if pred_dt else x['dt']
        x['dt'] = x['dt'][:, 0] if pred_dt else x['dt']

        current_id_stoi = torch.empty(0, device=device)
        current_dt_stoi = torch.empty(0, device=device)
        for i in range(T_id):
            t_pad = torch.tensor([stoi['PAD']] * (T_id - x['id_full'].shape[-1]), device=device)
            t_pad_dt = torch.tensor([0] * (T_id - x['dt_full'].shape[-1]), device=device)
            x['id'] = torch.cat((x['id_full'], t_pad)).unsqueeze(0).long()
            x['dt'] = torch.cat((x['dt_full'], t_pad_dt)).unsqueeze(0).long() if pred_dt else x['dt']

            logits, features, _ = model(x)
            logits['id'] = logits['id'][:, tf + i] / temp
            logits['dt'] = logits['dt'][:, tf + i] / temp_t
            
            # optionally crop probabilities to only the top k / p options
            if top_k or top_p != 0:
                logits['id'] = top_k_top_p_filtering(logits['id'], top_k=top_k, top_p=top_p)
                logits['dt'] = top_k_top_p_filtering(logits['dt'], top_k=top_k, top_p=top_p_t)

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
            
            # if ix > stoi['PAD']:
            #     ix = torch.tensor([513])
            
            # convert ix_dt to dt and add to current time
            current_id_stoi = torch.cat((current_id_stoi, ix.flatten()))
            current_dt_stoi = torch.cat((current_dt_stoi, ix_dt.flatten()))
            dtx = torch.tensor(itos_dt[int(ix_dt.flatten())], device=device).unsqueeze(0)
            
            # append true and predicted in lists
            # get last unpadded token
            data['ID'] = torch.cat((data['ID'], ix.flatten()))
            data['dt'] = torch.cat((data['dt'], dtx))
            data['Trial'] = torch.cat((data['Trial'], x['trial']))
            data['Interval'] = torch.cat((data['Interval'], x['interval']))
            x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
            x['dt_full'] = torch.cat((x['dt_full'], ix_dt.flatten())) if pred_dt else x['dt']

            if ix == stoi['EOS'] or len(data['ID']) == T: # and dtx == 0.5:    # dtx >= window:   # ix == stoi['EOS']:
            # if len(current_id_stoi) == T_id - x['pad']:
                # if ix != stoi['EOS']:
                #     torch.cat((current_id_stoi, torch.tensor([stoi['EOS']])))
                # if dtx <= window:
                #     torch.cat((current_dt_stoi, torch.tensor([max(list(itos_dt.keys()))])))
                id_prev_stoi = current_id_stoi
                dt_prev_stoi = current_dt_stoi
                break
            
        dty = torch.tensor([itos_dt[int(dt)] for dt in y['dt'][:, :T_id - pad].flatten()], device=device)
        # dty = torch.tensor(itos_dt[y['dt'][:, i].item()]).unsqueeze(0)
        data['time'] = torch.cat((data['time'], dty))   
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
        
        ix, dt = beam_decode(model, stoi, itos_dt, x, frame_end)
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
