import collections
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.special import kl_div

from torch.utils.data.dataloader import DataLoader

from SpikeVidUtils import SpikeTimeVidData2
from utils import process_predictions, predict_raster_recursive_time_auto



def get_rates(df, intervals):
    df_true = df.groupby(['True', 'Interval']).count().unstack(fill_value=0).stack()['Predicted']
    df_pred = df.groupby(['Predicted', 'Interval']).count().unstack(fill_value=0).stack()['True']
    def set_rates(df, id, intervals):
        df = df[id]
        rates = np.zeros_like(intervals)
        for i in df.index:
            n = int((i * 2) - 1)
            rates[n] = df[i]            
        return rates
    rates_true = dict()
    rates_pred = dict()
    for id in list(set(df['True'].unique()) & set(df['Predicted'].unique())):
        rates_true[id] = set_rates(df_true, id, intervals)
        rates_pred[id] = set_rates(df_pred, id, intervals)
    return rates_true, rates_pred

def get_rates(df, ids, intervals, interval='Interval'):
    intervals = np.array(intervals)
    df = df.groupby(['ID', interval]).count().unstack(fill_value=0).stack()['Time']
    def set_rates(df, id, intervals):
        rates = np.zeros_like(intervals, dtype=np.float32)
        if id not in df.index:
            return rates
        else:
            df = df[id]
            for i in df.index:
                if i in intervals:
                    n = np.where(intervals == i)[0][0]
                    rates[n] = df.loc[i]            
            return rates
    rates = dict()
    for id in ids:
        rates[id] = set_rates(df, id, intervals)
    return rates

def calc_corr_psth(rates1, rates2):
    pearson_r = dict()
    for id in list((set(rates1.keys()) & set(rates2.keys()))):
        pearson_r[id] = stats.pearsonr(rates1[id], rates2[id])[0]
    # pearson_r = dict(sorted(pearson_r.items(), reverse=True, key=lambda item: item[1]))
    pearson_r = pd.DataFrame(pearson_r, index=['pearson_r']).T.sort_values(by=['pearson_r'], ascending=False)
    return pearson_r

def get_rates_trial(df, intervals):
    intervals = np.array(intervals)
    df_rates = df.groupby(['ID', 'Interval']).count().unstack(fill_value=0).stack()
    def set_rates(df, id, intervals):
        df = df.loc[id]
        rates = np.zeros_like(intervals, dtype=np.float32)
        for i in df.index:
            if i not in intervals:
                continue
            n = np.where(intervals == i)[0][0]
            rates[n] = df['Time'].loc[i] 
        return rates
    rates = dict()
    for id in list(set(df['ID'].unique())):
        rates_id = set_rates(df_rates, id, intervals)
        rates[id] = rates_id
    return rates

def label_data(data):
    data['label'] = pd.Categorical(data['ID'], ordered=True).codes
    return data

def get_accuracy(true, pred):
    precision = []
    intervals = pred[['Interval', 'Trial']].drop_duplicates().reset_index(drop=True)
    pred[['Interval', 'Trial']].drop_duplicates().reset_index(drop=True)
    for idx in range(len(intervals)):
        interval = intervals.iloc[idx][0]
        trial = intervals.iloc[idx][1]
        true_int = true[(true['Interval'] == interval) & (true['Trial'] == trial)]
        pred_int = pred[(pred['Interval'] == interval) & (pred['Trial'] == trial)]
        
        set_true = set(true_int['ID'])
        set_pred = set(pred_int['ID'])

        common_set = set_true & set_pred
        uncommon_set = set_true | set_pred
        precision.append(len(common_set) / (len(common_set) + len(uncommon_set)))
    precision = sum(precision) / len(precision)
    
    return precision


def compute_scores(true, pred):
    scores = collections.defaultdict(list)
    intervals = pred[['Interval', 'Trial']].drop_duplicates().reset_index(drop=True)
    pred[['Interval', 'Trial']].drop_duplicates().reset_index(drop=True)
    for idx in range(len(intervals)):
        interval = intervals.iloc[idx][0]
        trial = intervals.iloc[idx][1]
        true_int = true[(true['Interval'] == interval) & (true['Trial'] == trial)]
        pred_int = pred[(pred['Interval'] == interval) & (pred['Trial'] == trial)]
        
        set_true = set(true_int['ID'])
        set_pred = set(pred_int['ID'])

        true_positives = set_true & set_pred
        false_positives = set_pred - set_true
        false_negatives = set_true - set_pred
        if 0 not in {len(true_positives), len(false_positives), len(false_negatives)}:
            scores['precision'].append(len(true_positives) / (len(true_positives) + len(false_positives)))
            scores['recall'].append(len(true_positives) / (len(true_positives) + len(false_negatives)))
        else:
            scores['precision'].append(0)
            scores['recall'].append(0)
        if (scores['precision'][idx] + scores['recall'][idx]) == 0:
            scores['F1'].append(0)
        else:
            scores['F1'].append(2 * (scores['precision'][idx] * scores['recall'][idx]) / (scores['precision'][idx] + scores['recall'][idx]))
    for score in scores.keys():
        scores[score] = sum(scores[score]) / len(scores[score])

    return scores


def compute_score(width, t_k, tt_k):
    
    """ Applies the CosMIC metric.
    
    Arguments:
        width - The width of the triangular pulse with which the spike trains are convolved.
        t_k   - The true spike times.
        tt_k  - The estimated spike times. 
        
    Outputs:
        score         - Value of CosMIC score
        cos_precision - Value of the ancestor metric analogous to the precision
        cos_recall    - Value of the ancestor metric analogous to the recall
        y             - Pulse train (membership function) generated by convolution of true spike train and triangular pulse
        y_hat         - Pulse train (membership function) generated by convolution of true spike train and triangular pulse   
    """
    
    K     = len(t_k)
    K_hat = len(tt_k)
    
    if K == 0 or K_hat == 0:
        
        score         = 0
        cos_recall    = 0
        cos_precision = 0
        y             = []
        y_hat         = []
        
    else:
        
        # get time stamps of membership functions (pulse trains)
        dt      = width/50
        t_lower = min(min(t_k), min(tt_k)) + width     
        t_upper = max(max(t_k), max(tt_k)) + width         
        t       = np.arange(t_lower, t_upper, dt)
        t_len   = len(t)
        
        # maximum offset from spike to receive non-zero score
        dist    = width/2     
        
        # get membership function (pulse train) of true spikes
        t_mat                         = np.array([t,] * K)
        t_k_mat                       = np.array([t_k,] * t_len).transpose()
        time_delay                    = abs(np.subtract(t_mat, t_k_mat))
        time_delay[time_delay > dist] = dist
        weight                        = 1 - time_delay/dist
        y                             = np.sum(weight, axis = 0)
        
        # get membership function (pulse train) of estimated spikes
        t_mat                         = np.array([t,] * K_hat)
        tt_k_mat                      = np.array([tt_k,] * t_len).transpose()
        time_delay                    = abs(np.subtract(t_mat, tt_k_mat))
        time_delay[time_delay > dist] = dist
        weight                        = 1 - time_delay/dist
        y_hat                         = np.sum(weight, axis = 0)
                     
        # calculate scores
        intersection  = np.minimum(y, y_hat)
        score         = 2 * np.sum(intersection)/(np.sum(y) + np.sum(y_hat))
    
        # calculate scores of ancestor metrics
        cos_recall    = np.sum(intersection)/np.sum(y)
        cos_precision = np.sum(intersection)/np.sum(y_hat)

    
    return score, cos_precision, cos_recall, y, y_hat, t


def get_scores(model, width, data, trials, stoi, itos_dt, window, window_prev, device):
    precision = []
    recall = []
    f1 = []
    mconf = model.config
    id_block_size = mconf.id_block_size
    frame_block_size = mconf.frame_block_size
    prev_id_block_size = mconf.prev_id_block_size
    for trial in trials:
        df_trial_true = data[data['Trial'] == trial]
        trial_dataset = SpikeTimeVidData2(df_trial_true, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False)
        trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
        results_trial = predict_raster_recursive_time_auto(model, trial_loader, window, window_prev, stoi, itos_dt, sample=True, top_p=0.9, top_p_t=0.95, temp=1, temp_t=1, frame_end=0, get_dt=True, gpu=False, pred_dt=True)
        df_trial_pred, _ = process_predictions(results_trial, stoi, window)
        for n_id in df_trial_true['ID'].unique():
            # spikes_true = np.array(df_true_trial[df_true_trial['ID'] == n_id]['Time'])
            # spikes_pred = np.array(df_pred_trial[df_pred_trial['ID'] == n_id]['Time'])
            spikes_true = df_trial_true['Time'][df_trial_true['ID'] == n_id]
            spikes_pred = df_trial_pred['Time'][df_trial_pred['ID'] == n_id]
            if len(spikes_pred) > 0:
                [cos_score, cos_prec, cos_call, y, y_hat, t_y] = compute_score(width, spikes_true, spikes_pred)
            else:
                cos_score = 0
                cos_prec = 0
                cos_call = 0
        # scores = compute_scores(df_trial_true, df_trial_pred)
        
        precision.append(scores['precision'])
        recall.append(scores['recall'])
        f1.append(scores['F1'])
    av_precision = np.mean(np.nan_to_num(precision))
    av_recall = np.mean(np.nan_to_num(recall))
    av_f1 = np.mean(np.nan_to_num(f1))
    # precision_score[len(precision_score.keys())].append(np.mean(np.nan_to_num(precision)))
    # recall_score[len(recall_score.keys())].append(np.nan_to_num(np.mean(recall)))
    # f1_score[len(f1_score.keys())].append(np.mean(np.nan_to_num(f1)))

    return av_precision, av_recall, av_f1

def get_scores(model, width, data, trials, stoi, itos_dt, window, window_prev, device):
    precision = []
    recall = []
    f1 = []
    df_true = []
    df_pred = []
    mconf = model.config
    id_block_size = mconf.id_block_size
    frame_block_size = mconf.frame_block_size
    prev_id_block_size = mconf.prev_id_block_size
    for n, trial in enumerate(trials):
        df_trial_true = data[data['Trial'] == trial]
        trial_dataset = SpikeTimeVidData2(df_trial_true, None, mconf.block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False)
        trial_loader = DataLoader(trial_dataset, shuffle=False, pin_memory=False)
        results_trial = predict_raster_recursive_time_auto(model, trial_loader, window, window_prev, stoi, itos_dt, sample=True, top_p=0.9, top_p_t=0.95, temp=1, temp_t=1, frame_end=0, get_dt=True, gpu=False, pred_dt=True)
        df_trial_pred, _ = process_predictions(results_trial, stoi, window)
        df_true.append(df_trial_true)
        df_pred.append(df_trial_pred)
        if n > 0 and n % 4 == 0:
            df_true = pd.concat(df_true).sort_values(by=['Trial', 'Time'])
            df_pred = pd.concat(df_pred).sort_values(by=['Trial', 'Time'])
            for n_id in df_trial_true['ID'].unique():
                # spikes_true = np.array(df_true_trial[df_true_trial['ID'] == n_id]['Time'])
                # spikes_pred = np.array(df_pred_trial[df_pred_trial['ID'] == n_id]['Time'])
                spikes_true = df_true['Time'][df_true['ID'] == n_id]
                spikes_pred = df_pred['Time'][df_pred['ID'] == n_id]
                if len(spikes_pred) > 0:
                    [cos_score, cos_prec, cos_call, y, y_hat, t_y] = compute_score(width, spikes_true, spikes_pred)
                else:
                    continue
                # scores = compute_scores(df_trial_true, df_trial_pred)
                
                precision.append(cos_prec)
                recall.append(cos_call)
                f1.append(cos_score)
            df_true = []
            df_pred = []
    av_precision = np.mean(np.nan_to_num(precision))
    av_recall = np.mean(np.nan_to_num(recall))
    av_f1 = np.mean(np.nan_to_num(f1))
    # precision_score[len(precision_score.keys())].append(np.mean(np.nan_to_num(precision)))
    # recall_score[len(recall_score.keys())].append(np.nan_to_num(np.mean(recall
    return av_precision, av_recall, av_f1


def get_score(precision_score, recall_score, f1_score, model, loader, device):

    model.eval().to(device)
    precision = []
    recall = []
    f1 = []
    for x, y in loader:
        # place data on the correct device
        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        
        preds, features, loss = model(x, y)
        precision.append(preds['precision'])
        recall.append(preds['recall'])
        f1.append(preds['F1'])
    av_precision = np.mean(np.nan_to_num(precision))
    av_recall = np.mean(np.nan_to_num(recall))
    av_f1 = np.mean(np.nan_to_num(f1))
    # precision_score[len(precision_score.keys())].append(np.mean(np.nan_to_num(precision)))
    # recall_score[len(recall_score.keys())].append(np.nan_to_num(np.mean(recall)))
    # f1_score[len(f1_score.keys())].append(np.mean(np.nan_to_num(f1)))

    return av_precision, av_recall, av_f1


