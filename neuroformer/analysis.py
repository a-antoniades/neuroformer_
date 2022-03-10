import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.special import kl_div


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
    df = df.groupby(['ID', interval]).count().unstack(fill_value=0).stack()['Time']
    def set_rates(df, id, intervals):
        rates = np.zeros_like(intervals, dtype=np.float32)
        if id not in df.index:
            return rates
        else:
            df = df[id]
            for i in df.index:
                n = int((i * 2) - 1)
                rates[n] = df[i]            
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
    df_rates = df.groupby(['ID', 'Interval']).count().unstack(fill_value=0).stack()
    def set_rates(df, id, intervals):
        df = df.loc[id]
        rates = np.zeros_like(intervals, dtype=np.float32)
        for i in df.index:
            n = int((i * 2) - 2)
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
