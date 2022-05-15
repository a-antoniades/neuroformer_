from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns

from utils import *
from analysis import *
from SpikeVidUtils import *

def tidy_axis(ax, top=False, right=False, left=False, bottom=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', left='off', direction='out', width=1)

def plot_neurons(ax, df, neurons, color_map):
    # print(df.head())
    for id_ in neurons:
        df_id = df[df['ID'] == id_]
        if len(df_id) == 0:
            break
        color = color_map[id_]
        ax.scatter(df_id['Time'], df_id['ID'], color=color, marker="|", s=150, label='Simulated')
        
        # ax.set_ylim(0, len(neurons))
        xlim = int(max(df['Interval']))
        ax.set_xlim(0, xlim)
        ax.set_xticks(np.linspace(0, xlim, num=3))
        
        ax.tick_params(axis='y', labelsize=15) 
        ax.tick_params(axis='x', labelsize=15)  


def plot_raster_trial(df1, df2, trials, neurons):

    color_labels = neurons
    rgb_values = sns.color_palette("bright", len(neurons))
    color_map = dict(zip(color_labels, rgb_values))

    fig, ax = plt.subplots(nrows=len(trials), ncols=2, figsize=(12,10), squeeze=False)
    for n, trial in enumerate(trials):
        df1_trial_n = df1[df1['Trial'] == trial]
        df2_trial_n = df2[df2['Trial'] == trial]
        ax[n][0].set_ylabel(f'Trial {trial}')
        plot_neurons(ax[n][0], df1_trial_n, neurons, color_map)
        plot_neurons(ax[n][1], df2_trial_n, neurons, color_map)

        # ax[0][n].get_shared_x_axes().join(ax[0][0], ax[0][n])
        # ax[1][n].get_shared_x_axes().join(ax[0][0], ax[1][n])

    plt.setp(ax, yticks=neurons, yticklabels=neurons)
    ax[0][0].set_title('True')
    ax[0][1].set_title('Predicted')
    fig.supxlabel('Time (S)')
    fig.supylabel('Neuron ID')
    plt.tight_layout()


def plot_raster_trial(df1, df2, trials, neurons):

    color_labels = neurons
    rgb_values = sns.color_palette("bright", len(neurons))
    color_map = dict(zip(color_labels, rgb_values))

    fig, ax = plt.subplots(nrows=len(trials), ncols=2, figsize=(12,10), squeeze=False)
    for n, trial in enumerate(trials):
        df1_trial_n = df1[df1['Trial'] == trial]
        df2_trial_n = df2[df2['Trial'] == trial]
        ax[n][0].set_ylabel(f'Trial {trial}')
        plot_neurons(ax[n][0], df1_trial_n, neurons, color_map)
        plot_neurons(ax[n][1], df2_trial_n, neurons, color_map)

        # ax[0][n].get_shared_x_axes().join(ax[0][0], ax[0][n])
        # ax[1][n].get_shared_x_axes().join(ax[0][0], ax[1][n])

    plt.setp(ax, yticks=neurons, yticklabels=neurons)
    ax[0][0].set_title('True')
    ax[0][1].set_title('Predicted')
    fig.supxlabel('Time (S)')
    fig.supylabel('Neuron ID')
    plt.tight_layout()

def get_id_intervals(df, n_id, intervals):
    id_intervals = np.zeros(len(intervals))
    interval_counts = df[df['ID'] == n_id].groupby(df['Interval']).size()
    id_intervals[interval_counts.index.astype(int).tolist()] = interval_counts.index.astype(int).tolist()
    return id_intervals.tolist()


def get_id_intervals(df, n_id, intervals):
    id_intervals = np.zeros(len(intervals))
    interval_counts = df[df['ID'] == n_id].groupby(df['Interval']).size()
    id_intervals[interval_counts.index.astype(int).tolist()] = interval_counts.index.astype(int).tolist()
    return id_intervals.tolist()

def plot_var(ax, df, variable, values, color_map, m_s=150, l_w=1):
    for value in values:
        color = color_map[value]
        data = df[df[variable] == value]
        data[variable] = data[variable].astype('str')
        ax.scatter(data['Time'], data[variable], color=color,    # c=data[variable].map(color_map),
                   marker="|", s=m_s, linewidth=l_w)

        # ax.xaxis.set_tick_params(top='off', direction='out', width=1)
        ax.yaxis.set_tick_params(right='off', left='off', direction='out', width=1)
        
        ax.set_ylim(0, len(values))
        xlim = int(max(df['Interval']))
        ax.set_xlim(0, xlim)
        ax.set_xticks(np.linspace(0, xlim, num=3))

        ax.tick_params(axis='y', labelsize=10) 
        ax.tick_params(axis='x', labelsize=10)       

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.xaxis.set_tick_params(top='off', direction='out', width=1)
        # ax.yaxis.set_tick_params(right='off', direction='out', width=1)


ms_firing = 25
line_width = 0.75
lw_scatter = 0.1
def plot_firing_comparison(df_1, df_2, id, trials, intervals, figure_name=None):
    '''
    get trial averaged spikes (PSTH)
    '''
    id_ = id
    true = df_1[(df_1['Trial'].isin(trials)) & (df_1['ID'] == id_)].reset_index(drop=True)
    pred = df_2[(df_2['Trial'].isin(trials)) & (df_2['ID'] == id_)].reset_index(drop=True)
    rates_1_id = get_rates(true, [id_], intervals)[id_]
    rates_2_id = get_rates(pred, [id_], intervals)[id_]

    left, width = 0.15, 0.85
    bottom, height = 0.1, 0.1
    spacing = 0.005

    height_hist = 0.10
    rect_scatter_1 = [left, bottom*4, width, height]
    rect_scatter_2 = [left, bottom*3, width, height]
    rect_hist1 = [left, bottom*2, width, height_hist]
    # rect_hist2 = [left, bottom*1, width, height_hist]
    # rect_histy = [left + width + spacing, bottom, 0.2, height]

    if figure_name is None:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig = figure_name


    # ax_rast_1 = fig.add_subaxes(rect_scatter_1)
    # ax_rast_2 = fig.add_axes(rect_scatter_2, sharex=ax_rast_1)
    # ax_hist_1 = fig.add_axes(rect_hist1, sharex=ax_rast_1)
    # ax_hist_2 = fig.add_axes(rect_hist2, sharex=ax_rast_1)

    tidy_axis(fig)
    no_top_right_ticks(fig)
    fig.set_yticks([])
    fig.set_yticklabels([])
    fig.axis('off')

    ax_rast_1 = fig.inset_axes(rect_scatter_1)
    ax_rast_2 = fig.inset_axes(rect_scatter_2, sharex=ax_rast_1)
    ax_hist_1 = fig.inset_axes(rect_hist1, sharex=ax_rast_1)

    ax_rast_2.axis('off')
    ax_rast_1.axis('off')

    axes_list = [ax_rast_1, ax_rast_2, ax_hist_1]
    # colors = sns.color_palette("gist_ncar_r", 2)
    colors = ['black', 'red']

    def plot_raster_scatter(ax, data, color, label):
        ax.scatter(data['Interval'], data['ID'], c=color, s=ms_firing, linewidth=lw_scatter, marker='|', label=label)
        ax.set_xlabel(label)

    # ax.scatter(true['Interval'], true['ID'].astype('str'), color='#069AF3', marker='|')
    plot_raster_scatter(ax_rast_2, pred, colors[0], 'Simulated')
    plot_raster_scatter(ax_rast_1, true, colors[1], 'True')

    # sns.distplot(true['Interval'], hist=False)
    # sns.distplot(pred['Interval'], hist=False)
    sns.kdeplot(pred['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[0], lw=line_width, alpha=0.7)    #plot(np.array(intervals), rates_1_id, color=colors[0],  lw=3)
    sns.kdeplot(true['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[1], lw=line_width, alpha=0.7)   #plot(np.array(intervals), rates_2_id, color=colors[1], lw=3)
    
    ax_hist_1.set_ylabel('')
    ax_hist_1.set_yticks([])
    sns.despine(top=True, left=True)
    # tidy_axis(ax_hist_1, bottom=True)
    # tidy_axis(ax_hist_2, bottom=True)
    ax_hist_1.set_xlabel([])
    # ax_hist_1.spines['bottom'].set_visible(False)
    # ax_rast_1.spines['bottom'].set_visible(False)
    # ax_rast_2.spines['bottom'].set_visible(False)
    # ax_hist_1.spines['top'].set_visible(False)
    # ax_hist_2.spines['top'].set_visible(False)

    # xlabels = np.arange(0, max(intervals) + 1, 60)
    # xticks, xlabels = xlabels, xlabels
    max_intervals = math.ceil(df_1['Interval'].max())
        # max_intervals = max(intervals)
    xticks, xlabels = [0,max_intervals // 2, max_intervals], [0,max_intervals // 2, max_intervals]

    yticks, ylabels = np.arange(len(trials)), list(map(str, trials))
    for ax in axes_list:
        tidy_axis(ax, bottom=True)
        no_top_right_ticks(ax)
        ax.set_xlim(0, max(intervals))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks([])
        ax.set_yticklabels([])


    # ax_hist_1.set_xlabel('Time (s)', fontsize=20)
    ax_hist_1.set_xlabel('', fontsize=20)
    legend = fig.legend(bbox_to_anchor=(0.25, 0.01), ncol=3, frameon=True, fontsize=17.5)  # bbox_to_anchor=(0.75, 0.55)
    ax_rast_1.set_title("{}".format(id_), fontsize=20)



def plot_firing_comparison_sweeps(df_1, df_2, id, trials, intervals, figure_name=None):
    '''
    get trial averaged spikes (PSTH)
    '''

    left, width = 0.15, 0.85
    bottom, height = 0.1, 0.1
    spacing = 0.005

    height_hist = 0.10
    rect_hist1 = [left, bottom*2, width, height_hist]
    # rect_hist2 = [left, bottom*1, width, height_hist]
    # rect_histy = [left + width + spacing, bottom, 0.2, height]

    if figure_name is None:
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        fig = plt.subplot()
    else:
        fig = figure_name

    tidy_axis(fig)
    no_top_right_ticks(fig)
    fig.set_yticks([])
    fig.set_yticklabels([])
    fig.axis('off')

    ax_dict_true = dict()
    ax_dict_pred = dict()
    for n, trial in enumerate(trials):
        ax_dict_true[trial] = fig.inset_axes([left, bottom * (3+n), width, height_hist])
        ax_dict_pred[trial] = fig.inset_axes([left, bottom * (3+n+len(trials)), width, height_hist], sharex=ax_dict_true[trial])

        ax_dict_true[trial].axis('off')
        ax_dict_pred[trial].axis('off')
        
    ax_hist_1 = fig.inset_axes(rect_hist1, sharex=ax_dict_true[trials[0]])
    axes_list = [list(ax_dict_true.values()), list(ax_dict_pred.values()), [ax_hist_1]]

    # colors = sns.color_palette("gist_ncar_r", 2)
    colors = ['black', 'red']

    def plot_raster_scatter(ax, data, color, label):
        ax.scatter(data['Interval'], data['ID'], c=color, s=ms_firing, marker='|', linewidth=lw_scatter, label=label)
        ax.set_xlabel(label)

    # ax.scatter(true['Interval'], true['ID'].astype('str'), color='#069AF3', marker='|')
    for n, trial in enumerate(trials):
        id_ = id
        true = df_1[(df_1['Trial'] == trial) & (df_1['ID'] == id_)].reset_index(drop=True)
        pred = df_2[(df_2['Trial'] == trial) & (df_2['ID'] == id_)].reset_index(drop=True)
        if id_ == 345:
            print(true, pred)
        plot_raster_scatter(ax_dict_pred[trial], pred, colors[0], 'Simulated')
        plot_raster_scatter(ax_dict_true[trial], true, colors[1], 'True')

        sns.kdeplot(pred['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[0], lw=line_width, alpha=0.7)    #plot(np.array(intervals), rates_1_id, color=colors[0],  lw=3)
        sns.kdeplot(true['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[1], lw=line_width, alpha=0.7)   #plot(np.array(intervals), rates_2_id, color=colors[1], lw=3)
    
    
    max_intervals = df_1['Interval'].max()
    yticks, ylabels = np.arange(len(trials)), list(map(str, trials))
    xticks, xlabels = [0,max_intervals // 2, max_intervals], [0,max_intervals // 2, max_intervals]
    for ax in axes_list:
        ax = ax[0]
        tidy_axis(ax, bottom=True)
        no_top_right_ticks(ax)
        ax.set_xlim(0, max(intervals))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks([])
        ax.set_yticklabels([])
    # ax_hist_1.set_xlim(0, max(intervals))


    # sns.distplot(true['Interval'], hist=False)
    # sns.distplot(pred['Interval'], hist=False)
    
    ax_hist_1.set_ylabel('')
    ax_hist_1.set_yticks([])
    sns.despine(top=True, left=True)
    # tidy_axis(ax_hist_1, bottom=True)
    # tidy_axis(ax_hist_2, bottom=True)
    ax_hist_1.set_xlabel('')
    # ax_hist_1.set_xlabel('Time (s)', fontsize=20)
    legend = fig.legend(bbox_to_anchor=(0.25, 0.01), ncol=3, frameon=True, fontsize=17.5)  # bbox_to_anchor=(0.75, 0.55)
    list(ax_dict_pred.values())[-1].set_title("{}".format(id_), fontsize=20)
    

def get_psth(df, n_id, trials):
    df = df[df['ID'] == n_id]
    df = df[df['Trial'] == trial]
    df = df.groupby('Interval_dt').size().reset_index()
    df.columns = ['Interval_dt', 'Count']
    return df

def set_categorical_ticks(ax, yticks=None, ylabels=None, xticks=None, xlabels=None, fs=None):
    fs = fs if fs is not None else 10
    if yticks is not None:
        ax.set_ylim(0, len(ylabels))
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
    if xticks is not None:
        ax.set_xlim(0, max(xlabels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=fs)
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
    ax.get_yaxis().tick_left()

def no_top_right_ticks(ax):
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(top='off', right='off', left='on', direction='out', width=1)
    ax.tick_params(labelright='off', labeltop='off')
    ax.tick_params(axis='both', direction='out')
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
    ax.get_yaxis().tick_left()

def plot_neurons_trials_psth(df_1, df_2, neurons, trials, intervals, figuresize=None): 

    fs = 15
    plt.rcParams['xtick.labelsize']= fs
    plt.rcParams['ytick.labelsize']= fs
    plt.rcParams['axes.labelsize']= fs
    plt.rcParams['axes.titlesize']= fs
    plt.rcParams['legend.fontsize']= fs
    plt.rcParams['lines.linewidth']= 2
    # plt.rcParams['fig.supylabel']= fs
    
    df_1 = df_1.reset_index(drop=True)
    df_2 = df_2.reset_index(drop=True)
    dt = 4
    intervals_dt = [dt * n for n in range(int((intervals[-1]) // dt) + 1)]
    df_1['Interval_dt'] = pd.cut(df_1['Interval'], intervals_dt, include_lowest=True)
    df_2['Interval_dt'] = pd.cut(df_2['Interval'], intervals_dt, include_lowest=True)

    # neuron_list = list(map(str, sorted(top_corr[:6].index.tolist())))
    neurons = list(map(str, [i for i in neurons]))

    trials = df_1['Trial'].unique()
    # neuron_list = sorted(top_corr[:10].index.tolist())
    scale = 1
    nrows, ncols = 4, len(neurons)
    fig_size = figuresize if figuresize is not None else (2 * scale * len(neurons),10 * scale)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    variable = 'Trial'

    color_labels = trials
    rgb_values = sns.color_palette("gist_ncar_r", len(trials))
    color_map = dict(zip(color_labels, rgb_values))

    max_freq = 0
    for n, neuron in enumerate(neurons):
        df_1['ID'] = df_1['ID'].astype('str')
        df_2['ID'] = df_2['ID'].astype('str') 
        df_1_id = df_1[df_1['ID'] == neuron]
        df_2_id = df_2[df_2['ID'] == neuron] 

        max_intervals = 32
        # max_intervals = max(intervals)
        yticks, ylabels = np.arange(len(trials)), list(map(str, trials))
        xticks, xlabels = [0,max_intervals // 2, max_intervals], [0,max_intervals // 2, max_intervals]

        m_s = 45
        l_w = 0.5
        plot_var(ax[0][n], df_1_id, variable, trials, color_map, m_s, l_w=l_w)
        plot_var(ax[1][n], df_2_id, variable, trials, color_map, m_s, l_w=l_w)

        set_categorical_ticks(ax[0][n], yticks, ylabels, xticks, xlabels)
        set_categorical_ticks(ax[1][n], yticks, ylabels, xticks, xlabels)

        ax[0][n].set_yticks([])
        ax[1][n].set_yticks([])

        if n > 0:
            no_top_right_ticks(ax[0][n])
            no_top_right_ticks(ax[1][n])

        df_1['ID'] = df_1['ID'].astype('int')
        df_2['ID'] = df_2['ID'].astype('int')  

        neuron_int = int(neuron)
        df_1_id = df_1[df_1['ID'] == neuron_int]
        df_2_id = df_2[df_2['ID'] == neuron_int]
        # rates_1 = get_rates(df_1, [neuron_int], intervals_dt)[neuron_int]
        # rates_2 = get_rates(df_2, [neuron_int], intervals_dt)[neuron_int]

        freq_id_1 = df_1_id['Interval'].value_counts().reindex(intervals, fill_value=0)
        freq_id_2 = df_2_id['Interval'].value_counts().reindex(intervals, fill_value=0)
        bins = np.arange(len(intervals) // 2)
        # bins = len(intervals)
        # ax[2][n].bar(intervals_dt, freq_id_1)

        # ax[2][n].hist([freq_id_1, freq_id_2], bins=bins, histtype='step', edgecolor=['blue', 'red'], 
        #                lw=2, alpha=0.3, facecolor=['blue', 'red'], label=['True', 'Sim'])
        c_2, c_1 = rgb_values[2], rgb_values[-1]
        ax[2][n].hist(df_1_id['Interval'], bins=bins, edgecolor=None, lw=2, alpha=1, facecolor=c_1, label='True')
        ax[3][n].hist(df_2_id['Interval'], bins=bins, edgecolor=None, lw=2, alpha=1, facecolor=c_2, label='Predicted') # histtype='step'
        
        # xticks, xlabels = [0, max(intervals) // 2, max(intervals)], [0, max(intervals) // 2, max(intervals)]
        y_fs_hist = 15
        set_categorical_ticks(ax[2][n], None, None, xticks, xlabels, y_fs_hist)
        ax[2][n].spines['right'].set_visible(False)
        ax[2][n].spines['top'].set_visible(False)

        set_categorical_ticks(ax[3][n], None, None, xticks, xlabels, y_fs_hist)
        ax[3][n].spines['right'].set_visible(False)
        ax[3][n].spines['top'].set_visible(False)

        if n > 0:
            no_top_right_ticks(ax[2][n])
            ax[3][n].get_shared_y_axes().join(ax[2][n], ax[2][n-1])
            no_top_right_ticks(ax[3][n])
        max_lim = (max(ax[2][n].get_ylim()[1], ax[3][n].get_ylim()[1]))
        ax[0][n].set_xticklabels([])
        ax[1][n].set_xticklabels([])
        ax[2][n].set_xticklabels([])
        ax[2][n].set_ylim(0, max_lim)
        ax[3][n].set_ylim(0, max_lim)
        ax[2][n].get_shared_y_axes().join(ax[3][n], ax[3][n-1])

        
        # max_freq = max(freq_id_1.max(), freq_id_2.max(), max_freq)
        # yticks, ylabels = np.linspace(0, max(freq_id_1.max(), freq_id_2.max()), 3), [i for i in range(max(freq_id_1.max(), freq_id_2.max()))]
        # set_categorical_ticks(ax[2][n], yticks, ylabels, xticks, xlabels)

    plt.setp(ax[0])

    # ax[0][0].set_ylim(0, 32)
    ax[0][0].set_ylabel('Ground Truth')
    ax[1][0].set_ylabel('Simulated')
    # ax[2][0].set_ylabel('PSTH, True')
    # ax[3][0].set_ylabel('PSTH, Simulation')
    # ax[2][-1].legend()


    ax[0][0].legend(bbox_to_anchor=(0,0,1,1))
    # fig.supxlabel('Time (S)', fontsize=15, y=0.07)
    # fig.supylabel('Trials')
    fig.suptitle('Gabor 3D Sim', fontsize=20, y=0.925)
    # fig.gca().set_aspect('equal', adjustable='box')
    # plt.autoscale()
    # plt.tight_layout()
    


def get_boxplot_data(df_1, df_2, intervals, trials):
    data_boxplot_true = []
    data_boxplot_pred = []
    for n, trial in enumerate(trials):
        trial_prev = trials[n - 1] if n > 0 else trials[n + 1]
        true_prev = df_1[df_1['Trial'] == trial_prev].reset_index(drop=True)
        true = df_1[df_1['Trial'] == trial].reset_index(drop=True)
        pred = df_2[df_2['Trial'] == trial].reset_index(drop=True)
        rates_true_prev, rates_true, rates_pred = get_rates_trial(true_prev, intervals), get_rates_trial(true, intervals), get_rates_trial(pred, intervals)
        
        corr_trials_true = calc_corr_psth(rates_true, rates_true_prev)
        corr_trials_pred = calc_corr_psth(rates_true, rates_pred)

        data_boxplot_true.append(np.array(corr_trials_true).flatten())
        data_boxplot_pred.append(np.array(corr_trials_pred).flatten())

    return data_boxplot_true, data_boxplot_pred, corr_trials_true, corr_trials_pred

def plot_error_bar(x, n, color):
    """
    databoxplot_true, databoxplot_pred, corr_trials_true, corr_trials_pred = get_boxplot_data(df_1, df_2, intervals, n_trial)
    
    plot_error_bar(corr_trials_true, n, true_color)
    plot_error_bar(corr_trials_pred, n, pred_color)
    """
    mins = x.min()
    maxes = x.max()
    means = x.mean()
    std = x.std()

    # plt.errorbar(n, means, std, fmt='ok', lw=3)
    # plt.errorbar(n, means, [means - mins, maxes - means],
    #             fmt='.k', ecolor='gray', lw=1)
    # plt.xlim(-1, 8)
    
    green_diamond = dict(markerfacecolor=color, marker='o')
    # fig3, ax3 = plt.subplots()
    # ax3.set_title('Changed Outlier Symbols')
    ax.boxplot(x, flierprops=green_diamond)


def fancy_boxplot(fig, ax1, data, color):
    
    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                alpha=0.5)

    # ax1.set(
    #     axisbelow=True,  # Hide the grid behind plot objects
    #     title='Comparison of IID Bootstrap Resampling Across Five Distributions',
    #     xlabel='Distribution',
    #     ylabel='Value',
    # )

    # Now fill the boxes with desired colors
    # box_colors = ['darkkhaki', 'royalblue']
    # box_colors = sns.dark_palette("#69d", len(data), reverse=True)
    box_colors = [color]
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[0]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    # ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 40
    # bottom = -5
    # ax1.set_ylim(bottom, top)
    # ax1.set_xticklabels(np.repeat(random_dists, 2),
    #                     rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                transform=ax1.get_xaxis_transform(),
                horizontalalignment='center', size='x-small',
                weight=weights[k], color=box_colors[0])
    fig.supxlabel('Trials')
    fig.supylabel('Pearson Correlation (P)')
    fig.suptitle('Inter-Neuron Correlation Across Trials')
    plt.tight_layout()


def plot_intertrial_corr(corr_true, corr_pred, trial):
    
    def scatter_hist(x, y, ax, ax_histy):
        # no labels
        # ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        # ax.scatter(x, y)
        # bins = 250

        # now determine nice limits by hand:
        # binwidth = 0.25
        # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        # lim = (int(xymax/binwidth) + 1) * binwidth

        # bins = np.arange(-lim, lim + binwidth, binwidth)
        # ax_histx.hist(x, bins=bins)
        ax_hist = sns.distplot(y,  hist=False, ax=ax_histy, vertical=True) # (x, y, bins=10, orientation='horizontal')
        ax_hist.set(xlabel=None)

    # sns.distplot(top_corr, hist=False, ax=ax_histy, vertical=True)
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005


    rect_scatter = [left, bottom, width, height]
    # rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_axes(rect_scatter)
    # ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(np.array(corr_true.index), corr_true, ax, ax_histy)
    scatter_hist(np.array(corr_pred.index), corr_pred, ax, ax_histy)
    ax.grid(lw=0.8, alpha=0.7, color='gray')
    ax.scatter(corr_true.index, corr_true, label=f'Trial {trial} vs. 1', alpha=0.4)
    ax.scatter(corr_pred.index, corr_pred, label=f'Trial {trial} vs. Pred', alpha=0.5)
    ax.set_title('Pair-wise Correlation Between Trials', fontsize=25)
    ax.set_xlabel('Neuron ID', fontsize=20)
    ax.set_ylim(-0.1, 0.6)
    plt.ylabel('Pearson Correlation (p)')
    ax.legend(fontsize=20, title_fontsize=20)
    plt.show()