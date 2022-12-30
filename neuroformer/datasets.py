import math
import torch
from scipy import io as scipyio
from scipy.special import softmax
import skimage
from scipy.ndimage.filters import gaussian_filter, uniform_filter

import sys
sys.path.append('/local/home/antonis/neuroformer/neuroformer')
from SpikeVidUtils import image_dataset, r3d_18_dataset, make_intervals
from PIL import Image
from analysis import *
from utils import *
from SpikeVidUtils import create_full_trial


def load_V1_AL(stimulus_path=None, response_path=None, top_p_ids=None):
    if stimulus_path is None:
        # stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
        stimulus_path = "data/Combo3_V1AL/Combo3_V1AL_stimulus.pt"
    if response_path is None:
        # response_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/neural/NatureMoviePart1-A/20-NatureMovie_part1-A_spikes(1).mat"
        response_path = "data/Combo3_V1AL/Combo3_V1AL_response.csv"
    
    df = pd.read_csv(response_path)
    video_stack = torch.load(stimulus_path)

    # video_stack = [skimage.io.imread(os.path.join(stimulus_path, vid)) for vid in stim_names]
    # print(glob.glob(train_path + '/*.tif'))
    video_stack = video_stack.transpose(1, 2)

    # spike_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/neural/NatureMoviePart1-A" # "code/data/SImIm/simNeu_3D_WithNorm__Combo3.mat" 
    n_V1_AL = (351, 514)


    # df = None
    # filenames = ['Combo3_V1.mat', 'Combo3_AL.mat']
    # # filenames = ['Combo3_V1_shuffle.mat', 'Combo3_AL_shuffle.mat']

    # files = []
    # for filename in filenames: 
    #     spike_data = scipyio.loadmat(os.path.join(response_path, filename))
    #     spike_data = np.squeeze(spike_data['spiketrain'].T, axis=-1)
    #     # spike_data = np.squeeze(spike_data['spiketrain_shuffle'].T, axis=-1)
    #     spike_data = [trial_df_combo3(spike_data, n_stim) for n_stim in range(3)]
    #     spike_data = pd.concat(spike_data, axis=0)

    #     spike_data['Trial'] = spike_data['Trial'] + 1
    #     spike_data['Time'] = spike_data['Time'] * 0.0751
    #     spike_data = spike_data[(spike_data['Time'] > 0) & (spike_data['Time'] <= 32)]

    #     if df is None:
    #         df = spike_data.reset_index(drop=True)
    #     else:
    #         spike_data['ID'] += df['ID'].max() + 1
    #         df = pd.concat([df, spike_data], axis=0)

    # vid_duration = [len(vid) * 1/20 for vid in vid_list]

    df = df.sort_values(['Trial', 'Time']).reset_index(drop=True)
    if top_p_ids is not None:
        if isinstance(top_p_ids, int) or isinstance(top_p_ids, float):
            top_p_ids = df.groupby('ID').count().sort_values(by='Trial', ascending=False)[:int(top_p_ids * len(df['ID'].unique()))].index.tolist()
        df = df[df['ID'].isin(top_p_ids)]

    test_trials = []
    n_trial = [2, 8, 14, 19]
    for n_stim in range(df['Trial'].max() // 20):
        # n_trial = [2, 4, 6, 8, 10, 12, 14, 18]
        for n_t in n_trial:
            trial = (n_stim + 1) * 20 - (n_t)
            test_trials.append(trial)

    return video_stack, df, test_trials


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

from model_neuroformer import GPT, GPTConfig

def instantiate_dataset(config, trials=None):
    stimulus_path = config.data.stimulus_path 
    response_path = config.data.response_path 
    top_p_ids = config.data.top_p_ids
    print(config.data.dataset)
    if config.data.dataset == 'V1_AL':
        video_stack, df, test_trials = load_V1_AL(stimulus_path, response_path, top_p_ids)
    elif config.data.dataset == 'natmovieGabor3D':
        video_stack, df, test_trials = load_natural_movie(stimulus_path, response_path, top_p_ids)

    test_trials = test_trials if trials is None else trials
    block_size = config.data.id_block_size
    id_block_size = config.data.id_block_size
    kernel_size = config.model.kernel_size
    n_embd_frames = config.model.n_embd_frames
    frame_block_size = ((20 // kernel_size[0] * 64 * 112) // (n_embd_frames))
    prev_id_block_size = config.data.prev_id_block_size
    window = config.data.window
    window_prev = config.data.window_prev
    frame_window = config.data.frame_window
    dt = config.data.dt
    frame_memory = None
    neurons = sorted(list(set(df['ID'].unique())))
    trial_tokens = [f"Trial {n}" for n in df['Trial'].unique()]
    feat_encodings = neurons + ['SOS'] + ['EOS'] + ['PAD']  # + pixels 
    stoi = { ch:i for i,ch in enumerate(feat_encodings) }
    itos = { i:ch for i,ch in enumerate(feat_encodings) }
    max_window = max(window, window_prev)
    dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
    n_dt = [round(dt * n, 2) for n in range(dt_range)]
    stoi_dt = { ch:i for i,ch in enumerate(n_dt) }
    itos_dt = { i:ch for i,ch in enumerate(n_dt) }
    frame_feats = video_stack
    data_dict =  pickle.load(open(config.data.data_dict, "rb")) if config.data.data_dict is not None else None
    
    df['Interval'] = make_intervals(df, window)
    train_data = df[df['Trial'].isin(test_trials) == False]
    test_data = df[df['Trial'].isin(test_trials)]


    train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, 
                                      frame_block_size, prev_id_block_size, window, 
                                      dt, frame_memory, stoi, itos, neurons, stoi_dt, 
                                      itos_dt, frame_feats, pred=False, window_prev=window_prev, 
                                      frame_window=frame_window, data_dict=data_dict)
    
    test_dataset = SpikeTimeVidData2(test_data, None, block_size, id_block_size, 
                                      frame_block_size, prev_id_block_size, window, 
                                      dt, frame_memory, stoi, itos, neurons, stoi_dt, 
                                      itos_dt, frame_feats, pred=False, window_prev=window_prev, 
                                      frame_window=frame_window, data_dict=data_dict)

    
    # # instantiate model
    # config.model.n_dt = n_dt
    # mconf = GPTConfig(config.model)
    # model = 
    # print(mconf)



    return df, video_stack, train_dataset, test_dataset


from model_neuroformer import GPT, GPTConfig
def instantiate_model(config, dataset):
    window, window_prev = config.data.window, config.data.window_prev
    dt = config.data.dt
    max_window = max(window, window_prev)
    dt_range = math.ceil(max_window / dt) + 1  # add first / last interval for SOS / EOS'
    n_dt = len([round(dt * n, 2) for n in range(dt_range)])
    config.model['dt'] = dt
    config.model['n_dt'] = n_dt
    config.model['block_size'] = config.data.id_block_size
    config.model['id_block_size'] = config.data.id_block_size
    config.model['prev_id_block_size'] = config.data.prev_id_block_size
    config.model['frame_block_size'] = config.data.frame_block_size
    print(config.data.id_block_size, dataset.id_population_size, config.model)
    mconf = GPTConfig(dataset.id_population_size,
                      config.data.id_block_size,
                      id_vocab_size=dataset.id_population_size,
                      config=config.model)

    model = GPT(mconf)
    print(config.trainer.ckpt_path)
    return model


def instantiate_trainer(config, model, train_dataset, test_dataset=None):
    tconf = config.trainer
    mconf = config.model
    print(type(tconf))
    # if tconf.ckpt_path is not None:
    #     model.load_state_dict(torch.load(tconf.ckpt_path, map_location='cpu'), 
    #                                     strict=True)
    
    from trainer import Trainer, TrainerConfig
    # model_path = f"/models/tensorboard/V1_AL/w:{window}_wp:{window_prev}/{6}_Cont:{mconf.contrastive}_window:{window}_f_window:{frame_window}_df:{dt}_blocksize:{id_block_size}_sparse{mconf.sparse_mask}_conv_{conv_layer}_shuffle:{shuffle}_batch:{batch_size}_sparse_({mconf.sparse_topk_frame}_{mconf.sparse_topk_id})_blocksz{block_size}_pos_emb:{mconf.pos_emb}_temp_emb:{mconf.temp_emb}_drop:{mconf.id_drop}_dt:{shuffle}_2.0_{max(n_dt)}_max{dt}_{mconf.layers}_{mconf.n_head}_{mconf.n_embd}_nembframe{mconf.n_embd_frames}_{mconf.kernel_size}.pt"
    tconf = TrainerConfig(config=tconf)
    trainer = Trainer(model, train_dataset, test_dataset, tconf, mconf)
    return model, trainer

