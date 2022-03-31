import numpy as np
from sympy import Q
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import math
from scipy.special import softmax
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from sklearn.preprocessing import normalize
from matplotlib.colors import LinearSegmentedColormap
from numpy import linalg as LA
from SpikeVidUtils import get_frame_idx
from utils import top_k_top_p_filtering

from scipy import signal

def convolve_atts_3D(stim_atts):
    '''
    input: (ID, T, Y, X)
    '''
    sigma = 2.0     # width of kernel
    x = np.arange(-3,4,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))

    for n_id in range(stim_atts.shape[0]):
        stim_atts[n_id] = signal.convolve(stim_atts[n_id], kernel, mode="same")
    return stim_atts


def grad_rollout(attentions, gradients, discard_ratio=0.8):
        result = None
        with torch.no_grad():
                for attention, grad in zip(attentions[-1], gradients):     
                        weights = grad
                        attention_heads_fused = (weights*attention).mean(axis=1)
                        attention_heads_fused[attention_heads_fused < 0] = 0

                        # Drop the lowest attentions, but
                        # don't drop the class token
                        # flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                        # _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                        # #indices = indices[indices != 0]
                        # flat[0, indices] = 0

                        I = torch.eye(attention_heads_fused.size(-2), attention_heads_fused.size(-1))
                        a = (attention_heads_fused + 1.0*I)/2
                        # a = attention_heads_fused
                        a = a / a.sum(dim=-1).unsqueeze(-1)
                        # a = a[:, pos_index]
                        if result == None:
                                result = a
                        else:
                                result = a * result
        
        # print(result.shape)
        # # Look at the total attention between the class token,
        # # and the image patches
        # mask = result[0, 0 ,pos_index]
        # # In case of 224x224 image, this brings us from 196 to 14
        # width = int(mask.size(-1)**0.5)
        # mask = mask.reshape(width, width).numpy()
        # mask = mask / np.max(mask)
        return torch.nan_to_num(result)

def grad_att(attentions, gradients, discard_ratio=0.8):
        with torch.no_grad():
                # atts = attentions * gradients
                # return atts
                return attentions

class VITAttentionGradRollout:
        """
        This class is an adaptation of Jacob Gildenblat's implementation: 
        https://github.com/jacobgil/vit-explain

        We calculate Attention Rollou (Abnar, Zuidema, 2020), 
        for stimuluts-state attention, and condition
        it on the gradient of a specific target neuron.

        This way we can get neuron-specific attentions.
        """

        def __init__(self, model, module, attn_layer_name='attn_drop', discard_ratio=0.5):
                self.model = model
                self.module = module
                self.discard_ratio = discard_ratio
                for name, module in self.module.named_modules():
                        if attn_layer_name in name:
                                module.register_forward_hook(self.get_attention)
                                module.register_full_backward_hook(self.get_attention_gradient)
                
                self.attentions = []
                self.attention_gradients = []

        def get_attention(self, module, input, output):
                self.attentions.append(output.cpu())
                # print(output.shape)
                # print(len(self.attentions))
                
        
        def get_attention_gradient(self, module, grad_input, grad_output):
                # print(grad_input[0].shape)
                # print(grad_input)
                self.attention_gradients.append(grad_input[0].cpu())
                # print(grad_input[0].shape)

        def __call__(self, x, y):
                self.model.zero_grad()
                preds, features, loss = self.model(x, y)
                output = preds['id']
                category_mask = torch.zeros(output.size())
                # category_mask[:, category_index] = 1
                loss = loss['id'] 
                loss.backward()
                
                # print(len(self.attention_gradients))
                return grad_rollout(self.attentions, self.attention_gradients, self.discard_ratio)
                # return grad_att(torch.cat(self.attentions), torch.cat(self.attention_gradients))  # grad_rollout(self.attentions, self.attention_gradients, self.discard_ratio)


# def grad_rollout(attentions, gradients, discard_ratio):
#     result = torch.eye(attentions[0].size(-1))
# #     print(len(gradients))
# #     print(len(attentions))
#     with torch.no_grad():
#         for attention, grad in zip(attentions, gradients):                
#             weights = grad
#             attention_heads_fused = (attention*weights).mean(axis=1)
#             attention_heads_fused[attention_heads_fused < 0] = 0

#             # Drop the lowest attentions, but
#             # don't drop the class token
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             #indices = indices[indices != 0]
#             flat[0, indices] = 0

#             I = torch.eye(attention_heads_fused.size(-1))
#             a = (attention_heads_fused + 1.0*I)/2
#             a = a / a.sum(dim=-1)
#             result = torch.matmul(a, result)
    
#     # Look at the total attention between the class token,
#     # and the image patches
#     mask = result[0, 0 , 1 :]
#     # In case of 224x224 image, this brings us from 196 to 14
#     width = int(mask.size(-1)**0.5)
#     mask = mask.reshape(width, width).numpy()
#     mask = mask / np.max(mask)
#     return mask    

# class VITAttentionGradRollout:
#     def __init__(self, model, blocks, attention_layer_name='attn_drop',
#         discard_ratio=0.9):
#         self.model = model
#         self.discard_ratio = discard_ratio
#         self.blocks = blocks
#         # for name, module in self.model.named_modules():
#         #     if attention_layer_name in name:
#         #         module.register_forward_hook(self.get_attention)
#         #         module.register_full_backward_hook(self.get_attention_gradient)
        
#         for i in range(len(blocks)):
#                 self.blocks[i].attn.attn_drop.register_forward_hook(self.get_attention)
#                 self.blocks[i].attn.attn_drop.register_full_backward_hook(self.get_attention_gradient)
#                 # print(name, module)

#         self.attentions = []
#         self.attention_gradients = []

#     def get_attention(self, module, input, output):
#         self.attentions.append(output.cpu())

#     def get_attention_gradient(self, module, grad_input, grad_output):
#         self.attention_gradients.append(grad_input[0].cpu())

#     def __call__(self, x, y):
#         self.model.zero_grad()
#         preds, features, loss = self.model(x, y)
#         output = preds['id']
#         category_mask = torch.zeros(output.size())
#         # category_mask[:, category_index] = 1
#         loss = loss['id']
#         loss.backward()

#         return grad_rollout(self.attentions, self.attention_gradients,
#             self.discard_ratio)

class AttentionVis:
        '''attention Visualizer'''
        
        # def getAttention(self, spikes, n_Blocks):
        #         spikes = spikes.unsqueeze(0)
        #         b, t = spikes.size()
        #         token_embeddings = self.model.tok_emb(spikes)
        #         position_embeddings = self.model.pos_emb(spikes)
        #         # position_embeddings = self.model.pos_emb(spikes)
        #         x = token_embeddings + position_embeddings

        #         # aggregate attention from n_Blocks
        #         atts = None
        #         for n in n_Blocks:
        #                 attBlock = self.model.blocks[n].attn
        #                 attBlock(x).detach().numpy()    # forward model
        #                 att = attBlock.att.detach().numpy()
        #                 att = att[:, 1, :, :,].squeeze(0)
        #                 atts = att if atts is None else np.add(atts, att)
                
        #         # normalize
        #         atts = atts/len(n_Blocks)
        #         return atts
        
        def visAttention(att):
                plt.matshow(att)
                att_range = att.max()
                cb = plt.colorbar()
                cb.ax.tick_params()
                plt.show()
        
        
        # # this is for gpt style models
        # @torch.no_grad()
        # def getAttention(x, model, blocks=None):
        #         idx = x['id']
        #         dtx = x['dt']
        #         frames = x['frames']
        #         pad = x['pad']
                
        #         features, pad = model.process_features(x)
        #         x = torch.cat((features['frames'], features['id']), dim=1)

        #         # aggregate attention from n_Blocks
        #         atts = None
        #         n_blocks = model.config.n_layer
        #         blocks = range(n_blocks) if blocks is None else blocks
        #         for n in range(n_blocks):
        #                 attBlock = model.blocks[n].attn
        #                 attBlock(x, pad).detach().to('cpu').numpy()    # forward model
        #                 att = attBlock.att.detach().to('cpu')
        #                 att = F.softmax(att, dim=-1).numpy()
        #                 att = att[:, 1, :, :,].squeeze(0)
        #                 atts = att if atts is None else np.add(atts, att)
                
        #         # # normalize
        #         # atts = atts / n_blocks
        #         return atts
        
        
        # this is for neuroformer model
        @torch.no_grad()
        def get_attention(module, n_blocks, block_size, pad=0, rollout=False):
                # aggregate attention from n_Blocks
                atts = None
                T = block_size
                # TODO: get index of 166, get attentions up until that stage
                for n in range(n_blocks):
                        att = module[n].attn.att
                        # n_heads = att.size()[1]
                        if pad != 0:
                                att = att[:, :, T - pad, :,]
                        att = att.detach().squeeze(0).to('cpu').numpy()
                        atts = att[None, ...] if atts is None else np.concatenate((atts, att[None, ...]))
                return atts
                

        # @torch.no_grad()
        # def att_models(models, dataset, neurons):
        #         ''' 
        #         Input list of models
        #         Returns Attentions over dataset
        #         '''
        #         models_atts = []
        #         for model in models:
        #                 attention_scores = np.zeros(len(neurons))
        #                 data = dataset
        #                 pbar = tqdm(enumerate(data), total=len(data))
        #                 for it, (x, y) in pbar:
        #                         # scores = np.array(np.zeros(len(neurons)))
        #                         att = np.zeros(len(neurons))
        #                         score = AttentionVis.getAttention(x, model)
        #                         if score.size >= 1: score = score[-1]
        #                         # scores.append(score)
        #                         for idx, neuron in enumerate(x[:, 0]):
        #                                 """ 
        #                                 for each neuron in scores,
        #                                 add its score to the array
        #                                 """
        #                                 neuron = int(neuron.item())
        #                                 att[neuron] += score[idx]
        #                         attention_scores = np.vstack((attention_scores, att))
        #                         if it > len(dataset):
        #                                 models_atts.append(attention_scores.sum(axis=0))
        #                                 break
        #         return models_atts

        def rollout_attentions(self, att):
                ''' Rollout attentions
                Input: (L, H, ID, F)
                '''
                rollout_att = np.eye(att.shape[-2], att.shape[-1])
                for i in range(att.shape[0]):
                        if i==0:
                                continue
                        I = np.eye(att.shape[-2], att.shape[-1])
                        a = att[i]
                        a = a.max(axis=0)[0]
                        a = (a + 1.0*I) / 2
                        a = a / a.sum(axis=-1, keepdims=True)
                        rollout_att = a * rollout_att
                return rollout_att
        
        # @torch.no_grad()
        def att_interval_frames(self, model, module, loader, n_blocks, block_size, rollout=False, pad_key=None, agg=False, stoi=None, max_it=None):
                device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
                device = 'cpu'
                model.to(device)
                mconf = model.config
                model = model.eval()
                T = block_size
                attention_scores = None
                len_loader = len(loader) if max_it is None else max_it
                pbar = tqdm(enumerate(loader), total=len_loader)

                if rollout:
                        grad_rollout = VITAttentionGradRollout(model, module)
                for it, (x, y) in pbar:
                        pad = x[pad_key] if pad_key is not None else 0
                        # place data on the correct device
                        for key, value in x.items():
                                x[key] = x[key].to(device)
                        for key, value in y.items():
                                y[key] = y[key].to(device)
                        # att = np.swapaxes(att, -1, -2)
                        if rollout:
                                # preds, features, loss, = model(x, y)
                                # att = AttentionVis.get_attention(module, n_blocks, T)
                                # att = self.rollout_attentions(att)
                                att = grad_rollout(x, y)[0]

                        if not rollout:
                                with torch.no_grad():
                                        preds, features, loss, = model(x, y)
                                        # preds_id = F.softmax(preds['id'] / 0.8, dim=-1).squeeze(0)
                                        # ix = torch.multinomial(preds_id, num_samples=1).flatten()
                                        att = AttentionVis.get_attention(module, n_blocks, T)
                                        ## predict iteratively
                                        # ix, att = self.predict_iteratively(model, mconf, x, stoi, top_k=0, top_p=0.5, temp=0.5, sample=True, pred_dt=False)
                        # with torch.no_grad():
                        if agg: 
                                t_seq = int(T - x['pad'])
                                # att = att - att.mean(axis=-2, keepdims=True)
                                # att = att - att.mean(axis=(0, 1, 2), keepdims=True)
                                if not rollout:
                                        att = np.max(att, axis=1)
                                        att = np.sum(att, axis=0)
                                        # att = np.max(att, axis=(0, 1))
                                # att = np.mean(att, axis=0)
                                # att = att[-1]   # take last layer
                                # att_n = LA.norm(att, axis=-1, ord=2, keepdims=True)
                                # att = softmax(att, axis=-2)
                                # att = normalize(att, norm='l2', axis=1)
                                # att = att / (n_L * n_H * (t_seq))
                                score = np.zeros((mconf.id_vocab_size, mconf.frame_block_size))
                                xid = x['id'].cpu().flatten().tolist()
                                yid = y['id'].cpu().flatten().tolist()
                                # score[ix] = att
                                score[y['id']] = att
                                # score[t_seq:] == 0
                        else:
                                score = att
                        
                        if attention_scores is None:
                                attention_scores = score[None, ...]
                        else:
                                attention_scores = np.vstack((attention_scores, score[None, ...]))
                        
                        if max_it is not None and it == max_it:
                                break

                        # att_dict[int(y['id'][:, n])] = step
                        # atts[tuple(x['interval'].cpu().numpy().flatten())] = att_dict
                return attention_scores
                        # take attentions from last step
        @torch.no_grad()
        def att_models(model, module, loader, n_blocks, block_size, pad_key=None):
                device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
                device = 'cpu'
                model.to(device)
                model = model.eval()
                mconf = model.config
                T = block_size
                attention_scores = np.zeros(mconf.id_vocab_size)
                pbar = tqdm(enumerate(loader), total=len(loader))
                for it, (x, y) in pbar:
                        pad = x[pad_key] if pad_key is not None else 0
                        # place data on the correct device
                        for key, value in x.items():
                                x[key] = x[key].to(device)
                        for key, value in y.items():
                                y[key] = y[key].to(device)
                        # forward model to calculate attentions
                        _, _, _ = model(x)
                        # scores = np.array(np.zeros(len(neurons)))
                        att = np.zeros(len(mconf.id_vocab_size))
                        score = AttentionVis.get_attention(module, n_blocks, T, pad)
                        score = np.sum(score, axis=0)   # sum over all heads 
                        score = np.sum(score, axis=0)   # sum over all steps
                        # take attentions from last step
                        # if score.size >= 1: score = score[-1]
                        # scores.append(score)
                        real_ids = x['id'][..., :T - pad].flatten()
                        for idx, code in enumerate(real_ids):
                                """ 
                                for each code in scores,
                                add its score to the array
                                """
                                code = int(code.item())
                                att[code] += score[idx]
                        attention_scores = np.vstack((attention_scores, att))
                return attention_scores.sum(axis=0)

        def heatmap2d(self, arr: np.ndarray, ax=None, alpha=0.5, clim=None, blur=0):
                ncolors = 256
                color_array = plt.get_cmap('jet')(range(ncolors))

                # change alpha values
                n = 20
                color_array[:,-1] = [0.0] * n +  np.linspace(0.0,1.0,(ncolors - n)).tolist()

                # create a colormap object
                map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

                # register this new colormap with matplotlib
                plt.register_cmap(cmap=map_object)
                if blur > 0:
                        arr = gaussian_filter(arr, blur)
                if ax:
                        h = ax.imshow(arr, cmap='rainbow_alpha', alpha=alpha)
                else:
                        h = plt.imshow(arr, cmap='rainbow_alpha', alpha=alpha)

                # plt.colorbar()
                # plt.colorbar(mappable=h)
                if clim is not None:
                        h.set_clim(clim)
                                              
        @torch.no_grad()
        def plot_stim_attention_step(self, dataset, n_embd, video_stack, attention_scores, ix_step=None):
                '''
                In: (S, ID, Frame)
                Out: Attention heatmaps of neurons (y) - frames (x): (S, ID, Frame)
                '''

                # ix_step = [1, 2, 3, 4]
                if ix_step is None:
                        ix_step = np.random.choice(len(attention_scores), 1)

                for step in ix_step:
                        interval_trials = dataset.t


                        H, W = video_stack[0].shape[-2], video_stack[0].shape[-1]
                        xy_res = int(n_embd ** (1/2))
                        print(xy_res)

                        # # step, layer, head, row = sorted_att_std # layer, head, 
                        # step = ix_step   # 5, 3   # layer, head

                        interval_trials = dataset.t
                        dataset_step = dataset[step]
                        x, y = dataset_step[0], dataset_step[1]
                        x_id = x['id'].flatten().tolist()
                        x_pad = int(x['pad'].flatten())
                        neuron_idx = x_id[: len(x_id) - x_pad]

                        ncol = 10
                        nrow = len(neuron_idx)
                        fig, ax = plt.subplots(figsize=(60, 4 * nrow), nrows=nrow, ncols=ncol)


                        # attention_scores[ix_step] /= attention_scores[ix_step].max()
                        # att_max, att_min = attention_scores[ix_step].max(), attention_scores[ix_step].min()
                        att_step = attention_scores[step]
                        # att_step = softmax(att_step, axis=0)   # softmax over IDs
                        att_mean, att_std = att_step.mean(), att_step.std()
                        att_min, att_max = att_step.max(), att_step.min()
                        # attention_scores[ix_step] = attention_scores[ix_step] - att_mean / att_std
                        for n, idx in enumerate(neuron_idx):
                                top_n = n
                                att = att_step[n]
                                att_min, att_max = att.min(), att.max()
                                att_mean, att_std = att.mean(), att.std()
                                # att = (att - att_mean) / att.std()
                                # att = softmax(att, axis=-1)
                                # att = att / att.max()
                                att_im = att.reshape(1, 20, H // xy_res, W // xy_res)
                                # att_im = att_im - att_im.mean(axis=1)
                                # att_im = (att_im - att_mean) / att_std
                                att_im = att_im[-1, :, :, :]
                                
                                t = interval_trials.iloc[ix_step]
                                t_trial = t['Trial'].item()

                                # print(n_stim, math.ceil(t['Interval'] * 20))
                                frame_idx = get_frame_idx(t['Interval'], 1/20)
                                frame_idx = frame_idx if frame_idx >= 20 else 20
                                im_interval = x['frames'][0]
                                # im_interval = video_stack[n_stim, frame_idx - 20: frame_idx]
                                # att_grid =  softmax(att_top_std_im)
                                # att_grid = np.repeat(att_im, xy_res, axis=-2)
                                # att_grid = np.repeat(att_grid, xy_res, axis=-1)
                                att_grid = F.interpolate(torch.as_tensor(att_im[None, ...]), size=(H, W), mode='bilinear', align_corners=False).numpy()[0]

                                
                                tdx_range = range(10, att_grid.shape[0])
                                for tdx in tdx_range:
                                        axis = ax[n][tdx - 10]
                                        # print(att_grid[tdx, :, :].shape)
                                        axis.imshow(im_interval[tdx], cmap='gray')
                                        # clim = (att_trials_id[ix_step].min(), att_trials_id[ix_step].max())
                                        std_n = 3
                                        self.heatmap2d(att_grid[tdx, :, :], ax=axis, alpha=0.85)        # , clim=(att_mean + att_std * std_n, att_mean + att_std * std_n))
                                        # axis.axis('off')
                                        axis.set_title(str(tdx))
                                        axis.set_xticks([])
                                        axis.set_yticks([])
                                        if tdx == min(tdx_range):
                                                axis.set_ylabel(f"ID {idx}", fontsize=40)
                                        # fig.suptitle(f'Neuron {idx}', y=0.8)
                        
                        # fig.supylabel('Neurons', fontsize=nrow * 6)
                        # fig.supxlabel('Frames (N)', fontsize=nrow * 6)
                        fig.suptitle(f"Interval {int(t['Interval'])} Trial {int(t['Trial'])}", fontsize=40)
                        plt.tight_layout()
                        # plt.savefig(f"SimNeu3D_Combo4, Interval {int(t['Interval'])} Trial {int(t['Trial'])}.png")
        
        @torch.no_grad()
        def predict_iteratively(self, model, mconf, x, stoi, temp, top_p, top_k, sample=True, pred_dt=True, device='cpu'):
                t = x['id'].shape[-1]
                pad = x['pad'] if 'pad' in x else 0
                x['id_full'] = x['id'][:, 0]
                x['id'] = x['id'][:, 0]
                x['dt_full'] = x['dt'][:, 0]
                x['dt'] = x['dt'][:, 0] if pred_dt else x['dt']
                T_id = mconf.id_block_size
                current_id_stoi = torch.empty(0, device=device)
                current_dt_stoi = torch.empty(0, device=device)
                att_total = None
                for i in range(T_id):
                        t_pad = torch.tensor([stoi['PAD']] * (T_id - x['id_full'].shape[-1]), device=device)
                        t_pad_dt = torch.tensor([0] * (T_id - x['dt_full'].shape[-1]), device=device)
                        x['id'] = torch.cat((x['id_full'], t_pad)).unsqueeze(0).long()
                        x['dt'] = torch.cat((x['dt_full'], t_pad_dt)).unsqueeze(0).long()

                        logits, features, _ = model(x)
                        logits['id'] = logits['id'][:, i] / temp
                        if pred_dt:
                                logits['dt'] = logits['dt'][:, i] / temp


                        att_step = AttentionVis.get_attention(model.neural_visual_transformer.neural_state_stimulus, mconf.n_stimulus_layers, mconf.id_block_size)
                        att_step = att_step[:, :, i]
                        att_total = att_step[None, ...] if att_total is None else np.concatenate((att_total, att_step[None, ...]))
                        # optionally crop probabilities to only the top k / p options
                        if top_k or top_p != 0:
                                logits['id'] = top_k_top_p_filtering(logits['id'], top_k=top_k, top_p=top_p)
                                if pred_dt:
                                        logits['dt'] = top_k_top_p_filtering(logits['dt'], top_k=top_k, top_p=top_p)

                        # apply softmax to logits
                        probs = F.softmax(logits['id'], dim=-1)
                        if pred_dt:
                                probs_dt = F.softmax(logits['dt'], dim=-1)
                        if sample:
                                ix = torch.multinomial(probs, num_samples=1)
                                if pred_dt:
                                        ix_dt = torch.multinomial(probs_dt, num_samples=1)
                                # ix = torch.poisson(torch.exp(logits), num_samples=1)
                        else:
                                # choose highest topk (1) sample
                                _, ix = torch.topk(probs, k=1, dim=-1)
                                if pred_dt:
                                        _, ix_dt = torch.topk(probs_dt, k=1, dim=-1) 
                        
                        # if ix > stoi['PAD']:
                        #     ix = torch.tensor([513])
                        
                        # convert ix_dt to dt and add to current time
                        current_id_stoi = torch.cat((current_id_stoi, ix.flatten()))
                        if pred_dt:
                                current_dt_stoi = torch.cat((current_dt_stoi, ix_dt.flatten()))
                        
                        # append true and predicted in lists
                        # get last unpadded token
                        x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
                        if pred_dt:
                                x['dt_full'] = torch.cat((x['dt_full'], ix_dt.flatten()))

                        if ix == stoi['EOS']: # and dtx == 0.5:    # dtx >= window:   # ix == stoi['EOS']:
                        # if len(current_id_stoi) == T_id - x['pad']:
                                # if ix != stoi['EOS']:
                                #     torch.cat((current_id_stoi, torch.tensor([stoi['EOS']])))
                                # if dtx <= window:
                                #     torch.cat((current_dt_stoi, torch.tensor([max(list(itos_dt.keys()))])))
                                id_prev_stoi = current_id_stoi
                                dt_prev_stoi = current_dt_stoi
                                break
                return x['id_full'].flatten().tolist()[1:], att_total.transpose(1, 2, 0, 3)
        
        @torch.no_grad()
        def plot_stim_attention_step_realtime(self, model, mconf, dataset, n_embd, video_stack, ix_step=None, rollout=False):
                '''
                In: (S, ID, Frame)
                Out: Attention heatmaps of neurons (y) - frames (x): (S, ID, Frame)
                '''

                # ix_step = [1, 2, 3, 4]
                if ix_step is None:
                        ix_step = np.random.choice(len(dataset), 1)

                dataset = dataset
                interval_trials = dataset.t


                H, W = video_stack.shape[-2], video_stack.shape[-1]
                xy_res = int(n_embd ** (1/2))

                # # step, layer, head, row = sorted_att_std # layer, head, 
                # step = ix_step   # 5, 3   # layer, head

                interval_trials = dataset.t
                data_step = dataset[ix_step]
                for key in data_step[0].keys():
                        data_step[0][key] = data_step[0][key].unsqueeze(0)
                x = data_step[0]
                x_id = dataset[ix_step][0]['id'].flatten().tolist()
                x_pad = int(dataset[ix_step][0]['pad'].flatten())
                neuron_idx = x_id[: len(x_id) - x_pad]

                print(x.keys())
                
                # model.eval()
                # with torch.no_grad():
                #         preds, features, loss, = model(x)
                # preds_id = F.softmax(preds['id'] / 0.95, dim=-1).squeeze(0)
                # ix = torch.multinomial(preds_id, num_samples=1).flatten().tolist()

                ix, att_step = self.predict_iteratively(model, mconf, x, dataset.stoi, top_k=0, top_p=0.85, temp=0.85, sample=True, pred_dt=False)
                print(f"ix: {ix}, att_step: {att_step.shape}")
                # ix = torch.argmax(preds_id, dim=-1)
                neuron_idx = []
                neuron_idx = []
                for idx in ix:
                        neuron_idx.append(idx)
                        if idx >= dataset.stoi['EOS']:
                                break
                
                no_frames = 6
                ncol = no_frames
                nrow = len(neuron_idx) if len(neuron_idx) > 1 else 2
                nrow = 5
                fig, ax = plt.subplots(figsize=(ncol * 6, 4 * nrow), nrows=nrow, ncols=ncol)

                # attention_scores[ix_step] /= attention_scores[ix_step].max()
                # att_max, att_min = attention_scores[ix_step].max(), attention_scores[ix_step].min()
                # att_step = AttentionVis.get_attention(model.neural_visual_transformer.neural_state_stimulus, mconf.n_stimulus_layers, mconf.id_block_size)
                att_step = att_step.max(axis=0).max(axis=0) if rollout is False else self.rollout_attentions(att_step)
                # att_step = softmax(att_step, axis=0)   # softmax over IDs
                att_mean, att_std = att_step.mean(), att_step.std()
                att_min, att_max = att_step.max(), att_step.min()
                # attention_scores[ix_step] = attention_scores[ix_step] - att_mean / att_std
                for n, idx in enumerate(neuron_idx):
                        if n > 4: break
                        top_n = n
                        att = att_step[n]
                        att_min, att_max = att.min(), att.max()
                        att_mean, att_std = att.mean(), att.std()
                        # att = (att - att_mean) / att.std()
                        # att = softmax(att, axis=-1)
                        # att = att / att.max()
                        att_im = att.reshape(1, 20, H // xy_res, W // xy_res)
                        # att_im = (att_im - att_mean) / att_std
                        att_im = att_im[-1, :, :, :]
                        
                        t = interval_trials.iloc[ix_step]
                        t_trial = t['Trial'].item()
                        if video_stack.shape[0] == 1:
                            n_stim = 0
                        elif video_stack.shape[0] <= 4:
                                if t['Trial'] <= 20: n_stim = 0
                                elif t['Trial'] <= 40: n_stim = 1
                                elif t['Trial'] <= 60: n_stim = 2
                        elif video_stack.shape[0] <= 8:
                                n_stim = int(t['Trial'] // 200) - 1

                        # print(n_stim, math.ceil(t['Interval'] * 20))
                        frame_idx = get_frame_idx(t['Interval'], 1/20)
                        frame_idx = frame_idx if frame_idx >= 20 else 20
                        frame_idx = frame_idx if frame_idx < video_stack.shape[1] else video_stack.shape[1]
                        im_interval = video_stack[n_stim, frame_idx - 20: frame_idx]

                        # att_grid =  softmax(att_top_std_im)
                        # att_grid = np.repeat(att_im, xy_res, axis=-2)
                        # att_grid = np.repeat(att_grid, xy_res, axis=-1)
                        print(att_im.shape)
                        att_grid = F.interpolate(torch.tensor(att_im[None, ...]), size=(H, W), mode='bilinear', align_corners=False).numpy()[0]

                        
                        tdx_range = range(10, 10 + no_frames)
                        for tdx in tdx_range:
                                axis = ax[n][tdx - 10]
                                # print(att_grid[tdx, :, :].shape)
                                axis.imshow(im_interval[tdx, 0], cmap='gray')
                                # clim = (att_trials_id[ix_step].min(), att_trials_id[ix_step].max())
                                std_n = 3
                                self.heatmap2d(att_grid[tdx, :, :], ax=axis, alpha=0.7, blur=2)
                                # axis.axis('off')
                                axis.set_title(str(tdx))
                                axis.set_xticks([])
                                axis.set_yticks([])
                                if tdx == min(tdx_range):
                                        axis.set_ylabel(f"ID {idx}", fontsize=40)
                                # fig.suptitle(f'Neuron {idx}', y=0.8)
                
                # fig.supylabel('Neurons', fontsize=nrow * 6)
                # fig.supxlabel('Frames (N)', fontsize=nrow * 6)
                fig.suptitle(f"Interval {int(t['Interval'])} Trial {int(t['Trial'])}", fontsize=40)
                plt.tight_layout()
                # plt.savefig(f"SimNeu3D_Combo4, Interval {int(t['Interval'])} Trial {int(t['Trial'])}.png")
        
        
        @torch.no_grad()
        def plot_stim_attention_time_agg(self, dataset, mconf, video_stack, attention_scores, ix_step=None):
                '''
                In: (I, ID, Time, Frame)
                Out: Attention heatmaps of neurons (y) - frames (x)
                '''

                # ix_step = [1, 2, 3, 4]
                if ix_step is None:
                        ix_step = np.random.choice(len(attention_scores), 1)

                dataset = dataset
                interval_trials = dataset.t


                H, W = video_stack.shape[-2], video_stack.shape[-1]
                xy_res = int(mconf.n_embd ** (1/2))

                # step, layer, head, row = sorted_att_std # layer, head, 
                step = ix_step   # 5, 3   # layer, head

                interval_trials = dataset.t
                x_id = dataset[int(ix_step)][0]['id'].flatten().tolist()
                x_pad = int(dataset[int(ix_step)][0]['pad'].flatten())
                neuron_idx = x_id[: len(x_id) - x_pad]

                ncol = 10
                nrow = len(neuron_idx)
                fig, ax = plt.subplots(figsize=(60, 4 * nrow), nrows=nrow, ncols=ncol)


                print(neuron_idx)
                for n, idx in enumerate(neuron_idx):
                        top_n = n
                        att_idx = ix_step, n  # att_idx_1[0], att_idx_1[1], att_idx_1[2], ix
                        att = attention_scores[att_idx]
                        att = att / att.max()
                        att_im = att.reshape(1, 20, H // xy_res, W // xy_res)
                        att_im = att_im[-1, :, :, :]
                        
                        t = interval_trials.iloc[att_idx[0]]
                        if video_stack.shape[0] == 1:
                            n_stim = 0
                        elif video_stack.shape[0] <= 4:
                                if t['Trial'] <= 20: n_stim = 0
                                elif t['Trial'] <= 40: n_stim = 1
                                elif t['Trial'] <= 60: n_stim = 2
                        elif video_stack.shape[0] <= 8:
                                n_stim = int(t['Trial'] // 200) - 1

                        # print(n_stim, math.ceil(t['Interval'] * 20))
                        t_interval = math.ceil(t['Interval'] * 20)
                        im_interval = video_stack[n_stim, t_interval - 20: t_interval]

                        # att_grid =  softmax(att_top_std_im)
                        att_grid = np.repeat(att_im, (H // xy_res), axis=-2)
                        att_grid = np.repeat(att_grid, (H // xy_res), axis=-1)

                        for tdx in range(10, att_grid.shape[0]):
                                axis = ax[n][tdx - 10]
                                # print(att_grid[tdx, :, :].shape)
                                axis.imshow(im_interval[tdx, 0], cmap='gray')
                                # clim = (att_trials_id[ix_step].min(), att_trials_id[ix_step].max())
                                self.heatmap2d(att_grid[tdx, :, :], ax=axis, alpha=0.6, clim=None)
                                axis.axis('off')
                                axis.set_title(str(tdx))
                                axis.set_ylabel(f"Neuron {idx}")
                                # fig.suptitle(f'Neuron {idx}', y=0.8)

                fig.suptitle(f"Interval {int(t['Interval'])} Trial {int(t['Trial'])}", fontsize=30, y=0.9)
                # plt.savefig(f"SimNeu3D_Combo4, Interval {int(t['Interval'])} Trial {int(t['Trial'])}.png")

        
        def plot_stim_att_layer_head(self, dataset, mconf, video_stack, attention_scores, n_embd, ix_step=None):
                """
                In: (I, Layer, Head, ID, Frame)
                Out: Attention heatmaps for neurons
                """

                H, W = video_stack.shape[-2], video_stack.shape[-1]
                xy_res = int(n_embd ** (1/2))

                # # ix_step = [1, 2, 3, 4]
                if ix_step is None:
                        ix_step = np.random.choice(len(attention_scores), 1)

                ncol = mconf.n_head
                nrow = mconf.n_stimulus_layers

                interval_trials = dataset.t

                # sorted_att_std = np.unravel_index(np.argsort(-att_trials_id_std.ravel()), att_trials_id_std.shape)
                # step, layer, head, row = sorted_att_std # layer, head, 
                # step = ix_step   # 5, 3   # layer, head

                xid = dataset[ix_step][0]['id'].flatten().tolist()
                x_pad = int(dataset[ix_step][0]['pad'].flatten())
                neuron_idx = xid[: len(xid) - x_pad]

                fig, ax = plt.subplots(figsize=(60, 4 * nrow), nrows=nrow, ncols=ncol)


                for n, idx in enumerate([ix_step]):
                        print(idx)
                        xid_n = np.random.choice(range(len(neuron_idx)), 1)
                        att_n = attention_scores[int(idx), :, :, int(xid_n)]
                        for layer in range(att_n.shape[0]):
                                for head in range(att_n.shape[1]):
                                        att_l_h = att_n[layer, head]
                                        att_l_h = att_l_h / att_l_h.max()
                                        att_im = att_l_h.reshape(1, 20, H // xy_res, W // xy_res)
                                        att_im = att_im[-1, :, :, :]
                                        
                                        t = interval_trials.iloc[ix_step]
                                        if video_stack.shape[0] == 1:
                                                n_stim = 0
                                        elif video_stack.shape[0] <= 4:
                                                if t['Trial'] <= 20: n_stim = 0
                                                elif t['Trial'] <= 40: n_stim = 1
                                                elif t['Trial'] <= 60: n_stim = 2
                                        elif video_stack.shape[0] <= 8:
                                                n_stim = int(t['Trial'] // 200) - 1

                                        t_interval = math.ceil(t['Interval'] * 20)
                                        im_interval = video_stack[n_stim, t_interval - 20: t_interval]

                                        # att_grid =  softmax(att_top_std_im)
                                        att_grid = np.repeat(att_im, xy_res, axis=-2)
                                        att_grid = np.repeat(att_grid, xy_res, axis=-1)

                                        axis = ax if nrow and ncol == 1 else ax[layer][head]
                                        # plt.subplot(nrow, ncol, n + layer + head + 1)
                                        axis.imshow(im_interval[10, 0], cmap='gray')
                                        self.heatmap2d(att_grid[10, :, :], ax=axis, alpha=0.6, blur=0, clim=None)
                                        axis.axis('off')
                                        axis.set_title(f'Layer {layer}, Head {head}', fontsize=15)
                plt.suptitle(f'Interval {t_interval}, Neuron {neuron_idx[int(xid_n)]}', y=0.97, fontsize=30)
                # plt.savefig(f"SimNeu_att_layer_head_{neuron_idx[int(xid_n)]}_interval_{t_interval}.png")
        
        
        def export_att_frames(self, model, module, mconf, loader, video_stack, xy_res, path):
                """
                Input: 
                Attentions Scores of shape (S, L, H, ID, F)
                (where S = Steps, L = Layers, H = Heads, ID = Neurons, F = Frames)
                Video Stack of shape (T_idx, 1, H, W)
                (where T_idx = Frame Idx, 1 = Channels, H = Height, W = Width)

                Ouput:
                Attention heatmaps overlayed on stimulus
                """
                n_blocks = mconf.n_stimulus_layers
                T = mconf.id_block_size

                H, W = video_stack.shape[-2], video_stack.shape[-1]
                counter = 0
                for it, (x, y) in enumerate(loader):
                        # forward model to calculate attentions
                        _, _, _ = model(x)
                        # scores = np.array(np.zeros(len(neurons)))
                        score = AttentionVis.get_attention(module, n_blocks, T)
                        # att = self.rollout_attentions(score).sum(axis=0)
                        att = score.mean(axis=0).sum(axis=0).sum(axis=0)
                        # att = softmax(att, axis=-1)
                        att = att.reshape(20, H // xy_res, W // xy_res)
                        att_grid = np.repeat(att, (H // xy_res), axis=-2)
                        att_grid = np.repeat(att_grid, (H // xy_res), axis=-1)
                        att_grid = softmax(att_grid, axis=-1)
                        t_trial = x['trial'].item()
                        t_interval = math.ceil(x['interval'] * 20)
                        video_interval = x['frames'][0][0, 5:15]
                        if len(video_interval) < 10:
                                continue
                        for frame in range(len(att_grid[8:11])):
                                plt.imshow(video_interval[frame], cmap='gray')
                                self.heatmap2d(att_grid[frame], alpha=0.7, blur=2.5)
                                plt.savefig(f"{path}/natstim{str(counter).zfill(5)}.png")
                                plt.close()
                                counter += 1
                        
