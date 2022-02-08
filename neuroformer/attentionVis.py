import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

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
        
        
        # this is for gpt style models
        @torch.no_grad()
        def getAttention(x, model, blocks=None):
                idx = x['id']
                dtx = x['dt']
                frames = x['frames']
                pad = x['pad']
                
                features, pad = model.process_features(x)
                x = torch.cat((features['frames'], features['id']), dim=1)

                # aggregate attention from n_Blocks
                atts = None
                n_blocks = model.config.n_layer
                blocks = range(n_blocks) if blocks is None else blocks
                for n in range(n_blocks):
                        attBlock = model.blocks[n].attn
                        attBlock(x, pad).detach().to('cpu').numpy()    # forward model
                        att = attBlock.att.detach().to('cpu')
                        att = F.softmax(att, dim=-1).numpy()
                        att = att[:, 1, :, :,].squeeze(0)
                        atts = att if atts is None else np.add(atts, att)
                
                # # normalize
                # atts = atts / n_blocks
                return atts
        
        
        # this is for neuroformer model
        @torch.no_grad()
        def get_attention(module, n_blocks, block_size, pad=0):
                # aggregate attention from n_Blocks
                atts = None
                T = block_size
                # TODO: get index of 166, get attentions up until that stage
                for n in range(n_blocks):
                        att = module[n].attn.att
                        n_heads = att.size()[1]
                        # zero attention steps of paded positions
                        # att[:, :, :T_id - pad, :,] == 0
                        # att = att[:, :, :T_id - pad, :,]
                        # att = F.softmax(att, dim=-1).detach().to('cpu').numpy()
                        if pad != 0:
                                att = att[:, :, T - pad, :,].squeeze(0)
                        # att = torch.sum(att, axis=1, dtype=torch.float32).squeeze(0)
                        # att = F.softmax(att, dim=-1)
                        att = att.detach().squeeze(0).to('cpu').numpy()
                        # att = att / n_heads
                        # divide by number of blocks
                        atts = att if atts is None else np.add(atts, att)
                return atts
                

        @torch.no_grad()
        def att_models(models, dataset, neurons):
                ''' 
                Input list of models
                Returns Attentions over dataset
                '''
                models_atts = []
                for model in models:
                        attention_scores = np.zeros(len(neurons))
                        data = dataset
                        pbar = tqdm(enumerate(data), total=len(data))
                        for it, (x, y) in pbar:
                                # scores = np.array(np.zeros(len(neurons)))
                                att = np.zeros(len(neurons))
                                score = AttentionVis.getAttention(x, model)
                                if score.size >= 1: score = score[-1]
                                # scores.append(score)
                                for idx, neuron in enumerate(x[:, 0]):
                                        """ 
                                        for each neuron in scores,
                                        add its score to the array
                                        """
                                        neuron = int(neuron.item())
                                        att[neuron] += score[idx]
                                attention_scores = np.vstack((attention_scores, att))
                                if it > len(dataset):
                                        models_atts.append(attention_scores.sum(axis=0))
                                        break
                return models_atts