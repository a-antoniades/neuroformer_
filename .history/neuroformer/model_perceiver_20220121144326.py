# from code.transformer_vid.utils import convert_weights
# import rotary_embedding_torch
from torch.nn.modules.activation import GELU, ReLU
# from data.OneCombo3.trainer import TrainerConfig
import math
import numpy as np
import itertools
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.models.video import r3d_18
# from ResNet3D import r3d_18

from scipy.optimize import linear_sum_assignment
# from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):    # nn.Conv3d,
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

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.2
    resid_pdrop = 0.2
    attn_pdrop = 0.2
    pos_pdrop = 0.2
    temp_pdrop = 0.2
    pos_emb = True
    temp_emb = True
    start_prune = 30
    epoch = 0

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class neuralGPTConfig:
    """ base GPT config, params common to all GPT versions """
    n = 0.4
    im_drop = 0.2
    id_drop = n
    embd_pdrop = n
    resid_pdrop = n
    attn_pdrop = n
    pos_pdrop = n
    temp_pdrop = n
    pos_emb = True
    temp_emb = True

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class VideoFeaturesExtractor(nn.Module):
    """ 
    R3D: (3 x T x H x W)
    H, W = 112
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(*(list(r3d_18(pretrained=True).children())[:-2]))
        convert_weights(self.backbone)
        # # freeze backbone
        # for k, v in self.backbone.named_parameters():
        #     v.requires_grad = False

    def forward(self, x):
        # B = Batch, T, C, Fm, H, W
        features = self.backbone(x)     # (B, C, T, H, W)
        B, C, T, H, W = features.shape
        features = features.permute(0, 2, 3, 4, 1)
        features = features.view(B, -1, C)
        return features

class VideoEncoder(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        p1, p2 = 16

        assert 
        c = n_embd / 
        self.to_patch_embedding = nn.Sequential(
            Rearrange(f'b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=16, p2=16)
        )
    
    def forward(self, x):
        return self.to_patch_embedding(x)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("mask", self.build_mask(config.block_size))  
        self.n_head = config.n_head

        self.att = None
        self.T = config.block_size

        # self.rotary_embedding = RotarySpatioTemporalEmbedding(config)
    
    def build_mask(self, block_size):
        mask = torch.tril(torch.ones((block_size, block_size)),
                                     ).view(1, 1, block_size, block_size)
        return mask
    
    def generate_sparse_mask(self, att, p, config):
        """
        Generate a sparse mask according to p.
        """
        assert p >= 0 and p <= 1, "p should be in [0, 1]"
        T = config.block_size
        mask = torch.rand((1, T)) < p
        mask = mask.repeat(T, 1)
        
        mask[0, 0] = False  # don't mask 1st step
        # check if any step is fully masked and umask it
        idx_all_true = (True == torch.all(mask, dim=0)).nonzero()
        for step in idx_all_true:
            sampler = torch.distributions.Uniform(low=0, high=step.item()+1)
            idx_false = sampler.sample((1,1)).long()
            mask[step, idx_false] = False

        # mask = mask.repeat(T, 1)
        mask = mask.view(1, 1, T, T).cuda() if att.is_cuda else mask.view(1, 1, T, T)
        att = att.masked_fill(mask, float('-inf'))
        return att

    def forward(self, x, pad=None, dtx=None):
        # B = Batch, T = Sequence, C = n_embed
        B, T, C = x.size()

        # calculate query, key, values for all head in batch and move head forward to the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # apply rotary embeddings
        # if dtx is not None:
        #     q, k = self.rotary_embedding(q, k, dtx)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        if self.training:
            att = self.generate_sparse_mask(att, 0.25, self.config)
        if pad is not None:
            for idx, i in enumerate(pad):
                att[idx, :, :, self.T - i:] = float('-inf')   # only able to see first padding token
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        self.att = att
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class PositionalEmbedding(nn.Module):
    """ Implement the PE function. """
    def __init__(self, n_embd, p_drop,  max_len=1500):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) *
                             -(math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


# class RotarySpatioTemporalEmbedding(nn.Module):
#     """ Rotary temporal embeddings - block_size = id_blk_sz """
#     def __init__(self, config):
#         super().__init__()
#         self.frame_block_size = config.frame_block_size
#         self.id_block_size = config.id_block_size
#         self.emb = RotaryEmbedding(dim=32)

#     def forward(self, q, k, t):
#         b = t.shape[0]
#         tf = self.frame_block_size
#         queries = []
#         keys = []
#         for B in range(b):
#             im_temp_emb = torch.tensor([-0.5] * (tf//2) + [0.5] * (tf//2))
#             im_pos_emb = torch.arange(self.frame_block_size)
#             im_emb = torch.stack([im_temp_emb, im_pos_emb], dim=0)
#             id_temp_emb = self.temp_emb(t[B], cache_key=self.block_size)
#             freqs = self.emb(torch.cat(im_emb, id_temp_emb))
#             queries.append(apply_rotary_emb(freqs, q[B][None, ...]))
#             keys.append(apply_rotary_emb(freqs, k[B][None, ...]))
#         q, k = torch.cat(queries), torch.cat(keys)
#         return q, k


class TemporalEmbedding(nn.Module):
    """ encoding temporal information using fourrier signals """
    def __init__(self, n_embd, p_drop, max_len=1500):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) *
                             -(math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class LearntTemporalEmbedding(nn.Module):
    """
    Project B x T x 1 time sequence to
            B x T x C
    """
    def __init__(self, block_sz, n_embd, p_drop=0.2):
        super().__init__()
        self.temp_emb = nn.Sequential(
            nn.Linear(1, n_embd // 2),
            nn.GELU(),
            nn.Linear(n_embd // 2, n_embd),
            nn.Dropout(p_drop)
        )
    
    def forward(self, x):
        return self.temp_emb(x.unsqueeze(-1))


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        # decoder_layer = nn.TransformerDecoderLayer(config.n_embd, config.n_head, 
        #                                            activation='gelu', dropout=0.2, batch_first=True)
        # self.decoder = nn.TransformerDecoder(decoder_layer, config.n_layer)
        self.decoder = nn.Transformer(d_model=config.n_embd, nhead=config.n_head, 
                                      num_encoder_layers=3, num_decoder_layers=config.n_layer,
                                      activation="gelu", dropout=0.4, batch_first=True)
        self.register_buffer("tgt_mask", self.generate_square_subsequent_mask(config.id_block_size))
        # self.register_buffer("tgt_pad_mask", self.generate_padding_mask(config.ids_block_size))
        self.T = config.id_block_size

    def generate_square_subsequent_mask(self, sz: int, pad=None):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz), diagonal=0) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_padding_mask(self, sz: int, pad=None):
        r"""Build a (B x T) mask that resides on the GPU and can be 
            manipulated by build_padding_mask according to padded sequence
        """
        mask = torch.zeros(1, sz, dtype=torch.bool)
        return mask

    def generate_sparse_mask(self, sz: int, pad=None):
        r""" Build a square mask that employs 
             teacher forcing according to P
        """
        rand_mat = torch.rand(1, sz)
        k = round(0.75 * sz)
        k_th_quant = torch.topk(rand_mat, k, largest = False)[0][:,-1:]
        bool_tensor = rand_mat <= k_th_quant
        mask = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0)).repeat(sz, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda(self.tgt_mask.get_device()) if self.tgt_mask.is_cuda else mask
    
    def build_padding_mask(self, tgt, pad):
        # mask = self.tgt_pad_mask.repeat(tgt.shape[0], 1)
        mask = torch.zeros(tgt.shape[0], self.T, dtype=torch.bool)
        for B, P in enumerate(pad):
            mask[B, self.T - P:] = True
        return mask # .to(torch.cuda.current_device())

    def forward(self, tgt, memory, pad):
        # padding_mask = self.build_padding_mask(tgt, pad)
        # tgt_mask = self.generate_sparse_mask(self.T) if self.training else self.tgt_mask
        return self.decoder(src=memory, tgt=tgt, tgt_mask=self.tgt_mask, 
                                         tgt_key_padding_mask=None)


class ProjectNorm(nn.Module):

    def __init__(self, feat_size, target_size):
        super().__init__()
        self.ln = nn.LayerNorm(feat_size)
        self.mlp = nn.Sequential(
            nn.Linear(feat_size, math.floor(2 * feat_size), bias=False),
            nn.GELU(),
            nn.Linear(math.floor(2 * feat_size), target_size, bias=False),
        )

    def forward(self, x):
        return self.mlp(self.ln(x))


class TimeProjection(nn.Module):
    
    def __init__(self, seq_size, id_seq_size, feat_size, target_size):
        super().__init__()
        self.mlp_seq = nn.Sequential(
            nn.Linear(seq_size, id_seq_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(id_seq_size, id_seq_size)
        )
        self.mlp_t = nn.Sequential(
            nn.Linear(feat_size, feat_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(feat_size // 2, target_size)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # B, T, C -> B, C, T
        x = self.mlp_seq(x)     # B, C, T / 2
        x = x.permute(0, 2, 1)  # B, T / 2, C
        return self.mlp_t(x)    # B, T / 2, 1


class PSTHProjection(nn.Module):
    """Takes Last Output of Block -> (B, C) 
       Builds PSTH table                  
    """
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.id_vocab_size, bias=False)
        )
    
    def forward(self, x):
        return self.mlp(x)


# class PSTHProjection(nn.Module):
    
#     def __init__(self, config):
#         super().__init__()
#         self.mlp_seq = nn.Sequential(
#             nn.Linear(config.id_block_size, config.id_block_size // 2, bias=False),
#             nn.GELU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(config.id_block_size // 2, 1, bias=False)
#         )
#         self.mlp_t = nn.Sequential(
#             nn.Linear(config.n_embd, config.n_embd * 4, bias=False),
#             nn.GELU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(config.n_embd * 4, config.id_vocab_size, bias=False)
#         )
    
#     def forward(self, x):
#         x = x.transpose(-1, -2)  # B, T, C -> B, C, T
#         x = self.mlp_seq(x)     # B, C, 1
#         x = x.transpose(-2, -1)  # B, 1, Vocab_id
#         return self.mlp_t(x)


class TimeRNN(nn.Module):
    def __init__(self, feat_size, target_size):
        super().__init__()


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, pad=None, dtx=None):
        x = x + self.attn(self.ln1(x), pad)
        x = x + self.mlp(self.ln2(x))
        return x


class BlockSequential(nn.Sequential):
    def forward(self, x, pad=None, dtx=None):
        for module in self._modules.values():
            x = module(x, pad, dtx)
        return x


class DiceLossPSTH(nn.Module):
    def __init__(self, size_average=True, smooth=1):
        super().__init__()
    
    def cross_entropy(self, input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))
    
    def forward(self, logits, targets, smooth=1, class_weights=None):
        total_logits = F.layer_norm(torch.sum(logits, dim=-2), [logits.size()[-1]])
        # probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(total_logits, dim=-1)
        # logits = F.gelu(logits)
        # probs = logits / (logits.max(dim=-1).values.unsqueeze(-1))
        # flatten label and prediction tensors
        outputs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        labels = torch.zeros_like(outputs)
        labels[targets] = 1 / len(targets)
        # intersection = (outputs * labels).sum()
        # dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
        return self.cross_entropy(outputs[None, ...], labels[None, ...])


class SetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def cross_entropy(self, input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))
    
    def forward(self, logits, targets):
        targets = targets.contiguous().view(-1)
        loss = 0
        for n_step, n_logits in enumerate(logits):
            n_logits = F.softmax(n_logits, dim=-1)
            n_target = targets[n_step:]
            n_target_dist = torch.zeros_like(n_logits)
            if len(n_target) != 0:
                n_target_dist[n_target] = 1 / len(n_target)
                loss += self.cross_entropy(n_logits[None,...], n_target_dist[None, ...])
        return loss / len(logits)


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.8, k=0.2, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=-1)
        Yg = torch.gather(p, 2, targets.unsqueeze(2))

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=-1)
        Yg = torch.gather(p, 2, targets.unsqueeze(2))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


# class PSTHLOSS(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, logits, targets):
#         total_logits = torch.sum(logits, dim=-2)    # sum over sequence dimension
#         probs = F.softmax(total_logits, dim=-1)
#         outptu


class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, logits, targets):
        T, C = logits.size()
        probs = F.softmax(logits, dim=-1)
        cost_id = (1 - probs[:, targets]).cpu().view(T, -1).unsqueeze(0)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_id.split(len(targets), -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.KLdiv = nn.KLDivLoss()
    def forward(self, logits, targets):
        log_probs = self.log_softmax(logits)
        return self.KLdiv(log_probs.long(), targets)


class PoissonCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)
        self.nll_poisson = nn.PoissonNLLLoss()
        # self.nll_poisson = nn.NLLLoss()

    def forward(self, logits, targets):
        log_probs = self.log_softmax(logits)
        return self.nll_poisson(log_probs, targets)


class GPT(nn.Module):
    """ the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.config = config
        # input embedding stem
        self.n_embd = config.n_embd
        self.tok_emb = nn.Embedding(config.id_vocab_size, config.n_embd)
        self.pos_emb = PositionalEmbedding(config.n_embd, p_drop=0.2)
        # self.pos_emb_id = nn.Parameter(torch.zeros(1, config.id_block_size, config.n_embd))
        self.pos_emb_frames = nn.Parameter(torch.zeros(1, config.frame_block_size, config.n_embd))
        # self.temp_emb = TemporalEmbedding(config.n_embd, p_drop=0.2)
        # self.temp_emb = RotaryTemporalEmbedding(config.id_block_size)
        self.temp_emb = LearntTemporalEmbedding(config.id_block_size, config.n_embd)
        self.frame_temp_emb = LearntTemporalEmbedding(config.frame_block_size, config.n_embd)
        self.id_drop = nn.Dropout(config.id_drop)
        self.im_drop = nn.Dropout(config.im_drop)
        self.drop = nn.Dropout(config.embd_pdrop)

        # -- Visual Backbone -- #
        # self.visual_backbone = VideoFeaturesExtractor()
        self.video_encoder = VideoEncoder()
        frame_temp_emb = torch.tensor(list(itertools.chain(*[[n * 0.05] * (config.frame_block_size//20) for n in range(20)]))).unsqueeze(0)
        self.register_buffer("frame_temp_emb_seq", frame_temp_emb)

        # -- Contrastive Loss -- ##
        # self.proj_id = ProjectNorm(config.n_embd, config.n_embd)
        # self.proj_vid = VidProjectNorm(config.n_embd, config.n_embd)    # im_shape
                
        ## -- IM_Decoder -- ##
        # self.blocks_id = BlockSequential(*[Block(config) for _ in range(2)])
        # self.blocks_im = BlockSequential(*[Block(config) for _ in range(2)])
        # self.ln_f_id = nn.LayerNorm(config.n_embd)
        # self.ln_f_im = nn.LayerNorm(config.n_embd)

        ## -- Decoder -- ##
        # self.ln_f = nn.LayerNorm(config.n_embd)
        ## GPT
        # self.blocks = BlockSequential(*[Block(config) for _ in range(config.n_layer)])
        # self.ln_f = nn.LayerNorm(config.n_embd)
        ## enc_dec
        self.state_decoder = Decoder(config)
        self.ln_f_state_dec = nn.LayerNorm(config.n_embd)
        self.stimulus_decoder = Decoder(config)
        self.ln_f_stimulus_dec = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
       
        ## -- Time -- ##
        # self.proj_time = TimeProjection(config.block_size, config.id_block_size, config.n_embd, config.n_dt)
        self.proj_time = ProjectNorm(config.n_embd, config.n_dt)
        # self.proj_time = ProjectNorm(config.n_embd, 1)
        
        ## -- PSTH -- ##
        # self.proj_psth = PSTHProjection(config)

        # Loss
        # self.dice_loss = DiceLossPSTH()
        # self.poisson_loss = PoissonCrossEntropyLoss()
        # self.hungarian_matcher = HungarianMatcher()
        # self.kldiv_loss = KLDivLoss()
        # self.truncated_loss = TruncatedLoss(trainset_size=config.data_size)
        # self.set_loss = SetLoss()
        # self.a = torch.tensor(0.5, requires_grad=True)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        
        if config.class_weights is not None:
            for key in config.class_weights.keys():
                self.register_buffer(f"class_weights_{key}", config.class_weights[key])  
        
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        Separates parameters into those who will experience weight decay and those that will not
        """
        if train_config.decay_weights:
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    else: no_decay.add(fpn)

            # special case the position embedding parameter in the root GPT module as not decayed
            black_list_mods = ['pos_emb', 'temp_emb']
            for mods in black_list_mods:
                for name, param in self.named_parameters():
                    if mods in name:
                        no_decay.add(name)    # also pos_emb
            
            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            no_decay -= decay & no_decay
            inter_params = decay & no_decay
            union_params = decay | no_decay

            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            
            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        
        else:
            parameters = self.parameters()
            optimizer = torch.optim.Adam(parameters, lr=train_config.learning_rate)
        
        return optimizer
    
    def process_features(self, x):
        # batch, block_size, feature
        p_idx = x['id_prev']
        idx = x['id']
        dtx = x['dt']
        dtx_prev = x['dt_prev']
        frames = self.video_encoder(x['frames'])
        pad = x['pad']

        b, t = idx.size()
        # b_p, t_p = p_idx.size()
        bf, tf = frames.size()[0:2]

        # forward the GPT model
        ''' 
        positional and temporal embeddings implemented in multiple ways, learnt, 
        fourrier decomposition and in the case of time, just passed as is. 
        '''
        # # Embeddings
        prev_id_position_embeddings = self.pos_emb(p_idx)
        prev_id_temporal_embeddings = self.temp_emb(dtx_prev.float())
        id_position_embeddings = self.pos_emb(idx)  
        im_position_embeddings = self.pos_emb_frames
        temporal_embeddings = self.temp_emb(dtx.float())
        
        # Extract ID features
        prev_token_embeddings = self.id_drop(self.tok_emb(p_idx) + prev_id_temporal_embeddings + prev_id_position_embeddings)
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        token_embeddings = token_embeddings + temporal_embeddings + id_position_embeddings
        token_embeddings = self.id_drop(token_embeddings)

        # Extract image features and add time embeddings
        im_temporal_embeddings = self.frame_temp_emb(self.frame_temp_emb_seq)
        im_embeddings = frames    # self.tok_emb(frames)
        im_embeddings = im_embeddings + im_position_embeddings + im_temporal_embeddings
        im_embeddings = self.im_drop(im_embeddings)   # separate pos emb?
        
        # Tidy up
        features = dict()
        features['id_prev'] = prev_token_embeddings
        features['id'] = token_embeddings
        features['frames'] = im_embeddings
        
        return features, pad

    def perceiver(self, features, pad):
        x = self.state_decoder(tgt=features['id'], memory=features['id_prev'], pad=pad)
        x = self.ln_f_state_dec(x)
        x = self.stimulus_decoder(tgt=features['id'], memory=features['frames'], pad=pad)
        x = self.ln_f_stimulus_dec(x)
        logits = self.head(x)

        return logits, x

    def enc_dec(self, features, pad):
        x = self.stimulus_decoder(tgt=features['id'], memory=features['frames'], pad=pad)
        x = self.ln_f_stimulus_dec(x)
        logits = self.head(x)

        return logits, x

    def GPTdecoder(self, features, pad, dtx=None):
        # image + neural features
        x = torch.cat((features['frames'], features['id']), dim=1)

        # Decoder
        x = self.blocks(x, pad, dtx)     # (B, T, C)
        x = self.ln_f(x)
        logits = self.head(x)

        # print(logits.shape)       # (B, T, Vocab)
        # logits_psth = x[:, -1]      # (B, C)

        return logits, x

    def forward(self, x, targets=None):
        idx = x['id']
        dtx = x['dt']
        frames = x['frames']
        pad = x['pad']

        b, t = idx.size()
        # b, t = x['id'].shape[0], x['id'].shape[1] + x['id_prev'].shape[1]
        bf, tf = frames.size()[0:2]
        tf = self.config.frame_block_size
        # assert t + tf == self.config.block_size, f"{tf} {t}"
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted"
        
        features, pad = self.process_features(x)
        logits, x = self.perceiver(features, pad)
        # logits, x = self.enc_dec(features, pad)
        # logits, x = self.GPTdecoder(features, pad)
        time = self.proj_time(x)    # (B, T_id, 1)

        # print(x[:, 0].shape)
        # psth = self.proj_psth(x)    # (B, Vocab_id)

        # if targets, calculate loss
        # calculate loss on logits up to padding token for each batch
        loss = None
        loss_frames = 0
        loss_id = []
        loss_time = []
        loss_dice = []
        loss_psth = []
        loss_hungarian = []
        if targets is not None:
            # loss_psth = self.dice_loss(psth, targets['modes'][:, tf:])    
            for B, P in enumerate(pad):
                tf = 0
                # im_logits = logits[B, :tf]
                # im_targets = targets['frames'][B, :tf]
                # loss_frames += F.cross_entropy(im_logits.view(-1, im_logits.size(-1)), im_targets.view(-1))
                id_logits = logits[B, tf:tf + t - P]
                id_targets = targets['id'][B, :t - P]

                loss_id_ = F.cross_entropy(id_logits.view(-1, id_logits.size(-1)), id_targets.view(-1), weight=self.class_weights_id)
                # if self.config.epoch >= 15:
                    # self.truncated_loss.update_weight(id_logits[None, ...], id_targets[None, ...], id_indexes[None, ...])
                # loss_id_ = self.truncated_loss(id_logits[None, ...], id_targets[None, ...], id_indexes[None, ...])
                time_preds = time[B, :t - P]
                time_targets = targets['dt'][B, :t - P]
                loss_time_ = F.cross_entropy(time_preds.view(-1, time_preds.size(-1)), time_targets.view(-1), weight=self.class_weights_dt)
                # loss_time_ = F.mse_loss(time_preds.squeeze(-1), time_targets)
                # loss_id_ = self.poisson_loss(id_logits.view(-1, id_logits.size(-1)), F.one_hot(id_targets, self.config.vocab_size))
                # if len(id_targets) > 0:
                #     indices = self.hungarian_matcher(id_logits, id_targets)
                #     probs_matching, targets_matching = id_logits[indices[0][0]], id_targets[indices[0][1]]
                #     loss_hungarian_ = F.cross_entropy(probs_matching, targets_matching, weight=self.class_weights).to(self.device)
                    # loss_hungarian.append(loss_hungarian_)
                #     # psth = self.proj_psth(x[B, -1]) # from the EOS position
                    
                    # loss_psth.append(torch.nan_to_num(self.set_loss(id_logits, id_targets)))
                    # loss_psth_ = self.dice_loss(id_logits, id_targets)
                    # loss_psth.append(torch.nan_to_num(loss_psth_))
                
                loss_time.append(torch.nan_to_num(loss_time_))
                loss_id.append(torch.nan_to_num(loss_id_))
            
            loss = dict()
            # loss['frames'] = loss_frames / (b / 3)
            loss['id'] = sum(loss_id) / (b * 2)   # sum(loss_id) / (b * 2)   # / len(loss_id)
            loss['time'] = sum(loss_time) / (b * 2)
            # loss['dice'] = sum(loss_dice) / len(loss_dice)
            # loss['dt'] = loss_time / (b * 50)
            # loss['hungarian'] = sum(loss_hungarian) / (b * 2)
            # loss['psth'] = sum(loss_psth) / (b * 2)

            for key in list(loss):
                if isinstance(loss[key], float):
                    del loss[key]
            
        preds = dict()
        preds['id'] = logits    # [:, tf:]    # only id logits
        preds['dt'] = time

        return preds, features, loss