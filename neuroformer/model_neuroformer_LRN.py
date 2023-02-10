# from code.transformer_vid.utils import convert_weights
# import rotary_embedding_torch
from xml.etree.ElementPath import xpath_tokenizer

from grpc import xds_channel_credentials
from sympy import xfield
from torch.nn.modules.activation import GELU, ReLU
# from data.OneCombo3.trainer import TrainerConfig
import math
import numpy as np
import itertools
import logging
from inspect import isfunction

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import copy

from torchvision.models.video import r3d_18
# from ResNet3D import r3d_18
# from torchvision.models._utils import IntermediateLayerGetter
import torchmetrics

from scipy.optimize import linear_sum_assignment
# from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

from einops import rearrange
from einops.layers.torch import Rearrange

import timm


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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

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
    sparse_topk = None
    conv_layer = False
    self_att_layers = 0
    resnet_backbone = False
    vit_encoder = True
    dataset = None
    sparse_topk_id_prev = None


    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class neuralGPTConfig:
    """ base GPT config, params common to all GPT versions """
    n = 0.4
    im_drop = 0.4
    id_drop = n
    embd_pdrop = n
    resid_pdrop = n
    attn_pdrop = n
    pos_pdrop = n
    temp_pdrop = n
    pos_emb = True
    temp_emb = True
    epoch = 0
    step = 0
    sparse_topk = None
    conv_layer = False
    self_att_layers = 0
    sparse_topk = None

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



class Resnet3DBackbone(nn.Module):
    """ 
    R3D: (3 x T x H x W)
    H, W = 112
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(*(list(r3d_18(pretrained=True).children())[:4]))
        convert_weights(self.backbone)
        # freeze backbone
        for k, v in self.backbone.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # B = Batch, T, C, Fm, H, W
        features = self.backbone(x)     # (B, C, T, H, W)
        B, C, T, H, W = features.shape
        features = rearrange(features, 'B C T H W -> B T H W C')
        return features


#     def __init__(self, config):
#         super().__init__()
#         self.backbone = model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True).features[:27]
#         convert_weights(self.backbone)
#         # # freeze backbone
#         # for k, v in self.backbone.named_parameters():
#         #     v.requires_grad = False
#     def forward(self, x):
#         features = None
#         for i in range(len(x)):
#             x = x.tranpose(0, 1)    # Flip batch and T
#             feat = self.backbone(x[i])
#             if features is None:
#                 features = feat
#             else:
#                 features = torch.cat((features, feat))
#         # (B, t, C, H, W)
#         x = x.tranpose(0, 1)    # Flip batch and T
#         x = self.backbone(x)
#         return x


class VideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layer = config.conv_layer
        self.f_emb = int(config.n_embd_frames ** (1/2))
        
        if self.conv_layer:
            kernel_size = config.kernel_size
            self.conv_block = torch.nn.Sequential(
                    nn.Conv3d(1, config.n_embd, kernel_size=kernel_size, stride=kernel_size, padding=0),
                    Rearrange('b e t h w -> b t h w e'),
                    nn.LayerNorm([config.n_embd]),
                    nn.ReLU()
        )

        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c t (h p1) (w p2) -> b t h w (p1 p2 c)', p1=self.f_emb, p2=self.f_emb),
                nn.Linear(config.n_embd_frames, config.n_embd),
                nn.ReLU()
            )

    def forward(self, x):
        # if self.conv_layer:
        #     # x: (B, C, T, H, W)
        #     B, C, T, H, W = x.size()
        #     # Flip C and T and merge B and T]
        #     x = x.transpose(1, 2).view(-1, C, H, W)
        #     # Reshape to (B, C, T, H, W)
        #     x = x.view(B, C, T, H, W)
        # x = self.to_patch_embedding(x)
        # print(x.shape)
        if self.conv_layer:
            x = self.conv_block(x)
        else:
            x = self.to_patch_embedding(x)

        return x
        # return x

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:3]))
        # self.embedding = model[0:2]
        self.backbone = model[2][:2]
    
    def forward(self, x):
        x = rearrange(x, 'b t h w c -> b (t h w) c')
        features = []
        # for frame in x:
            # frame = self.embedding(frame)
        x = self.backbone(x)
        # features.append(x)
        return x

class MultiheadfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    
    """

    def __init__(self, config, kv_embd=None, sparse_topk=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

        kv_embd = kv_embd if kv_embd != None else config.n_embd
        # key, query, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(kv_embd, config.n_embd)
        self.value = nn.Linear(kv_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # self.register_buffer("mask", self.build_mask(config.id_block_size))  
        self.n_head = config.n_head
        self.sparse_topk = config.sparse_topk if sparse_topk is None else sparse_topk

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

    def forward(self, q, k=None, v=None, tgt_mask=None, pad=None, dtx=None, sparse_topk=None):
        assert torch.equal(k, v) is True, "Keys and Values must be the same"
        if None not in (k, v):
            assert k.size() == v.size(), "Keys and Values must be of same size"
            # assert q.size(-1) == k.size(-1) == v.size(-1), "Embedding dims must be of same size"
        else:
            k, v = q, q
        
        Bt, Tt, Ct = q.size()
        Bs, Ts, Cs = k.size()


        # calculate query, key, values for all head in batch and move head forward to the batch dim]
        q = self.query(q).view(Bt, Tt, self.n_head, Ct // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(k).view(Bs, Ts, self.n_head, Ct // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(v).view(Bs, Ts, self.n_head, Ct // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Normalize value
        # v = F.normalize(v, p=2.0, dim=-1)

        # # apply rotary embeddings
        # if dtx is not None:
        #     q, k = self.rotary_embedding(q, k, dtx)

        # causal self-attention; Self-attend: (B, nh, Tt, hs) x (B, nh, hs, Ts) -> (B, nh, Tt, Ts)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if tgt_mask is not None:
            att = att.masked_fill(tgt_mask[:,:,:Tt,:Tt] == 0, float('-inf'))
            # if self.training:
            #     att = self.generate_sparse_mask(att, 0.25, self.config)
            # if pad is not None and self.training:
            #     for idx, i in enumerate(pad):
            #         att[idx, :, :, Tt - i:] = float('-inf')   # only able to see first padding token
        

        # Explicit Sparse Attention - Use top-K qk values
        if self.sparse_topk is not None and self.sparse_topk < att.shape[-1]:
            top, _ = torch.topk(att, self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(att)
            mask = att < vk
            att.masked_fill_(mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        # self.att = att
        att = self.attn_drop(att)
        self.att = att
        y = att @ v # (B, nh, Tt, Ts) x (B, nh, Ts, hs) -> (B, nh, Tt, hs)
        y = y.transpose(1, 2).contiguous().view(Bt, Tt, Ct) # re-assemble all head outputs side by side

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
        # print(f"divterm: {div_term.shape}, pe: {pe.shape}, position: {position.shape}")
        # print(f"posdiv: {(position * div_term).shape}")
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # print(f"x: {x.shape}, pe: {self.pe[:, :x.size(1)].shape}")
        x = Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc

    
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class TemporalEmbedding(nn.Module):
    """ encoding temporal information using fourrier signals """
    def __init__(self, n_embd, p_drop, position=None, max_len=1200):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)

        # Compute the positional encodings once in log space.
        if position is None:
            pe = torch.zeros(max_len, n_embd)
            position = torch.arange(0, max_len).unsqueeze(1)
        else:
            pe = torch.zeros(max_len, n_embd)
        div_term = torch.exp(torch.arange(0, n_embd, 2) *
                             -(math.log(10000.0) / n_embd))
        # div_term = div_term.unsqueeze(0).repeat(batch_size, 1)
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('div_term', div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x.unsqueeze(-1)
        pe = self.pe.repeat(x.size(0), 1, 1)
        div_term = self.div_term
        pe = pe[:, :x.size(1), :]
        pe[:, :, 0::2] = torch.sin(x * div_term)
        pe[:, :, 1::2] = torch.cos(x * div_term)

        x = Variable(pe, requires_grad=False)
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


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.8, k=0.2, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes=None):
        p = F.softmax(logits, dim=-1)
        Yg = torch.gather(p, 2, targets.unsqueeze(2))

        loss = ((1-(Yg**self.q))/self.q) - ((1-(self.k**self.q))/self.q)
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



class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, logits, targets, device):
        T, C = logits.size()
        probs = F.softmax(logits, dim=-1)
        cost_id = (1 - probs[:, targets]).cpu().view(T, -1).unsqueeze(0)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_id.split(len(targets), -1))]
        return [(torch.as_tensor(i, dtype=torch.int64, device=device), torch.as_tensor(j, dtype=torch.int64, device=device)) for i, j in indices]

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, kv_embd=None, sparse_topk=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadfAttention(config, kv_embd, sparse_topk)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, q, k=None, v=None, mask=None, pad=None, dtx=None):
        x = q
        # x = self.ln1(x + self.attn(x, k, v, mask, pad))
        # x = self.ln2(x + self.mlp(x))
        x = x + self.attn(self.ln1(x), k, v, mask, pad)
        x = x + self.mlp(self.ln2(x))
        return self.ln_f(x)


class BlockSequential(nn.Sequential):
    def forward(self, q, k, v, mask=None, pad=None, dtx=None):
        for module in self._modules.values():
            x = module(q, k, v, mask, pad, dtx)
        return x



def contrastive_loss(image_features, neural_features, temp=0.5):

    device = image_features.device

    Bid, Tid = image_features.size()
    Bim, Tim = neural_features.size()

    assert Tid==Tim, "feature sequences not equal"
    B = Bid = Bim
    T = Tid = Tim

    # resize
    # image_features = image_features.contiguous().view(B * T, -1) # (B x T, C) 
    # neural_features = neural_features.contiguous().view(B * T, -1) # (B x T, C)

    # normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    neural_features = neural_features / neural_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = temp * image_features @ neural_features.t()
    logits_per_neurons = temp * neural_features @ image_features.t()

    # (a)symmetric loss function
    labels = torch.arange(B).to(device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_n = F.cross_entropy(logits_per_neurons, labels)
    loss = (1/2 * loss_i) + (1/2 * loss_n) 

    return loss


class CLIP(nn.Module):
    def __init__ (self, config):
        super().__init__()
        self.temp = config.clip_temp
        self.frame_proj = nn.Linear(config.frame_block_size * config.n_embd, config.clip_emb, bias=False)
        self.id_proj = nn.Linear(config.id_block_size * config.n_embd, config.clip_emb, bias=False)
        print(config.frame_block_size * config.n_embd)

    
    def forward(self, frame_feats, id_feats):

        # frame_feats = rearrange(frame_feats, 'b hw e -> b (hw e)')
        # id_feats = rearrange(id_feats, 'b t e -> b (t e)')

        # frame_proj = self.frame_proj(frame_feats)
        # id_proj = self.id_proj(id_feats)

        # frame_proj = frame_feats
        # id_proj = id_feats

        loss = contrastive_loss(frame_feats, id_feats, temp=self.temp)

        return loss

class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neural_state_blocks = _get_clones(Block(config, sparse_topk=config.sparse_topk_id), config.n_state_layers)
        if config.vit_encoder:
            self.frame_state_blocks = _get_clones(Block(config, sparse_topk=config.sparse_topk_frame), config.n_stimulus_layers)
        else:
            self.frame_state_blocks = None
        self.register_buffer("mask", self.build_mask(config.id_block_size))  
    
    def build_mask(self, block_size):
        mask = torch.tril(torch.ones((block_size, block_size)),
                                     ).view(1, 1, block_size, block_size)
        return mask
    
    def forward(self, neural, visual):
        for mod in self.neural_state_blocks:
            neural = mod(neural, neural, neural, self.mask)
        if self.frame_state_blocks is not None:
            for mod in self.frame_state_blocks:
                visual = mod(visual, visual, visual)
            
        features = dict()
        features['id'] = neural
        features['frames'] = visual
        
        return features
        

class MultimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.neural_state_block = nn.Sequential(*[Block(config, sparse_topk=config.sparse_topk_id) for _ in range(config.n_state_layers)])
        # self.neural_state_history_block = nn.Sequential(*[Block(config, sparse_topk=config.sparse_topk_id) for _ in range(config.n_state_history_layers)])
        # self.neural_state_history_self_attention = BlockSequential(*[Block(config) for _ in range(config.n_state_layers)])
        # self.neural_state_stimulus = BlockSequential(*[Block(config, sparse_topk=config.sparse_topk_frame) for _ in range(config.n_stimulus_layers)])

        # self.frame_encoder = _get_clones(Block(config, sparse_topk=config.sparse_topk_id), 4)
        self.neural_state_blocks = _get_clones(Block(config, sparse_topk=config.sparse_topk_id), config.n_state_layers)
        self.neural_state_history_blocks = _get_clones(Block(config, sparse_topk=config.sparse_topk_id), config.n_state_history_layers)
        self.neural_state_history_self_attention = _get_clones(Block(config, sparse_topk=config.sparse_topk_id_prev), config.self_att_layers)
        self.neural_state_stimulus_blocks =  _get_clones(Block(config, sparse_topk=config.sparse_topk_frame), config.n_stimulus_layers)
        # self.output_att = _get_clones(Block(config, sparse_topk=config.sparse_topk_id), 1)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.epoch = 0

        self.register_buffer("mask", self.build_mask(config.id_block_size))  
    
    def build_mask(self, block_size):
        mask = torch.tril(torch.ones((block_size, block_size)),
                                     ).view(1, 1, block_size, block_size)
        return mask
    
    def generate_sparse_mask(self, p, T):
        """
        Generate a sparse mask of length T, according to p.
        """
        assert p >= 0 and p <= 1, "p should be in [0, 1]"
        mask = torch.rand((1, T)) < 1 - p
        mask = mask.repeat(T, 1)
        
        mask[0, 0] = True  # don't mask 1st step
        # check if any step is fully masked and umask it
        idx_all_true = (False == torch.all(mask, dim=0)).nonzero()
        for step in idx_all_true:
            sampler = torch.distributions.Uniform(low=0, high=step.item()+1)
            idx_false = sampler.sample((1,1)).long()
            mask[step, idx_false] = True

        # mask = mask.repeat(T, 1)
        mask = torch.tril(mask.long()).view(1, 1, T, T)
        return mask.cuda() if self.mask.is_cuda else mask
    
    def build_padding_mask(self, tgt, pad):
        # mask = self.tgt_pad_mask.repeat(tgt.shape[0], 1)
        B, T = tgt.shape[0], tgt.shape[1]
        mask = torch.zeros(B, 1, 1, T, dtype=torch.bool)
        for B, P in enumerate(pad):
            mask[B, T - P:] = True
        return mask.to(torch.cuda.current_device())
    
    def forward(self, features):
        """
        Args:
            neural_state: [batch_size, seq_len_t, hidden_size]
            neural_history: [batch_size, seq_len_t, hidden_size]
            stimulus: [batch_size, seq_len_s, hidden_size]
        """
        self.epoch += 1/48
        if self.epoch == 10: print(self.epoch)
        if self.config.sparse_mask and self.epoch >= 10 and self.training:
            p = 0.4 / (1 + np.exp( -self.epoch / 10))
            mask = self.generate_sparse_mask(p, self.config.id_block_size)   # self.config.p_sparse
            # logger.info(f"p_sparse = {p}")
        else:
            mask = self.mask
            # mask = self.build_padding_mask(features['id'], features['pad'])

        neural_state = features['id']
        neural_history = features['id_prev']
        stimulus = features['frames']
        
        x = neural_state
        y = neural_history

        for mod in self.neural_state_stimulus_blocks:
            x = mod(x, stimulus, stimulus)
        for mod in self.neural_state_blocks:
            x = mod(x, x, x, mask)
        for mod in self.neural_state_history_blocks:
            x = mod(x, neural_history, neural_history)
        for mod in self.neural_state_history_self_attention:
            x = mod(x, x, x, mask)
        return x



class GPT(nn.Module):
    """ the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.config = config
        # -- Input Embedding Stem -- #        self.n_embd = config.n_embd
        self.tok_emb = nn.Embedding(config.id_vocab_size, config.n_embd)
        if config.pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.id_block_size, config.n_embd))
            self.pos_emb_prev = nn.Parameter(torch.zeros(1, config.prev_id_block_size, config.n_embd))
        if config.temp_emb:
            # self.temp_emb = LearntTemporalEmbedding(config.id_block_size, config.n_embd)
            # self.temp_emb_prev = LearntTemporalEmbedding(config.prev_id_block_size, config.n_embd)
            self.temp_emb = TemporalEmbedding(config.n_embd, config.id_drop)
            self.temp_emb_prev = TemporalEmbedding(config.n_embd, config.id_drop)
        
        self.id_drop = nn.Dropout(config.id_drop)
        self.im_drop = nn.Dropout(config.im_drop)
        self.drop = nn.Dropout(config.embd_pdrop)
        # self.frame_1d_emb = nn.Parameter(torch.zeros(1, config.frame_block_size, config.n_embd_frames))

        # LIF2 functions
        # if self.config.dataset == 'LIF2':
            # self.mlp_frames = nn.Linear(config.n_embd_frames, config.n_embd)
        self.mlp_frames = ProjectNorm(config.n_embd_frames, config.n_embd)
        # self.frame_emb = PositionalEncoding2D(config.n_embd)
        self.frame_emb = PositionalEmbedding(config.n_embd, config.im_drop)

        # -- Multimodal Transformer -- #
        if config.contrastive:
            self.clip = CLIP(config)
        self.neural_visual_transformer = MultimodalTransformer(config)
       
        ## -- ID, dt, Logit Projections -- ##
        self.head_id = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # ProjectNorm(config.n_embd, config.vocab_size)
        self.head_dt = nn.Linear(config.n_embd, config.n_dt, bias=False)        # ProjectNorm(config.n_embd, config.n_dt)

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
                        print(name)
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
            # optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate, momentum=0.9)
            print(f"weight_decay: {train_config.weight_decay}")
        
        else:
            parameters = self.parameters()
            # optimizer = torch.optim.SGD(parameters, lr=train_config.learning_rate, momentum=0.9)
            optimizer = torch.optim.Adam(parameters, lr=train_config.learning_rate)
        
        return optimizer
    
    def process_features(self, x):
        # batch, block_size, feature
        p_idx = x['id_prev']
        idx = x['id']
        dtx = x['dt']
        dtx_prev = x['dt_prev']
        # frames = self.video_encoder(x['frames']) if 'frames' in x else None
        frames = x['frames'].transpose(-1, -2) if 'frames' in x else None
        pad = x['pad']

        # forward the GPT model
        ''' 
        positional and temporal embeddings implemented in multiple ways, learnt, 
        fourrier decomposition and in the case of time, just passed as is. 
        '''
        # Embeddings
        prev_id_position_embeddings = self.pos_emb_prev if self.config.pos_emb else 0
        prev_id_temporal_embeddings = self.temp_emb_prev(dtx_prev.float()) if self.config.temp_emb else 0
        id_position_embeddings = self.pos_emb if self.config.pos_emb else 0
        # id_temporal_embeddings = self.temp_emb(dtx.float()) if self.config.temp_emb else 0
        id_temporal_embeddings = self.temp_emb(dtx.float()) if self.config.temp_emb else 0
        
        # Extract ID features
        prev_token_embeddings = self.id_drop(self.tok_emb(p_idx) + prev_id_temporal_embeddings + prev_id_position_embeddings)
        token_embeddings = self.id_drop(self.tok_emb(idx) + id_temporal_embeddings + id_position_embeddings) # each index maps to a (learnable) vector

        # features = self.feature_encoder(token_embeddings, frame_embeddings)
        features = dict()
        features['id'] = token_embeddings
        features['id_prev'] = prev_token_embeddings
        features['frames'] = frames

        # frames for LargeRandNet
        if 'frames' in x:
            # frame_embeddings_projection = nn.Linear(1000, self.config.n_embd)
            # frames = frame_embeddings_projection(frames)
            frames = self.mlp_frames(frames)
            frame_embeddings = self.frame_emb(frames.unsqueeze(-1))
            frame_embeddings = frames + frame_embeddings.squeeze(-1)
            features['frames'] = frame_embeddings

        token_embeddings, frame_embeddings = features['id'], features['frames']
        
        # Tidy up
        features['id_prev'] = prev_token_embeddings
        features['id'] = token_embeddings
        features['frames'] = frame_embeddings
        features['raw_frames'] = frames
        features['pad'] = pad
        
        return features, pad

    def forward(self, x, targets=None):
        idx = x['id']
        idx_prev = x['id_prev']
        dtx = x['dt']
        frames = x['frames'] if 'frames' in x else None
        pad = x['pad']

        b, t = idx.size()
        # assert t == t_prev, "Neural states need to be the same size!"
        tf = self.config.frame_block_size
        # assert t + tf == self.config.block_size, f"{tf} {t}"
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted"
        
        features, pad = self.process_features(x)
        # features['id'] =
        x = self.neural_visual_transformer(features)
        id_logits = self.head_id(x)
        dt_logits = self.head_dt(x)    # (B, T_id, 1)

        # if targets, calculate loss
        # calculate loss on logits up to padding token for each batch
        loss = None
        loss_frames = 0
        loss_id = []
        loss_time = []
        precision = []
        recall = []
        F1 = []
        torch.cuda.empty_cache()
        if targets is not None:
            for B, P in enumerate(pad):         
                
                id_logits_ = id_logits[B, :t - P]
                id_targets = targets['id'][B, :t - P]
                dt_logits_ = dt_logits[B, :t - P]
                time_targets = targets['dt'][B, :t - P]

                id_logits_ = id_logits[B]
                id_targets = targets['id'][B]
                dt_logits_ = dt_logits[B]
                time_targets = targets['dt'][B]

                if self.config.class_weights is not None:
                    loss_id_ = F.cross_entropy(id_logits_.view(-1, id_logits_.size(-1)), id_targets.view(-1), weight=self.class_weights_id)
                    loss_time_ = F.cross_entropy(dt_logits_.view(-1, dt_logits_.size(-1)), time_targets.view(-1), weight=self.class_weights_dt)
                else:
                    loss_id_ = F.cross_entropy(id_logits_.view(-1, id_logits_.size(-1)), id_targets.view(-1)) #, weight=self.class_weights_id)
                    loss_time_ = F.cross_entropy(dt_logits_.view(-1, dt_logits_.size(-1)), time_targets.view(-1)) #, weight=self.class_weights_dt)

                id_logits_ = id_logits_.squeeze(-1)
                if len(id_targets) > 0:

                    ## score metrics
                    probs_neurons = F.softmax(id_logits_, dim=-1)
                    _, ix_top_k = torch.topk(probs_neurons, k=1, dim=-1)
                    pred_neurons = ix_top_k.detach().flatten()
                    true_neurons = id_targets.detach().flatten()
                    precision_score = torchmetrics.functional.precision(true_neurons, pred_neurons, task='multiclass', num_classes=self.config.vocab_size).to(self.device)
                    recall_score = torchmetrics.functional.recall(true_neurons, pred_neurons, task='multiclass', num_classes=self.config.vocab_size).to(self.device)
                    F1_score = torchmetrics.functional.f1_score(true_neurons, pred_neurons, task='multiclass', num_classes=self.config.vocab_size).to(self.device)
                    
                    if (precision, recall, F1) is not None:
                        precision.append(precision_score)
                        recall.append(recall_score)
                        F1.append(F1_score)
                
                loss_id.append(loss_id_)
                loss_time.append(loss_time_)
        else:
            zero_tensor = torch.zeros(1).to(self.device)
            precision.append(zero_tensor)
            recall.append(zero_tensor)
            F1.append(zero_tensor)

        if targets is not None:    
            loss = dict()
            # loss['frames'] = loss_frames / (b / 3)
            n = float('inf')
            if self.config.contrastive:
                n = 2
                loss['clip'] = self.clip(features['frames'][:, 0], features['id'][:, -1]) * (1 / n) 
            loss['id'] = ((3 / 4) * sum(loss_id) / b) * (1 - 1 / n)   # sum(loss_id) / (b * 2)   # / len(loss_id)
            loss['time'] = ((1 / 4) * sum(loss_time) / b) * (1 - 1 / n) 
            
        preds = dict()
        preds['id'] = id_logits    # [:, tf:]    # only id logits
        preds['dt'] = dt_logits
        preds['precision'] = torch.stack(precision).mean()
        preds['recall'] = torch.stack(recall).mean()
        preds['F1'] = torch.stack(F1).mean()

        return preds, features, loss