import math
import logging

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import set_plot_params
set_plot_params()
plt.rcParams["figure.figsize"] = (20,20)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from datetime import datetime
logger = logging.getLogger(__name__)

import os
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6   # these two numbers came from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, mconf=None):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mconf = mconf if mconf is not None else 0
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.train_losses = []
        self.test_losses = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.criterion = self.criterion.to(self.device)

    def save_checkpoint(self, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), parent_path + 'code/transformer_vid3/runs/models/{}-e:{}-b:{}-l:{}-h:{}-ne:{}-{}.pt'.format(
                                   datetime.now().strftime("%m-%d-%y-%H:%M"), 
                                   epoch, self.mconf.block_size, self.mconf.n_layer, self.mconf.n_head,
                                   self.mconf.n_embd, self.config.dataset)
                           )

    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                if p.grad is None:
                    ave_grads.append(0)
                    max_grads.append(0)
                else:
                    ave_grads.append(p.grad.abs().mean().to('cpu'))
                    max_grads.append(p.grad.abs().max().to('cpu'))
                plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
                plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
                plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
                plt.xticks(range(0,len(ave_grads), 1), [layer[7:] for layer in layers], size=20, rotation="vertical")
                plt.xlim(left=0, right=len(ave_grads))
                plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
                plt.xlabel("Layers", size=20)
                plt.ylabel("average gradient", size=20)
                plt.title("Gradient flow", size=30)
                plt.grid(True)
                plt.legend([Line2D([0], [0], color="c", lw=4),
                            Line2D([0], [0], color="b", lw=4),
                            Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
                plt.show()

    def contrastive_loss(self, image_features, neural_features, temp=0.5):
        """
        
        Multi-modal contrastive loss.
        
        # Mode1, Mode2: 
        IDs or Image features

        # return:
        logits, labels - for use in nn.CrossEntropyLoss(logits, labels)

        """
        
        Bid, Tid, Cid = image_features.size()
        Bim, Tim, Cim = neural_features.size()

        assert Tid==Tim, "feature sequences not equal"
        B = Bid = Bim
        T = Tid = Tim

        # resize
        image_features = image_features.contiguous().view(B, -1) # (B x T, C) 
        neural_features = neural_features.contiguous().view(B, -1) # (B x T, C)

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        neural_features = neural_features / neural_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = temp * image_features @ neural_features.t()
        logits_per_neurons = temp * neural_features @ image_features.t()

        # (a)symmetric loss function
        labels = torch.arange(B ).to(self.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_n = F.cross_entropy(logits_per_neurons, labels)
        loss = (1/2 * loss_i) + (1/2 * loss_n) 
    
        return loss


    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:
                # place data on the correct device
                for key, value in x.items():
                    x[key] = x[key].to(self.device)
                # x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, features, loss = model(x)
                    loss = self.contrastive_loss(features['frames'], features['id'])
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.detach().item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    
                    if config.show_grads is True:
                        self.plot_grad_flow(model.named_parameters())                    
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y>=0).sum() # number of tokens processed this step (i.e label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
            # # store epoch losses
            # if is_train:
            #     self.train_losses.append(np.array(losses))
            # else:
            #     self.test_losses.append(np.array(losses))
            
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if good_model:
                best_loss = test_loss
                self.save_checkpoint(epoch)
