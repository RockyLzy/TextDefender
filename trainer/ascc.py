import sys

from trainer import register_trainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from utils.luna import adv_utils
from .base import BaseTrainer
from .gradient import EmbeddingLevelGradientTrainer


@register_trainer('ascc')
class ASCCTrainer(EmbeddingLevelGradientTrainer):
    def __init__(self,
                 args,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        EmbeddingLevelGradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument('--alpha', default=10.0, type=float, help='weight of regularization')
        group.add_argument('--beta', default=4.0, type=float, help='weight of KL distance')
        group.add_argument('--num_steps', default=5, type=int, help='steps used to finetune weights of finding adv')

    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.cuda() for t in batch)
        golds = batch[3]

        clean_logits, adv_logits = self.forward(args, batch)
        clean_loss = torch.mean(self.loss_function(clean_logits, golds.view(-1)))
        adv_loss = F.kl_div(torch.softmax(adv_logits, dim=1).log(), torch.softmax(clean_logits, dim=1), None, None, 'batchmean')
        total_loss = clean_loss + args.beta * adv_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return total_loss.item()
