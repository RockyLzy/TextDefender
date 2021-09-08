import sys

from trainer import register_trainer
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from utils.luna import adv_utils
from .base import BaseTrainer
from .gradient import EmbeddingLevelGradientTrainer


@register_trainer('dne')
class DNETrainer(EmbeddingLevelGradientTrainer):
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
        group.add_argument('--dir_alpha', default=1.0, type=float)  # 0.1 for imdb; 1.0 for agnews
        group.add_argument('--dir_decay', default=0.5, type=float)  # 0.1 for imdb; 0.5 for agnews
        group.add_argument('--hard_prob', default=False, type=bool)
        group.add_argument('--nbr_num', default=50, type=int)
        group.add_argument('--big_nbrs', default=True, type=bool)
        # group.add_argument('--nbr_file', default='/disks/sdb/lzy/adversarialBenchmark/dne_external_data/euc-top8-d0.7.json')
        group.add_argument('--adv_steps', default=10.0, type=float, help='epsilon ball')
        group.add_argument('--adv_iteration', default=3, type=int)

    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device) for t in batch)
        golds = batch[3]
        total_loss = 0.0

        # 0 forward, reset embedding_hook
        adv_utils.reset_embedding_hook()
        logits = self.forward(args, batch)[0]
        # 1 backward
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        total_loss += loss.item()
        # 2-adv_step+2, adversarial samples
        adv_utils.set_adv_mode(True)
        adv_utils.send("step", args.adv_steps)
        for astep in range(args.adv_iteration):
            logits = self.forward(args, batch)[0]
            losses = self.loss_function(logits, golds.view(-1))
            loss = torch.mean(losses)
            loss.backward()
            total_loss += loss.item()
        adv_utils.set_adv_mode(False)
        # 3 update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return total_loss
