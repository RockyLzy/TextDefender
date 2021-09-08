from trainer import register_trainer
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from .gradient import EmbeddingLevelGradientTrainer
from .base import BaseTrainer


@register_trainer('pgd')
class PGDTrainer(EmbeddingLevelGradientTrainer):
    def __init__(self,
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
        group.add_argument('--adv-steps', default=5, type=int,
                            help='Number of gradient ascent steps for the adversary')
        group.add_argument('--adv-learning-rate', default=0.03, type=float,
                            help='Step size of gradient ascent')
        group.add_argument('--adv-init-mag', default=0.05, type=float,
                            help='Magnitude of initial (adversarial?) perturbation')
        group.add_argument('--adv-max-norm', default=0.0, type=float,
                            help='adv_max_norm = 0 means unlimited')
        group.add_argument('--adv-norm-type', default='l2', type=str,
                            help='norm type of the adversary')
        group.add_argument('--adv-change-rate', default=0.2, type=float,
                            help='change rate of a sentence')


    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device) for t in batch)
        golds = batch[3]

        # for PGD-K, clean batch is not used when training
        adv_batch = self.get_adversarial_examples(args, batch)

        self.model.zero_grad()
        self.optimizer.zero_grad()

        # (0) forward
        logits = self.forward(args, adv_batch)[0]
        # (1) backward
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        # (2) update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()