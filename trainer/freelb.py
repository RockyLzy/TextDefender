from trainer import register_trainer
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from .gradient import EmbeddingLevelGradientTrainer
from .base import BaseTrainer

@register_trainer('freelb')
class FreeLBTrainer(EmbeddingLevelGradientTrainer):
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
        word_embedding_layer = self.model.get_input_embeddings()

        # init input_ids and mask
        batch_in_token_ids = batch[0]
        attention_mask = batch[1]
        golds = batch[3]
        embedding_init = word_embedding_layer(batch_in_token_ids)

        delta = EmbeddingLevelGradientTrainer.delta_initial(args, embedding_init, attention_mask)

        total_loss = 0.0
        for astep in range(args.adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = (delta + embedding_init, batch[1], batch[2])
            logits = self.forward(args, batch)[0]

            # (1) backward
            losses = self.loss_function(logits, golds.view(-1))
            loss = torch.mean(losses)
            loss = loss / args.adv_steps
            total_loss += loss.item()
            loss.backward()

            if astep == args.adv_steps - 1:
                break

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            delta = EmbeddingLevelGradientTrainer.delta_update(args, embedding_init, delta, delta_grad)
            embedding_init = word_embedding_layer(batch_in_token_ids)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return total_loss