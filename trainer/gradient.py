from trainer import register_trainer
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from utils.hook import EmbeddingHook
from .base import BaseTrainer

@register_trainer('gradient')
class GradientTrainer(BaseTrainer):
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        BaseTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

        EmbeddingHook.register_embedding_hook(self.model.get_input_embeddings())

    def get_adversarial_examples(self, args, batch: Tuple) -> Tuple[torch.Tensor]:
        raise NotImplementedError


@register_trainer('embedding')
class EmbeddingLevelGradientTrainer(GradientTrainer):
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        GradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

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

    @staticmethod
    def delta_initial(args, embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if args.adv_init_mag > 0:
            input_mask = attention_mask.to(embedding)
            input_lengths = torch.sum(input_mask, 1)
            if args.adv_norm_type == 'l2':
                delta = torch.zeros_like(embedding).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embedding.size(-1)
                magnitude = args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * magnitude.view(-1, 1, 1).detach())
            elif args.adv_norm_type == 'linf':
                delta = torch.zeros_like(embedding).uniform_(-args.adv_init_mag,
                                                             args.adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embedding)
        return delta

    @staticmethod
    def delta_update(args, embedding: torch.Tensor, delta: torch.Tensor, delta_grad: torch.Tensor) -> torch.Tensor:
        if args.adv_norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_learning_rate * delta_grad / denorm).detach()
            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask
                             + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()
        elif args.adv_norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_learning_rate * delta_grad / denorm).detach()
            if args.adv_max_norm > 0:
                delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
        else:
            print("Norm type {} not specified.".format(args.adv_norm_type))
            exit()
        return delta

    def get_adversarial_examples(self, args, batch: Tuple) -> Tuple:
        word_embedding_layer = self.model.get_input_embeddings()

        # init input_ids and mask, sentence length
        batch_in_token_ids = batch[0]
        attention_mask = batch[1]
        golds = batch[3]
        embedding_init = word_embedding_layer(batch_in_token_ids)

        delta = EmbeddingLevelGradientTrainer.delta_initial(args, embedding_init, attention_mask)

        for astep in range(args.adv_steps):
            # (0) forward
            delta.requires_grad_()
            batch = (delta + embedding_init, batch[1], batch[2])
            logits = self.forward(args, batch)[0]

            # (1) backward
            losses = self.loss_function(logits, golds.view(-1))
            loss = torch.mean(losses)
            loss.backward()

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            delta = self.delta_update(args, embedding_init, delta, delta_grad)

            self.model.zero_grad()
            self.optimizer.zero_grad()
            embedding_init = word_embedding_layer(batch_in_token_ids)

        delta.requires_grad = False
        return (embedding_init + delta, batch[1], batch[2])


# never used
class TokenLevelGradientTrainer(GradientTrainer):
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        GradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
