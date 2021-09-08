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
from .gradient import EmbeddingLevelGradientTrainer
from .base import BaseTrainer

import utils.certified.data_util as data_util


@register_trainer('ibp')
class IBPTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        BaseTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
        # Linearly increase the weight of adversarial loss over all the epochs to end up at the final desired fraction
        self.cert_schedule = torch.tensor(
            np.linspace(args.initial_cert_frac, args.cert_frac,
                        args.epochs - args.full_train_epochs - args.non_cert_train_epochs),
            dtype=torch.float, device=args.device)
        self.eps_schedule = torch.tensor(
            np.linspace(args.initial_cert_eps, args.cert_eps,
                        args.epochs - args.full_train_epochs - args.non_cert_train_epochs),
            dtype=torch.float, device=args.device)
        self.epoch = 0

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        # Model
        # group.add_argument('--hidden-size', '-d', type=int, default=100)
        group.add_argument('--pool', choices=['max', 'mean', 'attn'], default='mean')
        group.add_argument('--num-layers', type=int, default=3, help='Num layers for SNLI baseline BOW model')
        group.add_argument('--no-wordvec-layer', action='store_true',
                           help="Don't apply linear transform to word vectors")
        # group.add_argument('--early-ibp', action='store_true',
        #                    help="Do to_interval_bounded directly on base word vectors")
        group.add_argument('--no-relu-wordvec', action='store_true', help="Don't do ReLU after word vector transform")
        group.add_argument('--unfreeze-wordvec', action='store_true', help="Don't freeze word vectors")
        # group.add_argument('--glove', '-g', choices=vocabulary.GLOVE_CONFIGS, default='840B.300d')
        group.add_argument('--dropout-prob', type=float, default=0.1)
        # Adversary
        group.add_argument('--adversary', '-a', choices=['exhaustive', 'greedy', 'genetic'],
                           default=None, help='Which adversary to test on')
        group.add_argument('--adv-num-epochs', type=int, default=10)
        group.add_argument('--adv-num-tries', type=int, default=2)
        group.add_argument('--adv-pop-size', type=int, default=60)
        # group.add_argument('--use-lm', action='store_true', help='Use LM scores to define attack surface')
        # Training
        group.add_argument('--cert-frac', '-c', type=float, default=0.8,
                           help='Fraction of loss devoted to certified loss term.')
        group.add_argument('--initial-cert-frac', type=float, default=0.0,
                           help='If certified loss is being used, where the linear scale for it begins')
        group.add_argument('--cert-eps', type=float, default=1.0,
                           help='Max scaling factor for the interval bounds of the attack words to be used')
        group.add_argument('--initial-cert-eps', type=float, default=0.0,
                           help='If certified loss is being used, where the linear scale for its epsilon begins')
        group.add_argument('--full-train-epochs', type=int, default=1,
                           help='If specified use full cert_frac and cert_eps for this many epochs at the end')
        group.add_argument('--non-cert-train-epochs', type=int, default=0,
                           help='If specified train this many epochs regularly in beginning')
        # group.add_argument('--augment-by', type=int, default=0,
        #                    help='How many augmented examples per real example')
        # Data and files
        group.add_argument('--adv-only', action='store_true',
                           help='Only run the adversary against the model on the given evaluation set')
        # group.add_argument('--test', action='store_true', help='Evaluate on test set')
        # group.add_argument('--data-cache-dir', '-D', help='Where to load cached dataset and glove',
        #                    default='/disks/sdb/lzy/adversarialBenchmark/cache/ibp_cache')
        # group.add_argument('--neighbor-file', type=str, default=data_util.NEIGHBOR_FILE)
        # group.add_argument('--glove-dir', type=str, default=vocabulary.GLOVE_DIR)
        # group.add_argument('--dataset-dir', type=str, default='/disks/sdb/lzy/adversarialBenchmark/IBP_data/aclImdb')
        # group.add_argument('--snli-dir', type=str, default=entailment.SNLI_DIR)
        # group.add_argument('--dataset-lm-file', type=str,
        #                    default='/disks/sdb/lzy/adversarialBenchmark/IBP_data/lm_scores/imdb_all.txt')
        # group.add_argument('--snli-lm-file', type=str, default=entailment.LM_FILE)
        group.add_argument('--prepend-null', action='store_true', help='If true add UNK token to sequences')
        group.add_argument('--normalize-word-vecs', action='store_true', help='If true normalize word vectors')
        # group.add_argument('--downsample-to', type=int, default=None,
        #                    help='Downsample train and dev data to this many examples')
        # group.add_argument('--downsample-shard', type=int, default=0,
        #                    help='Downsample starting at this multiple of downsample_to')
        # group.add_argument('--truncate-to', type=int, default=None,
        #                    help='Truncate examples to this max length')
        # Loading
        # group.add_argument('--load-dir', '-L', help='Where to load checkpoint')
        # group.add_argument('--load-ckpt', type=int, default=None,
        #                    help='Which checkpoint to load')

    def set_epoch(self, epoch: int):
        self.cur_epoch = epoch

    def train(self, args, batch: Tuple) -> float:
        batch = tuple(t.to(args.device) for t in batch)
        if self.epoch < args.non_cert_train_epochs:
            cur_cert_frac = 0.0
            cur_cert_eps = 0.0
        else:
            cur_cert_frac = self.cert_schedule[
                self.epoch - args.non_cert_train_epochs] if self.epoch - args.non_cert_train_epochs < len(
                self.cert_schedule) else self.cert_schedule[-1]
            cur_cert_eps = self.eps_schedule[
                self.epoch - args.non_cert_train_epochs] if self.epoch - args.non_cert_train_epochs < len(
                self.eps_schedule) else self.eps_schedule[-1]

        self.optimizer.zero_grad()
        clean_loss = 0
        cert_loss = 0
        if cur_cert_frac > 0.0:
            out = self.model.forward(batch, cert_eps=cur_cert_eps)
            logits = out.val
            loss = self.loss_function(logits, batch[3].long().t().squeeze(0))
            loss = torch.mean(loss)
            clean_loss += loss.item()
            certified_loss = torch.max(self.loss_function(out.lb, batch[3]),
                                       self.loss_function(out.ub, batch[3]))
            loss = cur_cert_frac * certified_loss + (1.0 - cur_cert_frac) * loss
            cert_loss += certified_loss.item()
        else:
            # Bypass computing bounds during training
            logits = out = self.model.forward(batch, compute_bounds=False)
            loss = self.loss_function(logits, batch[3].long().t().squeeze(0))
            loss = torch.mean(loss)
        total_loss = loss.item()
        loss.backward()
        if any(p.grad is not None and torch.isnan(p.grad).any() for p in self.model.parameters()):
            nan_params = [p.name for p in self.model.parameters() if
                          p.grad is not None and torch.isnan(p.grad).any()]
            print('NaN found in gradients: %s' % nan_params, file=sys.stderr)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.optimizer.step()
        # if args.cert_frac > 0.0:
        #     print(
        #         f"Epoch {self.epoch}: train loss: {total_loss}, clean_loss: {clean_loss}, cert_loss: {cert_loss}")
        # else:
        #     print(f"Epoch {self.epoch}: train loss: {total_loss}")

        return total_loss
