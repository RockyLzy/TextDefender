from transformers import BertPreTrainedModel

from trainer import register_trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from utils.config import DATASET_LABEL_NUM
from .gradient import EmbeddingLevelGradientTrainer
from .base import BaseTrainer


@register_trainer('mixup')
class MixUpTrainer(EmbeddingLevelGradientTrainer):
    def __init__(self,
                 args,
                 data_loader: DataLoader,
                 model: BertPreTrainedModel,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        EmbeddingLevelGradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument('--alpha', default=2.0, type=float, help="alpha for beta distribution")
        group.add_argument('--mix_layers', nargs='*', default=[7, 9, 12], type=int, help="define mix layer set")
        group.add_argument('--adv_ratio', type=float, default=1.0, help="proportion of adv examples to sample.")
        group.add_argument('--gamma', type=float, default=1.0, help='gamma for L_mix loss')

    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.cuda() for t in batch)
        golds = batch[3]

        # 0 origin_data + augmented data
        logits = self.forward(args, batch)[0]
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)

        # 1 mixup
        batch_size = golds.size(0)
        idx = torch.randperm(batch_size)
        input_ids_2 = batch[0][idx]
        attention_mask_2 = batch[1][idx]
        token_type_ids_2 = batch[2][idx]
        labels_2 = batch[3][idx]
        labels = torch.zeros(batch_size, DATASET_LABEL_NUM[args.dataset_name]).cuda().scatter_(
            1, golds.view(-1, 1), 1
        )
        labels_2 = torch.zeros(batch_size, DATASET_LABEL_NUM[args.dataset_name]).to(args.device).scatter_(
            1, labels_2.view(-1, 1), 1
        )

        l = np.random.beta(args.alpha, args.alpha)
        # l = max(l, 1-l) ## not needed when only using labeled examples
        mixed_labels = l * labels + (1 - l) * labels_2

        mix_layer = np.random.choice(args.mix_layers, 1)[0]
        mix_layer = mix_layer - 1

        logits, _ = self.model(batch[0], batch[1], batch[2],
                               input_ids_2, attention_mask_2, token_type_ids_2,
                               l, mix_layer)
        probs = torch.softmax(logits, dim=1)  # (bsz, num_labels)
        loss_2 = F.kl_div(probs.log(), mixed_labels, None, None, 'batchmean')

        total_loss = args.gamma * loss + (1 - args.gamma) * loss_2
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return total_loss.item()
