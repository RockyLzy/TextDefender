import logging

import torch
from tqdm import tqdm
from typing import Tuple
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from utils.my_utils import convert_batch_to_bert_input_dict


class BaseTrainer:
    def __init__(self,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        self.data_loader = data_loader
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.global_step = 0

    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group("Trainer")
        # group.add_argument('--epochs', default=10, type=int, 
        #     help='training epochs')
        group.add_argument('--gradient_accumulation_steps', default=1, type=int,
            help='Number of updates steps to accumulate before performing a backward/update pass')
        group.add_argument('--learning-rate', default=2e-5, type=float,
            help='The initial learning rate for Adam')
        group.add_argument('--weight-decay', default=1e-6, type=float,
            help='weight decay')
        group.add_argument('--adam-epsilon', default=1e-8, type=float,
            help='epsilon for Adam optimizer')
        group.add_argument('--max-grad-norm', default=1.0, type=float,
            help='max gradient norm')
        group.add_argument('--learning-rate-decay', default=0.1, type=float,
            help='Proportion of training to perform linear learning rate warmup for')
        # group.add_argument('--compare-key', type=str, default='+accuracy',
        #     help="the key to compare when choosing the best modeling to be saved, default is '-loss'"+
        #     "where '+X' means the larger the X is, the better the modeling."+
        #     "where '-X' means the smaller the X is, the better the modeling."+
        #     "e.g., when X == '-loss', using loss to compare which epoch is best")

    def train_epoch(self, args, epoch: int) -> None:
        print("Epoch {}:".format(epoch))
        logging.info("Epoch {}:".format(epoch))
        self.model.train()

        epoch_iterator = tqdm(self.data_loader)
        oom_number = 0
        for batch in epoch_iterator:
            try:
                loss = self.train_batch(args, batch)
                epoch_iterator.set_description('loss: {:.4f}'.format(loss))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.warning('oom in batch forward / backward pass, attempting to recover from OOM')
                    print('oom in batch forward / backward pass, attempting to recover from OOM')
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    oom_number += 1
                else:
                    raise e
        logging.warning('oom number : {}, oom rate : {:.2f}%'.format(oom_number, oom_number / len(self.data_loader) * 100))
        return

    def train_batch(self, args, batch: Tuple) -> float:
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.train(args, batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.write_tensorboard(loss)
        self.global_step += 1
        return loss

    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device)for t in batch)
        logits = self.forward(args, batch)[0]
        golds = batch[3]
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def forward(self, args, batch: Tuple) -> Tuple:
        '''
        for Bert-like model, batch_input should contains "input_ids", "attention_mask","token_type_ids" and so on
        '''
        inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
        return self.model(**inputs)

    def write_tensorboard(self, loss: float, **kwargs):
        # if self.writer is not None:
        #     self.writer.add_scalar('loss', loss, self.global_step)
        pass