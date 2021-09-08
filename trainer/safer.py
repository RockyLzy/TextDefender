import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from transformers import PreTrainedTokenizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from data.instance import InputInstance
from data.reader import BaseReader, ClassificationReader
from utils.my_utils import convert_dataset_to_batch, collate_fn, xlnet_collate_fn, build_forbidden_mask_words
from utils.safer import WordSubstitude
from .base import BaseTrainer
from trainer import register_trainer

'''
SAFER trainer, from paper: https://arxiv.org/pdf/2005.14424.pdf
SAFER: A Structure-free Approach for Certified Robustness to Adversarial Word Substitutions
'''


@register_trainer('safer')
class SAFERTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 tokenizer,
                 data_reader: BaseReader,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None):
        BaseTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)
        self.tokenizer = tokenizer
        self.data_reader = data_reader
        self.augmentor = WordSubstitude(args.safer_perturbation_set)

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument('--safer-perturbation-set',
                           default='/disks/sdb/lzy/adversarialBenchmark/cache/embed/agnews/perturbation_constraint_pca0.8_100.pkl',
                           type=str, help='the perturbation set of safer')
        # group.add_argument('--safer-perturbation-set',
        #                    default='/disks/sdb/lzy/adversarialBenchmark/cache/embed/agnews/perturbation_textfooler_50.pkl',
        #                    type=str, help='the perturbation set of safer')

    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], InputInstance)
        perturb_instances = self.perturb_batch(batch)
        train_batch = self.data_reader._convert_instance_to_dataset(perturb_instances, tokenizer=self.tokenizer,
                                                                    use_tqdm=False)
        train_batch = convert_dataset_to_batch(train_batch, args.model_type)
        return super().train(args, train_batch)

    def perturb_batch(self, instances: List[InputInstance]) -> List[InputInstance]:
        result_instances = []
        for instance in instances:
            perturb_sentences = self.augmentor.get_perturbed_batch(instance.perturbable_sentence().lower())
            tmp_instances = []
            for sentence in perturb_sentences:
                tmp_instances.append(InputInstance.create_instance_with_perturbed_sentence(instance, sentence))
            result_instances.extend(tmp_instances)
        return result_instances
