import math
from trainer import register_trainer
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Dict
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SequentialSampler
from torchnlp.samplers.bucket_batch_sampler import BucketBatchSampler
from transformers import PreTrainedTokenizer
from utils.hook import EmbeddingHook
from utils.my_utils import collate_fn, xlnet_collate_fn
from .gradient import TokenLevelGradientTrainer
from .base import BaseTrainer



def get_viable_candidates(vocab: Dict, language: str = 'english' ):
    def get_viable_chinese_candidate(vocab: Dict)  -> Dict[int, str]:
        candidate_dict = {}
        for index, char in enumerate(vocab.keys()):
            if len(char) == 1:
                if '\u4e00' <= char <= '\u9fff':
                    candidate_dict[index] = char
        return candidate_dict
    def get_viable_english_candidate(vocab: Dict)  -> Dict[int, str]:
        PRETRAINED_SUBWORD_PREFIX = ['Ġ', '##']
        candidate_dict = {}
        for index, key_in_vocab in tqdm(enumerate(vocab.keys())):
            tmp_key = key_in_vocab
            for prefix in PRETRAINED_SUBWORD_PREFIX:
                if tmp_key.startswith(prefix):
                    tmp_key = tmp_key.replace(prefix, '')
            if tmp_key.isalpha():
                candidate_dict[index] = tmp_key
        return candidate_dict

    return get_viable_chinese_candidate(vocab) if language == 'chinese' else get_viable_english_candidate(vocab)


@register_trainer('hotflip')
class HotflipTrainer(TokenLevelGradientTrainer):
    def __init__(self,
                 args,
                 tokenizer: PreTrainedTokenizer,
                 data_loader: DataLoader,
                 model: nn.Module,
                 loss_function: _Loss,
                 optimizer: Optimizer,
                 lr_scheduler: _LRScheduler = None,
                 writer: SummaryWriter = None, 
                 language: str = 'english'):
        TokenLevelGradientTrainer.__init__(self, data_loader, model, loss_function, optimizer, lr_scheduler, writer)

        self.constraints = self.build_constraint(language, tokenizer)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # for hotflip adversarial training, we make bucket-batch by the length of tokens
        # to avoid the influence of the added <pad> tokens when calculating the number of modified words
        dataset = data_loader.dataset
        sampler = SequentialSampler(data_loader.dataset)
        bucket_sampler = BucketBatchSampler(sampler, batch_size=data_loader.batch_size, drop_last=False, sort_key=lambda x:torch.sum(dataset[x][1]).item())
        self.data_loader = DataLoader(dataset, batch_sampler=bucket_sampler, collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)

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



    def build_constraint(self, language, tokenizer: PreTrainedTokenizer):
        vocab = tokenizer.get_vocab()
        candidate_dict = get_viable_candidates(vocab, language)
        if len(candidate_dict) == 0:
            return None
        constraint = torch.zeros(size=(len(vocab), len(vocab))).bool()
        for index in candidate_dict.keys():
            constraint[:, index] = True
        if torch.cuda.is_available():
            constraint = constraint.cuda()
        return constraint

    def apply_constraint(self,
                         raw_tokens_ids: torch.Tensor,
                         dir_dot_grad: torch.Tensor,
                         constraint: torch.Tensor):
        raw_token_constraint = constraint[raw_tokens_ids]
        raw_token_constraint = ~raw_token_constraint
        dir_dot_grad.masked_fill_(raw_token_constraint, -19260817.)

    def hotflip_attack(self,
                       raw_tokens_ids: torch.Tensor,
                       embedding: torch.Tensor,
                       gradient: torch.Tensor,
                       embedding_matrix: torch.Tensor,
                       replace_num: int,
                       constraint: torch.Tensor = None,
                       cls_token_id: int = None,
                       sep_token_id: int = None,
                       pad_token_id: int = None):

        # compute the direction vector dot the gradient
        prev_embed_dot_grad = torch.einsum("bij,bij->bi", gradient, embedding)
        new_embed_dot_grad = torch.einsum("bij,kj->bik", gradient, embedding_matrix)
        dir_dot_grad = new_embed_dot_grad - prev_embed_dot_grad.unsqueeze(-1)

        # apply some constraints based on the original tokens
        # to avoid semantic drift.
        # if searcher is not None:
        #     apply_constraint_(searcher, raw_tokens, dir_dot_grad)

        # supposing that vocab[0]=<pad>, vocab[1]=<unk>.
        # we set value of <pad> to be smaller than the <unk>.
        # if none of words in the vocab are selected, (all their values are -19260817)
        # the "argmax" will select <unk> instead of other words.
        if constraint is not None:
            self.apply_constraint(raw_tokens_ids, dir_dot_grad, constraint)

        # mask cls and sep
        # if cls_token_id is not None:
        #     dir_dot_grad[raw_tokens_ids == cls_token_id] = -19260818.

        # if sep_token_id is not None:
        #    dir_dot_grad[raw_tokens_ids == sep_token_id] = -19260818.

        # at each step, we select the best substitute(best_at_each_step)
        # and get the score(score_at_each_step), then select the best positions
        # to replace.
        score_at_each_step, best_at_each_step = dir_dot_grad.max(2)
        _, best_positions = score_at_each_step.topk(replace_num)

        # use the selected token index to replace the original one
        adv_tokens = raw_tokens_ids.clone()
        src = best_at_each_step.gather(dim=1, index=best_positions)
        adv_tokens.scatter_(dim=1, index=best_positions, src=src)

        # for nli, sentence 1 is supposed to be not changed
        # for all sentence, all pad is supposed to be not changed
        # sentence_mask = torch.cumsum(raw_tokens_ids == sep_token_ids, dim=1).bool()
        if cls_token_id is not None:
            adv_tokens[raw_tokens_ids == cls_token_id] = cls_token_id

        if sep_token_id is not None:
            adv_tokens[raw_tokens_ids == sep_token_id] = sep_token_id

        adv_tokens[raw_tokens_ids == pad_token_id] = pad_token_id
        return adv_tokens

    def get_adversarial_examples(self, args, batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        # init input_ids and mask, sentence length
        current_batch_ids, attention_mask, token_type_ids, golds = batch

        for astep in range(args.adv_steps):
            # (0) forward
            current_batch = (current_batch_ids, attention_mask, token_type_ids)
            logits = self.forward(args, current_batch)[0]

            # (1) backward
            losses = self.loss_function(logits, golds.view(-1))
            loss = torch.mean(losses)
            loss.backward()

            fw, bw = EmbeddingHook.reading_embedding_hook()
            replace_num = math.floor(fw.shape[1] * (args.adv_change_rate))
            current_batch_ids = self.hotflip_attack(current_batch_ids,
                                                    embedding=fw,
                                                    gradient=bw,
                                                    embedding_matrix=self.model.get_input_embeddings().weight,
                                                    replace_num=replace_num,
                                                    constraint=self.constraints,
                                                    cls_token_id=self.cls_token_id,
                                                    sep_token_id=self.sep_token_id,
                                                    pad_token_id=self.pad_token_id)
            self.model.zero_grad()
            self.optimizer.zero_grad()

        return (current_batch_ids, attention_mask, token_type_ids)

    def train(self, args, batch: Tuple[torch.Tensor]) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.cuda() for t in batch)
        golds = batch[3]

        adv_batch = self.get_adversarial_examples(args, batch)

        self.model.zero_grad()
        self.optimizer.zero_grad()

        # for hotflip adversarial training, clean batch is used to train models
        final_batch = tuple(torch.cat((input, adv_input),dim=0) for input, adv_input in zip(batch, adv_batch))
        golds = torch.stack((golds, golds))
        # (0) forward
        logits = self.forward(args, final_batch)[0]
        # (1) backward
        losses = self.loss_function(logits, golds.view(-1))
        loss = torch.mean(losses)
        loss.backward()
        # (2) update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        return loss.item()
