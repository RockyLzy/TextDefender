from trainer.base import BaseTrainer
from trainer import register_trainer
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from utils.info_regularizer import (CLUB, InfoNCE)

@register_trainer('infobert')
class InfoBertTrainer(BaseTrainer):
    def __init__(self, 
        args,
        data_loader: DataLoader, 
        model: nn.Module, 
        loss_function: _Loss, 
        optimizer: Optimizer, 
        lr_scheduler: _LRScheduler, 
        writer: SummaryWriter):
        super().__init__(data_loader, model, loss_function, optimizer, lr_scheduler=lr_scheduler, writer=writer)
        hidden_size = model.config.hidden_size
        self.mi_upper_estimator = CLUB(hidden_size, hidden_size, beta=args.beta).to(model.device)
        self.mi_estimator = InfoNCE(hidden_size, hidden_size).to(model.device)
 
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument('--adv-steps', default=3, type=int,
                            help='Number of gradient ascent steps for the adversary')
        group.add_argument('--adv-learning-rate', default=4e-2, type=float,
                            help='Step size of gradient ascent')
        group.add_argument('--adv-init-mag', default=8e-2, type=float,
                            help='Magnitude of initial (adversarial?) perturbation')
        group.add_argument('--adv-max-norm', default=0, type=float,
                            help='adv_max_norm = 0 means unlimited')
        group.add_argument('--adv-norm-type', default='l2', type=str,
                            help='norm type of the adversary')
        group.add_argument('--alpha', default=5e-3, type=float,
                            help='hyperparam of InfoNCE')
        group.add_argument('--beta', default=5e-3, type=float,
                            help='hyperparam of Info upper bound')
        group.add_argument('--cl', default=0.5, type=float,
                            help='lower bound of Local Anchored Feature Extraction')
        group.add_argument('--ch', default=0.9, type=float,
                            help='lower bound of Local Anchored Feature Extraction')
        group.add_argument('--info-seed', default=42, type=float,
                            help='seed for InfoBERT')

    @staticmethod
    def feature_ranking(grad, cl=0.5, ch=0.9):
        n = len(grad)
        import math
        lower = math.ceil(n * cl)
        upper = math.ceil(n * ch)
        norm = torch.norm(grad, dim=1)  # [seq_len]
        _, ind = torch.sort(norm)
        res = []
        for i in range(lower, upper):
            res += ind[i].item(),
        return res

    @staticmethod
    def get_seq_len(batch):
        lengths = torch.sum(batch[1], dim=-1)
        return lengths.detach().cpu().numpy()

    def _train_mi_upper_estimator(self, outputs, batch=None):
        hidden_states = outputs[1]  # need to set config.output_hidden = True
        last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
        embeddings = []
        lengths = self.get_seq_len(batch)
        for i, length in enumerate(lengths):
            embeddings.append(embedding_layer[i, :length])
        embeddings = torch.cat(embeddings)  # [-1, 768]   embeddings without masks
        return self.mi_upper_estimator.update(embedding_layer, embeddings)

    def _get_local_robust_feature_regularizer(self, args, outputs, local_robust_features):
        hidden_states = outputs[1]  # need to set config.output_hidden = True
        last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
        sentence_embeddings = last_hidden[:, 0]  # batch x 768  # CLS
        local_embeddings = []
        global_embeddings = []
        for i, local_robust_feature in enumerate(local_robust_features):
            for local in local_robust_feature:
                local_embeddings.append(embedding_layer[i, local])
                global_embeddings.append(sentence_embeddings[i])

        lower_bounds = []
        from sklearn.utils import shuffle
        local_embeddings, global_embeddings = shuffle(local_embeddings, global_embeddings, random_state=args.info_seed)
        for i in range(0, len(local_embeddings), args.batch_size):
            local_batch = torch.stack(local_embeddings[i: i + args.batch_size])
            global_batch = torch.stack(global_embeddings[i: i + args.batch_size])
            lower_bounds += self.mi_estimator(local_batch, global_batch),
        return -torch.stack(lower_bounds).mean()

    def local_robust_feature_selection(self, args, batch, grad):
        """
        :param input_ids: for visualization, print out the local robust features
        :return: list of list of local robust feature posid, non robust feature posid
        """
        grads = []
        lengths = self.get_seq_len(batch)
        for i, length in enumerate(lengths):
            grads.append(grad[i, :length])
        indices = []
        nonrobust_indices = []
        for i, grad in enumerate(grads):
            indices.append(self.feature_ranking(grad, args.cl, args.ch))
            nonrobust_indices.append([x for x in range(lengths[i]) if x not in indices])
        return indices, nonrobust_indices
    
    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device) for t in batch)
        word_embedding_layer = self.model.get_input_embeddings()

        # init input_ids and mask
        tr_loss, upperbound_loss, lowerbound_loss = 0.0, 0.0, 0.0
        input_ids, attention_mask, labels = batch[0], batch[1], batch[3]
        embeds_init = word_embedding_layer(input_ids)

        input_mask = attention_mask.float()
        input_lengths = torch.sum(input_mask, 1) # B 
        
        if args.adv_init_mag > 0:
            if args.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif args.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag,
                                                                args.adv_init_mag) * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)
        
        for astep in range(args.adv_steps):
            delta.requires_grad_()
            batch = (embeds_init + delta, batch[1], batch[2])


            # (1) backward
            outputs = self.forward(args, batch)
            logits = outputs[0]

            losses = self.loss_function(logits, labels.view(-1))
            loss = torch.mean(losses)
            loss = loss / args.adv_steps

            tr_loss += loss.item()

            if self.mi_upper_estimator:
                upper_bound = self._train_mi_upper_estimator(outputs, batch) / args.adv_steps
                loss += upper_bound
                upperbound_loss += upper_bound.item()

            loss.backward(retain_graph=True)

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()
            if self.mi_estimator:
                local_robust_features, _ = self.local_robust_feature_selection(args, batch, delta_grad)
                lower_bound = self._get_local_robust_feature_regularizer(args, outputs, local_robust_features) * \
                              args.alpha / args.adv_steps
                lower_bound.backward()
                lowerbound_loss += lower_bound.item()

            if astep == args.adv_steps - 1:  ## if no freelb, set astep = 1, adv_init=0
                # further updates on delta
                break

            # (3) update and clip
            if args.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + args.adv_learning_rate * delta_grad / denorm).detach()
                if args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                    reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
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

            embeds_init = word_embedding_layer(input_ids)
        # clear_mask()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()

        loss_dict = {"task_loss": tr_loss}
        if self.mi_upper_estimator:
            loss_dict.update({"upper_bound": upperbound_loss})
        if self.mi_estimator:
            loss_dict.update({"lower_bound": lowerbound_loss})
        return tr_loss
                                                
        