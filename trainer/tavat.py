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

@register_trainer('tavat')
class TokenAwareVirtualAdversarialTrainer(BaseTrainer):
    def __init__(self, 
        args,
        data_loader: DataLoader, 
        model: nn.Module, 
        loss_function: _Loss, 
        optimizer: Optimizer, 
        lr_scheduler: _LRScheduler, 
        writer: SummaryWriter):
        super().__init__(data_loader, model, loss_function, optimizer, lr_scheduler=lr_scheduler, writer=writer)
        self.use_global_embedding = args.use_global_embedding
        if self.use_global_embedding:
            self.delta_global_embedding = TokenAwareVirtualAdversarialTrainer.delta_embedding_init(args, model)
        

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Trainer")
        BaseTrainer.add_args(parser)
        group.add_argument('--adv-steps', default=2, type=int,
                            help='Number of gradient ascent steps for the adversary')
        group.add_argument('--adv-learning-rate', default=5e-2, type=float,
                            help='Step size of gradient ascent')
        group.add_argument('--adv-init-mag', default=2e-1, type=float,
                            help='Magnitude of initial (adversarial?) perturbation')
        group.add_argument('--adv-max-norm', default=5e-1, type=float,
                            help='adv_max_norm = 0 means unlimited')
        group.add_argument('--adv-norm-type', default='l2', type=str,
                            help='norm type of the adversary')
        group.add_argument('--adv-change-rate', default=0.2, type=float,
                            help='change rate of a sentence')
        group.add_argument('--use-global-embedding', default=True, type=bool,
                            help='global embedding')


    @staticmethod
    def delta_embedding_init(args, model):
        vocab_size = model.config.vocab_size
        hidden_size = model.config.hidden_size
        delta_global_embedding = torch.zeros([vocab_size, hidden_size]).uniform_(-1,1) # 30522 bert # 50265 roberta# 21128 bert-chinese

        dims = torch.tensor([hidden_size]).float()
        mag = args.adv_init_mag / torch.sqrt(dims)
        delta_global_embedding = (delta_global_embedding * mag.view(1, 1))
        delta_global_embedding = delta_global_embedding.to(model.device)
        return delta_global_embedding


    @staticmethod
    def delta_lb_token(args, input_lengths, embeds_init, input_mask, delta_global_embedding, input_ids_flat, bs, seq_len):
        delta_lb, delta_tok = None, None

        dims = input_lengths * embeds_init.size(-1) # B x(768^(1/2))
        mag = args.adv_init_mag / torch.sqrt(dims) # B
        delta_lb = torch.zeros_like(embeds_init).uniform_(-1,1) * input_mask.unsqueeze(2)
        delta_lb = (delta_lb * mag.view(-1, 1, 1)).detach()

        gathered = torch.index_select(delta_global_embedding, 0, input_ids_flat) # B*seq-len D
        delta_tok = gathered.view(bs, seq_len, -1).detach() # B seq-len D
        
        denorm = torch.norm(delta_tok.view(-1,delta_tok.size(-1))).view(-1, 1, 1)  # norm in total degree?
        delta_tok = delta_tok / denorm # B seq-len D  normalize delta obtained from global embedding

        return delta_lb, delta_tok


    def train(self, args, batch: Tuple) -> float:
        assert isinstance(batch[0], torch.Tensor)
        batch = tuple(t.to(self.model.device) for t in batch)
        word_embedding_layer = self.model.get_input_embeddings()

        # init input_ids and mask
        input_ids, attention_mask, labels = batch[0], batch[1], batch[3]
        embeds_init = word_embedding_layer(input_ids)
        input_ids_flat = input_ids.contiguous().view(-1)

        input_mask = attention_mask.float()
        input_lengths = torch.sum(input_mask, 1) # B 

        total_delta = None
        tr_loss = 0
        bs, seq_len = embeds_init.size(0), embeds_init.size(1)

        if self.use_global_embedding:
            delta_lb, delta_tok = TokenAwareVirtualAdversarialTrainer.delta_lb_token(args, 
                        input_lengths, embeds_init, input_mask, self.delta_global_embedding, 
                        input_ids_flat, bs, seq_len)
        else:
            delta_lb, delta_tok = TokenAwareVirtualAdversarialTrainer.delta_lb_token(args, 
                        input_lengths, embeds_init, input_mask, word_embedding_layer.weight, 
                        input_ids_flat, bs, seq_len)
        
        for astep in range(args.adv_steps):
            delta_lb.requires_grad_()
            delta_tok.requires_grad_()

            inputs_embeds = embeds_init + delta_lb + delta_tok
            batch = (inputs_embeds, batch[1], batch[2])
            logits = self.forward(args, batch)[0]

            losses = self.loss_function(logits, labels.view(-1))
            loss = torch.mean(losses)
            tr_loss += loss.item()
            loss.backward(retain_graph=True)

            if astep == args.adv_steps - 1:
                delta_tok = delta_tok.detach()
                if self.use_global_embedding:
                    self.delta_global_embedding = self.delta_global_embedding.index_put_((input_ids_flat,), delta_tok, True)
                break
            
            if delta_lb is not None:
                delta_lb_grad = delta_lb.grad.clone().detach()
            if delta_tok is not None:
                delta_tok_grad = delta_tok.grad.clone().detach()

            denorm_lb = torch.norm(delta_lb_grad.view(bs, -1), dim=1).view(-1, 1, 1)
            denorm_lb = torch.clamp(denorm_lb, min=1e-8)
            denorm_lb = denorm_lb.view(bs, 1, 1)

            denorm_tok = torch.norm(delta_tok_grad, dim=-1)
            denorm_tok = torch.clamp(denorm_tok, min=1e-8)
            denorm_tok = denorm_tok.view(bs, seq_len, 1)

            delta_lb = (delta_lb + args.adv_learning_rate * delta_lb_grad / denorm_lb).detach()
            delta_tok = (delta_tok + args.adv_learning_rate * delta_tok_grad / denorm_tok).detach()

            # calculate clip
            delta_norm_tok = torch.norm(delta_tok, p=2, dim=-1).detach()
            mean_norm_tok, _ = torch.max(delta_norm_tok, dim=-1, keepdim=True)
            reweights_tok = (delta_norm_tok / mean_norm_tok).view(bs, seq_len, 1)

            delta_tok = delta_tok * reweights_tok

            total_delta = delta_tok + delta_lb

            if args.adv_max_norm > 0:
                delta_norm = torch.norm(total_delta.view(bs, -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)

                delta_lb = (delta_lb * reweights).detach()
                delta_tok = (delta_tok * reweights).detach()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()

        return tr_loss