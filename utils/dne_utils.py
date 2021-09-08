import os

from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from overrides import overrides
import random
import numpy as np
from torch.nn.functional import embedding
from torch.nn.modules.sparse import Embedding
import torch.nn.functional as F

from utils.luna import batch_pad, adv_utils
from utils.luna import ram_has_flag, ram_set_flag
from utils.luna.cached_searcher import CachedWordSearcher
from functools import lru_cache
import torch



class WeightedHull:
    """
        Given tokens (token_num, ), return:
            nbr_tokens (token_num, max_nbr_num)
            coeff (token_num, max_nbr_num) --> Sometimes not required
    """

    def get_nbr_and_coeff(self, tokens, require_coeff=True):
        raise NotImplementedError


def build_neighbour_matrix(searcher, vocab):
    t2i = vocab.get_token_to_index_vocabulary("tokens")
    vocab_size = vocab.get_vocab_size("tokens")
    nbr_matrix = []
    for idx in range(vocab_size):
        token = vocab.get_token_from_index(idx)
        nbrs = [idx]
        for nbr in searcher.search(token):
            assert nbr in t2i
            nbrs.append(t2i[nbr])
        nbr_matrix.append(nbrs)
    nbr_lens = list(map(len, nbr_matrix))
    nbr_matrix = batch_pad(nbr_matrix)
    return nbr_matrix, nbr_lens


class SameAlphaHull(WeightedHull):
    def __init__(self, alpha, nbrs):
        self.alpha = alpha
        self.nbrs = nbrs

    @classmethod
    def build(cls, alpha, nbr_file, vocab, nbr_num, second_order):
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        searcher = CachedWordSearcher(
            nbr_file,
            vocab_list=t2i,
            second_order=second_order
        )
        nbrs, _ = build_neighbour_matrix(searcher, vocab)
        nbrs = torch.tensor(nbrs)[:, :nbr_num].cuda()
        return cls(alpha, nbrs)

    @overrides
    def get_nbr_and_coeff(self, tokens, require_coeff=True):
        nbr_tokens = self.nbrs[tokens]
        nbr_num_lst = (nbr_tokens != 0).sum(dim=1).tolist()
        max_nbr_num = max(nbr_num_lst)
        nbr_tokens = nbr_tokens[:, :max_nbr_num]
        if require_coeff:
            coeffs = dirichlet_sampling_fast(
                nbr_num_lst, self.alpha, self.nbrs.shape[1]
            )
            torch.cuda.empty_cache()
            coeffs = torch.Tensor(coeffs)[:, :max_nbr_num].to(tokens.device)
        else:
            coeffs = None
        return nbr_tokens, coeffs


class DecayAlphaHull(WeightedHull):
    def __init__(self, alpha, decay, nbrs, first_order_lens):
        self.alpha = alpha
        self.decay = decay
        self.nbrs = nbrs
        self.first_order_lens = first_order_lens

    @overrides
    def get_nbr_and_coeff(self, tokens, require_coeff=True):
        nbr_tokens = self.nbrs[tokens]
        first_order_lens_lst = self.first_order_lens[tokens].tolist()
        nbr_num_lst = (nbr_tokens != 0).sum(dim=1).tolist()
        max_nbr_num = max(nbr_num_lst)
        first_order_lens_lst = [min(num, max_nbr_num) for num in first_order_lens_lst]
        # try:
        nbr_tokens = nbr_tokens[:, :max_nbr_num]
        if require_coeff:
            coeffs = dirichlet_sampling_fast_2nd(
                nbr_num_lst, first_order_lens_lst, self.alpha, self.decay, self.nbrs.shape[1]
            )
            coeffs = torch.Tensor(coeffs)[:, :max_nbr_num].to(tokens.device)
        else:
            coeffs = None
        # except:
        #     import pdb; pdb.set_trace()
        return nbr_tokens, coeffs

    @classmethod
    def build(cls, alpha, decay, nbr_file, vocab, nbr_num, second_order):
        assert second_order
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        first_order_searcher = CachedWordSearcher(
            nbr_file, vocab_list=t2i, second_order=False
        )
        _, first_order_lens = build_neighbour_matrix(
            first_order_searcher, vocab
        )
        second_order_searcher = CachedWordSearcher(
            nbr_file, vocab_list=t2i, second_order=True
        )
        nbrs, second_order_lens = build_neighbour_matrix(
            second_order_searcher, vocab
        )
        nbrs = torch.tensor(nbrs)[:, :nbr_num].cuda()
        first_order_lens = torch.tensor(first_order_lens).cuda()
        return cls(alpha, decay, nbrs, first_order_lens)


_cache_dirichlet_size = 10000


@lru_cache(maxsize=None)
def _cache_dirichlet(alpha, vertex_num, max_vertex_num, v0_num=None, decay=None):
    if vertex_num == 0:
        return None
    # import pdb; pdb.set_trace()
    if alpha > 0.0:
        if v0_num is None:
            alphas = [alpha] * vertex_num
        else:
            alphas = [alpha] * v0_num + [alpha * decay] * (vertex_num - v0_num)
        diri = np.random.dirichlet(alphas, _cache_dirichlet_size).astype(np.float32)
    else:
        # if alpha == 0, generate a random one-hot matrix
        diri = np.eye(vertex_num)[np.random.choice(vertex_num, _cache_dirichlet_size)].astype(np.float32)
    zero = np.zeros((_cache_dirichlet_size, max_vertex_num - vertex_num),
                    dtype=np.float32)
    ret = np.concatenate((diri, zero), axis=1).tolist()
    return ret


_cache_probs_2nd = {}
_cache_offsets_2nd = {}


def dirichlet_sampling_fast_2nd(vertex_nums, v0_nums, alpha, decay, max_vertex_num):
    ret = []
    default_prob = [1.0] + [0.0] * (max_vertex_num - 1)
    for i, (s, s1) in enumerate(zip(vertex_nums, v0_nums)):
        if s == 0:
            ret.append(default_prob)
        else:
            hash_idx = s << 5 + s1
            if hash_idx not in _cache_probs_2nd:
                _cache_probs_2nd[hash_idx] = _cache_dirichlet(
                    alpha, s, max_vertex_num, s1, decay
                )
                _cache_offsets_2nd[hash_idx] = random.randint(0, _cache_dirichlet_size)
            _cache_offsets_2nd[hash_idx] = (_cache_offsets_2nd[hash_idx] + i) % _cache_dirichlet_size
            ret.append(_cache_probs_2nd[hash_idx][_cache_offsets_2nd[hash_idx]])
    return ret


def dirichlet_sampling_fast(vertex_nums, alpha, max_vertex_num):
    ret = []
    default_prob = [1.0] + [0.0] * (max_vertex_num - 1)
    cache_probs = []
    cache_offsets = []
    for v_num in range(max_vertex_num + 1):
        cache_probs.append(_cache_dirichlet(alpha, v_num, max_vertex_num))
        cache_offsets.append(random.randint(0, _cache_dirichlet_size))
    for i, n in enumerate(vertex_nums):
        if n == 0:
            ret.append(default_prob)
        else:
            cache_offsets[n] = (cache_offsets[n] + i) % _cache_dirichlet_size
            ret.append(cache_probs[n][cache_offsets[n]])
    return ret


def get_bert_vocab():
    vocab = Vocabulary(padding_token='[PAD]', oov_token='[UNK]')
    bert_indexer = PretrainedTransformerIndexer("bert-base-uncased", "tokens")
    bert_indexer._add_encoding_to_vocabulary_if_needed(vocab)
    assert vocab.get_vocab_size('tokens') == 30522
    return vocab


def combine_initial_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a (possibly higher order) tensor of ids with shape
    (d1, ..., dn, sequence_length)
    Return a view that's (d1 * ... * dn, sequence_length).
    If original tensor is 1-d or 2-d, return it as is.
    """
    if tensor.dim() <= 2:
        return tensor
    else:
        return tensor.view(-1, tensor.size(-1))


def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size) -> torch.Tensor:
    """
    Given a tensor of embeddings with shape
    (d1 * ... * dn, sequence_length, embedding_dim)
    and the original shape
    (d1, ..., dn, sequence_length),
    return the reshaped tensor of embeddings with shape
    (d1, ..., dn, sequence_length, embedding_dim).
    If original size is 1-d or 2-d, return it as is.
    """
    if len(original_size) <= 2:
        return tensor
    else:
        view_args = list(original_size) + [tensor.size(-1)]
        return tensor.view(*view_args)


class WeightedEmbedding(Embedding):
    def __init__(
            self,
            hull,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hull: WeightedHull = hull

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if not ram_has_flag("EXE_ONCE.weighted_embedding"):
            print("The weighted embedding is working")
            import sys
            sys.stdout.flush()
            ram_set_flag("EXE_ONCE.weighted_embedding")

        if ram_has_flag("warm_mode", True) or ram_has_flag("weighted_off", True):
            embedded = embedding(
                combine_initial_dims(tokens),
                self.weight,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse,
            )
            embedded = uncombine_initial_dims(embedded, tokens.size())
            return embedded
        nbr_tokens, _coeff = self.hull.get_nbr_and_coeff(tokens.view(-1))

        # n_words x n_nbrs x dim
        embedded = embedding(
            nbr_tokens,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        if not adv_utils.is_adv_mode():
            coeff_logit = (_coeff + 1e-6).log()
        else:
            last_fw, last_bw = adv_utils.read_var_hook("coeff_logit")
            # coeff_logit = last_fw + adv_utils.recieve("step") * last_bw
            norm_last_bw = last_bw / (torch.norm(last_bw, dim=-1, keepdim=True) + 1e-6)
            coeff_logit = last_fw + adv_utils.recieve("step") * norm_last_bw

        coeff_logit = coeff_logit - coeff_logit.max(1, keepdim=True)[0]

        coeff_logit.requires_grad_()
        adv_utils.register_var_hook("coeff_logit", coeff_logit)
        coeff = F.softmax(coeff_logit, dim=1)

        # if adv_utils.is_adv_mode():
        #     last_coeff = F.softmax(last_fw, dim=1)
        #     new_points = (embedded[:20] * coeff[:20].unsqueeze(-1)).sum(-2)
        #     old_points = (embedded[:20] * last_coeff[:20].unsqueeze(-1)).sum(-2)
        #     step_size = (new_points - old_points).norm(dim=-1).mean()
        #     inner_size = (embedded[:20, 1:] - embedded[:20, :1]).norm(dim=-1).mean()
        #     print(round(inner_size.item(), 3), round(step_size.item(), 3))
        embedded = (embedded * coeff.unsqueeze(-1)).sum(-2)
        embedded = embedded.view(*tokens.size(), self.weight.size(1))
        if adv_utils.is_adv_mode():
            if ram_has_flag("adjust_point"):
                raw_embedded = embedding(
                    tokens,
                    self.weight,
                    padding_idx=self.padding_idx,
                    max_norm=self.max_norm,
                    norm_type=self.norm_type,
                    scale_grad_by_freq=self.scale_grad_by_freq,
                    sparse=self.sparse,
                )
                delta = embedded.detach() - raw_embedded.detach()
                embedded = raw_embedded + delta
        return embedded