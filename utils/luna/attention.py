import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


def gen_mask_by_len(lens, max_len=-1, pad_zero=True, last_dim=0, device=None):
    if max_len == -1:
        max_len = max(lens)
    if pad_zero:
        mask = torch.tril(torch.ones(max_len, max_len, device=device))
    else:
        mask = torch.triu(torch.ones(max_len, max_len, device=device), diagonal=1)
    mask = mask.index_select(0, torch.tensor(lens, device=device) - 1).byte()
    if last_dim > 0:
        mask = mask.unsqueeze(2).repeat(1, 1, last_dim)
    return mask


def gen_att_mask(k_lens, max_k_len, max_q_len, device=None):
    mask = gen_mask_by_len(k_lens, max_len=max_k_len, pad_zero=False, last_dim=0, device=device)
    mask = mask.unsqueeze(1).expand(-1, max_q_len, -1)
    return mask


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,
                 d_q, d_k, d_v, d_out,
                 n_head, d_att_k, d_att_v,
                 dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_att_k = d_att_k
        self.d_att_v = d_att_v

        self.w_qs = nn.Linear(d_q, n_head * d_att_k)
        self.w_ks = nn.Linear(d_k, n_head * d_att_k)
        self.w_vs = nn.Linear(d_v, n_head * d_att_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_att_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k + d_att_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_v + d_att_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_att_k, 0.5))

        self.fc = nn.Linear(n_head * d_att_v, d_out)
        nn.init.xavier_normal_(self.fc.weight)

        # self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
            batch: q = k = v
            len:   k = v
            dim:   q = k = v = d_model
            About dim:
                dim of q & k =
                    d_q/d_k -> n_head * d_att_k
                dim of v =
                    d_v -> n_head * d_att_v
                dim of attentional results =
                    n_head * d_att_v -> d_out
            mask:  see ScaledDotProductAttention
        """
        d_att_k, d_att_v, n_head = self.d_att_k, self.d_att_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_att_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_att_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_att_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_att_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_att_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_att_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_att_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        # output = self.layer_norm(output)

        return output, attn


class PositionWiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
            batch: q = k = v
            len:   k = v
            dim:   q = k
            mask:  batch * len_q * len_k (for sequence of key, generate
                    batch * len_k to mask key's pad as one, then repeat
                    len_q times on the second dimension.)
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class VanillaAttention(nn.Module):
    def __init__(self,
                 query_size, mem_size, out_size,
                 dropout=0.1):
        super().__init__()
        if query_size != mem_size:
            self.aff_query = nn.Linear(query_size, mem_size, bias=False)
            nn.init.normal_(self.aff_query.weight, mean=0, std=np.sqrt(2.0 / (query_size + mem_size)))
        else:
            self.aff_query = None
        self.attention = ScaledDotProductAttention(temperature=np.power(mem_size, 0.5))
        if mem_size != out_size:
            self.aff_mem = nn.Linear(mem_size, out_size, bias=False)
            nn.init.normal_(self.aff_mem.weight, mean=0, std=np.sqrt(2.0 / (mem_size + out_size)))
        else:
            self.aff_mem = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, mask=None):
        if self.aff_query:
            query = self.aff_query(query)
        output, attn = self.attention(query, memory, memory, mask=mask)
        if self.aff_mem:
            output = self.aff_mem(output)
        output = self.dropout(output)
        return output, attn
