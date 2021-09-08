import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaConfig
# from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaClassificationHead
# from transformers import *
# from transformers.models.bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertPreTrainedModel, BertModel
# from transformers.models.bert import BertModel

from utils.certified import ibp_utils
from utils.luna import batch_pad


class LSTMModel(nn.Module):
    """LSTM text classification model.

    Here is the overall architecture:
      1) Rotate word vectors
      2) Feed to bi-LSTM
      3) Max/mean pool across all time
      4) Predict with MLP

    """

    def __init__(self, word_vec_size, hidden_size, word_mat, device, num_labels=2, pool='mean',
                 dropout=0.2, no_wordvec_layer=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.pool = pool
        self.no_wordvec_layer = no_wordvec_layer
        self.device = device
        self.embs = ibp_utils.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.lstm = ibp_utils.LSTM(word_vec_size, hidden_size, bidirectional=True)
        else:
            self.linear_input = ibp_utils.Linear(word_vec_size, hidden_size)
            self.lstm = ibp_utils.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.dropout = ibp_utils.Dropout(dropout)
        self.fc_hidden = ibp_utils.Linear(2 * hidden_size, hidden_size)
        self.fc_output = ibp_utils.Linear(hidden_size, num_labels)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0):
        """
        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        if type(batch) != tuple:
            B = len(batch)
            x = batch.view(B, -1, 1)
            mask = batch != 1
            mask = mask.long()
            lengths = mask.sum(dim=1)

        else:
            if compute_bounds:
                x = batch[0]
            else:
                x = batch[0].val
            mask = batch[1]
            lengths = batch[2]
            B = x.shape[0]

        x_vecs = self.embs(x)  # B, n, d
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp_utils.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer:
            z = x_vecs
        else:
            z = ibp_utils.activation(F.relu, x_vecs)  # B, n, h
        h0 = torch.zeros((B, 2 * self.hidden_size), device=self.device)  # B, 2*h
        c0 = torch.zeros((B, 2 * self.hidden_size), device=self.device)  # B, 2*h
        h_mat, c_mat = self.lstm(z, (h0, c0), mask=mask)  # B, n, 2*h each
        h_masked = h_mat * mask.unsqueeze(2)
        if self.pool == 'mean':
            fc_in = ibp_utils.sum(h_masked / lengths.to(dtype=torch.float).view(-1, 1, 1), 1)  # B, 2*h
        else:
            raise NotImplementedError()
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp_utils.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, num_labels

        return output


## if using BERT, just switch to BertModel4Mix in the MixText model
## and change name to self.bert

class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.output_attentions = False
        self.output_hidden_states = True
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None,
                hidden_states2=None, attention_mask2=None,
                l=None, mix_layer=1000, head_mask=None):
        all_hidden_states = () if self.output_hidden_states else None
        all_attentions = () if self.output_attentions else None

        # Perform mix till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1 - l) * hidden_states2

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i <= mix_layer:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            elif i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1 - l) * hidden_states2
                    attention_mask = attention_mask.long() | attention_mask2.long()
                    ## sentMix: (bsz, len, hid)
                    # hidden_states[:, 0, :] = l * hidden_states[:, 0, :] + (1-l)*hidden_states2[:, 0, :]
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print (len(outputs))
        # print (len(outputs[1])) ##hidden states: 13
        return outputs


class BertModel4Mix(BertPreTrainedModel, nn.Module):

    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    # def _resize_token_embeddings(self, new_num_tokens):
    #     old_embeddings = self.embeddings.word_embeddings
    #     new_embeddings = self._get_resized_embeddings(
    #         old_embeddings, new_num_tokens)
    #     self.embeddings.word_embeddings = new_embeddings
    #     return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids, attention_mask, token_type_ids,
                input_ids2=None, attention_mask2=None, token_type_ids2=None,
                l=None, mix_layer=1000, head_mask=None):

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0
            embedding_output2 = self.embeddings(input_ids2, token_type_ids=token_type_ids2)
            encoder_outputs = self.encoder(embedding_output, extended_attention_mask,
                                           embedding_output2, extended_attention_mask2,
                                           l, mix_layer, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output, embedding_output) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class MixText(BertPreTrainedModel, nn.Module):
    def __init__(self, config):
        super(MixText, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel4Mix(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids,
                input_ids2=None, attention_mask2=None, token_type_ids2=None,
                l=None, mix_layer=1000):

        if input_ids2 is not None:
            outputs = self.bert(input_ids, attention_mask, token_type_ids,
                                input_ids2, attention_mask2, token_type_ids2,
                                l, mix_layer)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        else:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)

        return logits, outputs


class ASCCModel(BertPreTrainedModel, nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_embeddings = self.get_input_embeddings().num_embeddings

        self.init_weights()

    def build_nbrs(self, nbr_file, vocab, alpha, num_steps):
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        loaded = json.load(open(nbr_file))
        filtered = defaultdict(lambda: [], {})
        for k in loaded:
            if k in t2i:
                for v in loaded[k]:
                    if v in t2i:
                        filtered[k].append(v)
        nbrs = dict(filtered)

        nbr_matrix = []
        vocab_size = vocab.get_vocab_size("tokens")
        for idx in range(vocab_size):
            token = vocab.get_token_from_index(idx)
            nbr = [idx]
            if token in nbrs.keys():
                words = nbrs[token]
                for w in words:
                    assert w in t2i
                    nbr.append(t2i[w])
            nbr_matrix.append(nbr)
        nbr_matrix = batch_pad(nbr_matrix)
        self.nbrs = torch.tensor(nbr_matrix).cuda()
        self.max_nbr_num = self.nbrs.size()[-1]
        # self.weighting_param = nn.Parameter(torch.empty([self.num_embeddings, self.max_nbr_num], dtype=torch.float32),
        #                                     requires_grad=True).cuda()
        self.weighting_mask = self.nbrs != 0
        self.criterion_kl = nn.KLDivLoss(reduction="sum")
        self.alpha = alpha
        self.num_steps = num_steps

    def forward(self, input_ids, attention_mask, token_type_ids):
        # clean_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # pooled_clean_output = self.dropout(clean_outputs[1])
        # clean_logits = self.classifier(pooled_clean_output)
        # return clean_logits, clean_logits

        # 0 initialize w for neighbor weightings
        batch_size, text_len = input_ids.shape
        w = torch.empty(batch_size, text_len, self.max_nbr_num, 1).to(self.device).to(torch.float)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()
        optimizer_w = torch.optim.Adam([w], lr=1, weight_decay=2e-5)

        # 1 forward and backward to calculate adv_examples
        input_nbr_embed = self.get_input_embeddings()(self.nbrs[input_ids])
        weighting_mask = self.weighting_mask[input_ids]
        # here we need to calculate clean logits with no grad, to find adv examples
        with torch.no_grad():
            clean_outputs = self.bert(input_ids, attention_mask, token_type_ids)
            pooled_clean_output = self.dropout(clean_outputs[1])
            clean_logits = self.classifier(pooled_clean_output)

        for _ in range(self.num_steps):
            optimizer_w.zero_grad()
            with torch.enable_grad():
                w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
                embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)
                adv_outputs = self.bert(attention_mask=attention_mask, token_type_ids=token_type_ids,
                                        inputs_embeds=embed_adv)
                pooled_adv_output = self.dropout(adv_outputs[1])
                adv_logits = self.classifier(pooled_adv_output)
                adv_loss = - self.criterion_kl(F.log_softmax(adv_logits, dim=1),
                                           F.softmax(clean_logits.detach(), dim=1))
                loss_sparse = (-F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1) * F.log_softmax(w_after_mask, -2)).sum(-2).mean()
                loss = adv_loss + self.alpha * loss_sparse

            loss.backward(retain_graph=True)
            optimizer_w.step()

        optimizer_w.zero_grad()
        self.zero_grad()

        # 2 calculate clean data logits
        clean_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_clean_output = self.dropout(clean_outputs[1])
        clean_logits = self.classifier(pooled_clean_output)

        # 3 calculate convex hull of each embedding
        w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
        embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)

        # 4 calculate adv logits
        adv_outputs = self.bert(attention_mask=attention_mask, token_type_ids=token_type_ids,
                                inputs_embeds=embed_adv)
        pooled_adv_output = self.dropout(adv_outputs[1])
        adv_logits = self.classifier(pooled_adv_output)


        return clean_logits, adv_logits
