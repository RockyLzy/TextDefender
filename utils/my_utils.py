import torch
from typing import Tuple, Set
from utils.certified import data_util, ibp_utils
import random
from torch.utils.data import BatchSampler


class PooledBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, sort_within_batch=True, sort_key=len):
        self.dataset_lens = [sort_key(el) for el in dataset]
        self.batch_size = batch_size
        self.sort_within_batch = sort_within_batch

    def __iter__(self):
        """
        1- Partitions data indices into chunks of batch_size * 100
        2- Sorts each chunk by the sort_key
        3- Batches sorted chunks sequentially
        4- Shuffles the batches
        5- Yields each batch
        """
        idx_chunks = torch.split(torch.randperm(len(self.dataset_lens)), self.batch_size * 100)
        for idx_chunk in idx_chunks:
            sorted_chunk = torch.tensor(sorted(idx_chunk.tolist(), key=lambda idx: self.dataset_lens[idx]))
            chunk_batches = [chunk.tolist() for chunk in torch.split(sorted_chunk, self.batch_size)]
            random.shuffle(chunk_batches)
            for batch in chunk_batches:
                if self.sort_within_batch:
                    batch.reverse()
                yield batch

    def __len__(self):
        return (len(self.dataset_lens) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    if len(batch[0]) == 4:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens = map(torch.stack, zip(*batch))
    else:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))

    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    if len(batch[0]) == 4:
        return all_input_ids, all_attention_mask, all_token_type_ids
    else:
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def xlnet_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    if len(batch[0]) == 4:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens = map(torch.stack, zip(*batch))
    else:
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, -max_len:]
    all_attention_mask = all_attention_mask[:, -max_len:]
    all_token_type_ids = all_token_type_ids[:, -max_len:]
    if len(batch[0]) == 4:
        return all_input_ids, all_attention_mask, all_token_type_ids
    else:
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def lstm_collate_fn(examples):
    """
        Turns a list of examples into a workable batch:
    """
    if len(examples) == 1:
        return examples[0]
    B = len(examples)
    max_len = max(ex['x'].shape[1] for ex in examples)
    x_vals = []
    choice_mats = []
    choice_masks = []
    y = torch.zeros((B, 1))
    lengths = torch.zeros((B,), dtype=torch.long)
    masks = torch.zeros((B, max_len))
    for i, ex in enumerate(examples):
        x_vals.append(ex['x'].val)
        choice_mats.append(ex['x'].choice_mat)
        choice_masks.append(ex['x'].choice_mask)
        cur_len = ex['x'].shape[1]
        masks[i, :cur_len] = 1
        y[i, 0] = ex['y']
        lengths[i] = ex['lengths'][0]
    x_vals = data_util.multi_dim_padded_cat(x_vals, 0).long()
    choice_mats = data_util.multi_dim_padded_cat(choice_mats, 0).long()
    choice_masks = data_util.multi_dim_padded_cat(choice_masks, 0).long()
    return ibp_utils.DiscreteChoiceTensor(x_vals, choice_mats, choice_masks, masks), masks, lengths, y

def convert_dataset_to_batch(dataset, model_type: str):
    batch = tuple(zip(*dataset.tensors))
    if model_type in ['xlnet']:
        return xlnet_collate_fn(batch)
    else:
        return collate_fn(batch)

# batch
def convert_batch_to_bert_input_dict(batch: Tuple[torch.Tensor] = None, model_type: str = None):
    '''
    :param model_type: model type for example, 'bert'
    :param batch: tuple, contains 3 element, batch[0]: embedding
            batch[1]:attention_mask, batch[2]: token_type_ids
    :return:
    '''

    def prepare_token_type_ids(type_ids: torch.Tensor, model_type: str) -> torch.Tensor:
        if model_type in ['bert', 'xlnet', 'albert', 'roberta']:
            return type_ids
        else:
            return None

    inputs = {}
    if len(batch[0].shape) == 3:
        inputs['inputs_embeds'] = batch[0]
    else:
        inputs['input_ids'] = batch[0]
    inputs['attention_mask'] = batch[1]
    # for distilbert and dcnn, token_type_ids is unnecessary
    if model_type != 'distilbert' and model_type != 'dcnn':
        inputs['token_type_ids'] = prepare_token_type_ids(batch[2], model_type)
    return inputs


# return type: set, to improve the search complexity. python set: O(1)
def build_forbidden_mask_words(file_path: str) -> Set[str]:
    sentiment_words_set = set()
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            sentiment_words_set.add(line.strip())
    return sentiment_words_set
