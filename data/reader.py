import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing import List
from torch.utils.data import TensorDataset, DataLoader
from data.dataset import ListDataset
from data.instance import InputInstance
from utils.certified import ibp_utils
from utils.my_utils import collate_fn, xlnet_collate_fn, PooledBatchSampler, lstm_collate_fn
from utils.config import LABEL_MAP


class BaseReader(object):
    def __init__(self, model_type='bert', max_seq_len=128):
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.label_map = dict()

    # given file_name, return a instance of Dataset
    # (ListDataset if tokenizer not given, else TensorDataset)
    def read_from_file(self, file_path, split='train'):
        raise NotImplementedError

    def get_dataset(self, *args):
        raise NotImplementedError

    def get_dataset_loader(self, instances, tokenizer):
        raise NotImplementedError

    def label_num(self):
        return len(self.label_map)

    def get_labels(self):
        return self.label_map

    def _convert_instance_to_dataset(self,
                                     instances: List[InputInstance],
                                     tokenizer=None,
                                     mask_padding_with_zero: bool = True,
                                     use_tqdm=True):
        # no tokenizer => return raw texts
        if tokenizer is None:
            return ListDataset(instances)

        pad_on_left = bool(self.model_type in ['xlnet'])
        pad_token_segment_id = 0 if self.model_type not in ['xlnet'] else 4
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

        all_ids = []
        if use_tqdm:
            iterator = tqdm(instances)
        else:
            iterator = instances

        for instance in iterator:
            inputs = tokenizer.encode_plus(
                instance.text_a,
                instance.text_b,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_len
            )
            input_ids = inputs["input_ids"]
            if "token_type_ids" in inputs:
                token_type_ids = inputs["token_type_ids"]
            else:
                token_type_ids = tokenizer.create_token_type_ids_from_sequences(input_ids[1:-1])

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_len - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token_id] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            if instance.label is not None:
                if instance.label.isdigit():
                    label = int(instance.label)
                else:
                    label = LABEL_MAP['nli'][instance.label]
                # label = self.label_map[instance.label]
            else:
                label = None

            all_ids.append((input_ids, attention_mask, token_type_ids, label, input_len))

        # convert ids to TensorDataset
        all_input_ids = torch.tensor([f[0] for f in all_ids], dtype=torch.long)
        all_attention_mask = torch.tensor([f[1] for f in all_ids], dtype=torch.long)
        all_token_type_ids = torch.tensor([f[2] for f in all_ids], dtype=torch.long)
        all_lens = torch.tensor([f[4] for f in all_ids], dtype=torch.long)

        none_labels = any([True if f[3] is None else False for f in all_ids])
        if none_labels:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens)
        else:
            all_labels = torch.tensor([f[3] for f in all_ids], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)

        return dataset

    def saving_instances(self, instances: List[InputInstance], dataset_dir, file_name):
        with open(f'{dataset_dir}/{file_name}.txt', 'w', encoding='utf-8') as fout:
            for instance in instances:
                if instance.is_nli():
                    fout.write(f'{instance.label}\t{instance.text_a}\t{instance.text_b}\n')
                else:
                    fout.write(f'{instance.text_a}\t{instance.label}\n')


class ClassificationReader(BaseReader):
    def __init__(self, model_type='bert', max_seq_len=128):
        super().__init__(model_type, max_seq_len)
        self.model_type = model_type
        self.max_seq_len = max_seq_len

    def read_from_file(self, file_path, split='train'):
        temp_label_set = set()
        instances = []
        with open(f'{file_path}/{split}.txt', 'r', encoding='utf-8') as fin:
            count = 0
            for line in fin:
                ss = line.strip().split('\t')
                idx = f'{split}-{count}'
                if len(ss) == 3:
                    temp_label_set.add(ss[0])
                    instances.append(InputInstance(idx, ss[1], ss[2], ss[0]))
                elif len(ss) == 2:
                    temp_label_set.add(ss[1])
                    instances.append(InputInstance(idx, ss[0], None, ss[1]))
                count += 1

        # first construct label map
        for (i, x) in enumerate(sorted(temp_label_set)):
            self.label_map[x] = i

        return instances

    def get_dataset(self, instances, tokenizer):
        return self._convert_instance_to_dataset(instances, tokenizer)

    def get_dataset_loader(self, dataset, tokenized=True,
                           batch_size=32, shuffle=False):
        # for collate function
        if tokenized:
            collate_function = xlnet_collate_fn if self.model_type in ['xlnet'] else collate_fn
        else:
            collate_function = lambda x: x
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_function)

        return data_loader


class ClassificationReaderSpacy(BaseReader):
    def __init__(self, model_type='lstm', max_seq_len=128):
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.label_map = dict()

    def read_from_file(self, file_path, split='train'):
        temp_label_set = set()
        instances = []
        with open(f'{file_path}/{split}.txt', 'r', encoding='utf-8') as fin:
            count = 0
            for line in fin:
                ss = line.strip().split('\t')
                idx = f'{split}-{count}'
                if len(ss) == 3:
                    temp_label_set.add(ss[0])
                    instances.append(InputInstance(idx, ss[1], ss[2], ss[0]))
                elif len(ss) == 2:
                    temp_label_set.add(ss[1])
                    instances.append(InputInstance(idx, ss[0], None, ss[1]))
                count += 1

        # first construct label map
        for (i, x) in enumerate(sorted(temp_label_set)):
            self.label_map[x] = i

        return instances

    def from_raw_data(self, instances, vocab, attack_surface=None) -> ListDataset:
        examples = []
        for instance in instances:
            all_words = [w.lower() for w in instance.text_a.split()]
            if instance.text_b is not None:
                all_words = all_words + [w.lower() for w in instance.text_b.split()]
            if attack_surface:
                all_swaps = attack_surface.get_swaps(all_words)
                words = [w for w in all_words if w in vocab]
                swaps = [s for w, s in zip(all_words, all_swaps) if w in vocab]
                choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
            else:
                words = [w for w in all_words if w in vocab]  # Delete UNK words

            words = words[:self.max_seq_len]

            word_idxs = [vocab.get_index(w) for w in words]
            x_torch = torch.tensor(word_idxs).view(1, -1, 1)  # (1, T, d)
            if attack_surface:
                choices_word_idxs = list()
                for c_list in choices:
                    temp_list = list()
                    for c in c_list:
                        if vocab.get_index(c) != 0:
                            temp_list.append(vocab.get_index(c))
                    choices_word_idxs.append(torch.tensor(temp_list, dtype=torch.long))
                # choices_word_idxs = [
                #     torch.tensor([vocab.get_index(c) for c in c_list], dtype=torch.long) for c_list in choices
                # ]
                if any(0 in c.view(-1).tolist() for c in choices_word_idxs):
                    print(choices_word_idxs)
                    raise ValueError("UNK tokens found")
                choices_torch = pad_sequence(choices_word_idxs, batch_first=True).unsqueeze(2).unsqueeze(
                    0)  # (1, T, C, 1)
                choices_mask = (choices_torch.squeeze(-1) != 0).long()  # (1, T, C)
            else:
                choices_torch = x_torch.view(1, -1, 1, 1)  # (1, T, 1, 1)
                choices_mask = torch.ones_like(x_torch.view(1, -1, 1))
            mask_torch = torch.ones((1, len(word_idxs)))
            x_bounded = ibp_utils.DiscreteChoiceTensor(x_torch, choices_torch, choices_mask, mask_torch)
            y_torch = torch.tensor(self.label_map[instance.label], dtype=torch.long).view(1, 1)
            lengths_torch = torch.tensor(len(word_idxs)).view(1)
            examples.append(dict(x=x_bounded, y=y_torch, mask=mask_torch, lengths=lengths_torch))
        dataset = ListDataset(examples)
        return dataset

    def get_dataset(self, instances, vocab, attack_surface=None):
        return self.from_raw_data(instances, vocab, attack_surface)

    def get_dataset_loader(self, dataset, vocab=None, batch_size=32):
        batch_sampler = PooledBatchSampler(dataset, batch_size, sort_key=self.example_len)
        return DataLoader(dataset, pin_memory=True, collate_fn=lstm_collate_fn, batch_sampler=batch_sampler)

    @staticmethod
    def example_len(example):
        return example['x'].shape[1]

    def get_word_set(self, instances, counter_fitted_file, attack_surface):
        with open(counter_fitted_file) as f:
            counter_vocab = set([line.split(' ')[0] for line in f])
        word_set = set()
        for instance in instances:
            words = [w.lower() for w in instance.text_a.split(' ')]
            if instance.text_b is not None:
                words = words + [w.lower() for w in instance.text_b.split(' ')]
            for w in words:
                word_set.add(w)
            try:
                swaps = attack_surface.get_swaps(words)
                for cur_swaps in swaps:
                    for w in cur_swaps:
                        word_set.add(w)
            except KeyError:
                # For now, ignore things not in attack surface
                # If we really need them, later code will throw an error
                pass
        return word_set & counter_vocab

