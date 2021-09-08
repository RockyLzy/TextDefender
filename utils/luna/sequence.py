from typing import List
import numpy as np
import random
import copy


def dump_count(dct: dict, file_path, value_decreasing=True):
    assert not isinstance(dct, Vocab)
    file = open(file_path, "w", encoding="utf8")
    for ele in sorted(dct.items(), key=lambda x: x[1], reverse=value_decreasing):
        print("{} {}".format(ele[0], ele[1]), file=file)
    file.close()


def count_token(file_path, verbose=False):
    file = open(file_path, encoding="utf8")
    _count = {}
    while True:
        line = file.readline()
        if line == '':
            break
        arr = line[:-1]
        if arr is not None:
            arr = arr.split(' ')
        for ele in arr:
            if ele in _count:
                _count[ele] += 1
            else:
                _count[ele] = 1
    if verbose:
        total = sum(map(lambda item: item[1], _count.items()))
        print("Total count: {}".format(len(_count)))
        num_covered = 0
        stones = [0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
        stone = stones.pop(0)
        k = 0
        for ele in sorted(_count.items(), key=lambda x: x[1], reverse=True):
            k += 1
            num_covered += ele[1]
            if num_covered / total >= stone:
                print("Top {:6} covers {}".format(k, stone))
                if len(stones) == 0:
                    break
                stone = stones.pop(0)
    return _count


def merge_count(*counts):
    tmp_count = {}
    for count in counts:
        for ele in count:
            if ele in tmp_count:
                tmp_count[ele] = tmp_count[ele] + count[ele]
            else:
                tmp_count[ele] = count[ele]
    return tmp_count


def build_vocab_from_count(count, topk=None):
    _vocab = {
        Vocab.bos_token: Vocab.bos_index,
        Vocab.pad_token: Vocab.pad_index,
        Vocab.eos_token: Vocab.eos_index,
        Vocab.unk_token: Vocab.unk_index
    }
    for ele in sorted(count.items(), key=lambda x: x[1], reverse=True):
        _vocab[ele[0]] = len(_vocab)
        if topk is not None and len(_vocab) > topk:
            break
    _rev_vocab = {item[1]: item[0] for item in _vocab.items()}
    return Vocab(_vocab, _rev_vocab)


def build_vocab_from_file(*file_paths):
    counts = []
    for file_path in file_paths:
        counts.append(count_token(file_path))
    vocab = build_vocab_from_count(merge_count(*counts))
    return vocab


def load_vocab_from_count_file(path):
    file = open(path, encoding="utf8")
    _count = {}
    while True:
        line = file.readline()
        if line == '':
            break
        arr = line[:-1]
        if arr is not None:
            arr = arr.split(' ')
        _count[arr[0]] = int(arr[1])
    vocab = build_vocab_from_count(_count)
    return vocab


class Vocab:
    # Keep consistency with fairseq
    bos_token, bos_index = "<bos>", 0
    pad_token, pad_index = "<pad>", 1
    eos_token, eos_index = "<eos>", 2
    unk_token, unk_index = "<unk>", 3

    def __init__(self, t2i_dct, i2t_dct):
        self.__t2i_dct = t2i_dct
        self.__i2t_dct = i2t_dct

    def seq2idx(self, seq) -> list:
        return list(map(lambda x: self.__t2i_dct[x] if x in self.__t2i_dct else self.unk_index, seq))

    def idx2seq(self, idx: list, bpe=None):
        if self.pad_index in idx:
            idx = idx[:idx.index(self.pad_index)]
        if self.eos_index in idx:
            idx = idx[:idx.index(self.eos_index)]
        ret = " ".join(list(map(lambda x: self.__i2t_dct[x], idx)))
        if bpe:
            return ret.replace("{} ".format(bpe), "")
        else:
            return ret

    def perplexity(self, idx: list, log_prob: list) -> float:
        if self.eos_index in idx:
            log_prob = log_prob[:idx.index(self.eos_index)]
        print(idx)
        N = len(log_prob)
        return np.exp(-1 / (N + 0.001) * np.sum(log_prob))

    def __getitem__(self, word):
        if word in self.__t2i_dct:
            return self.__t2i_dct[word]
        else:
            return self.unk_index

    def idx2word(self, idx):
        return self.__i2t_dct[idx]

    def convert_file_to_index(self, token_path, index_path,
                              add_bos=False,
                              add_eos=False):
        print("convert file {} to {}".format(token_path, index_path))
        lens = []
        token_file = open(token_path, encoding="utf8")
        index_file = open(index_path, "w", encoding="utf8")
        process = 0
        while True:
            raw_post = token_file.readline()
            if raw_post == '':
                break
            arr = raw_post[:-1]
            arr = arr.split(' ')
            if add_bos:
                arr.insert(0, Vocab.bos_token)
            if add_eos:
                arr.append(Vocab.eos_token)
            print(" ".join(map(str, self.seq2idx(arr))), file=index_file)
            process += 1
            lens.append(len(arr))
            # print("Process sentence {} in {}".format(process, token_path))
        return lens

    def __len__(self):
        return len(self.__t2i_dct)

    @property
    def t2i_dct(self):
        return self.__t2i_dct

    @property
    def i2t_dct(self):
        return self.__i2t_dct

    def pad(self):
        return self.pad_index

    def bos(self):
        return self.bos_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index


def lst2str(lst: list) -> str:
    return ''.join(lst)


def random_drop(idx: List, drop_rate) -> List:
    assert 0.0 < drop_rate < 0.5
    ret = list(filter(lambda x: x is not None,
                      map(lambda x: None if random.random() < drop_rate else x, idx)))
    if len(ret) == 0:
        return ret
    return ret


def __shuffle_slice(lst: List, start: int, stop: int):
    cp_lst = copy.copy(lst)
    # Fisher Yates Shuffle
    for i in range(start, stop):
        idx = random.randrange(i, stop)
        cp_lst[i], cp_lst[idx] = cp_lst[idx], cp_lst[i]
        i += 1
    return cp_lst


def random_shuffle_slice(lst: List, width: int) -> List:
    start = random.randrange(0, len(lst))
    stop = min(start + width, len(lst))
    return __shuffle_slice(lst, start, stop)


def batch_random_shuffle_slice(idx: List[List], width: int) -> List[List]:
    return list(map(lambda x: random_shuffle_slice(x, width), idx))


def batch_drop(idx: List[List], drop_rate) -> List[List]:
    return list(map(lambda x: random_drop(x, drop_rate), idx))


def batch_pad(idx: List[List], pad_ele=0, pad_len=None) -> List[List]:
    if pad_len is None:
        pad_len = max(map(len, idx))
    return list(map(lambda x: x + [pad_ele] * (pad_len - len(x)), idx))


def batch_mask(idx: List[List], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(map(len, idx))
    return list(map(lambda x: [good_ele] * len(x) + [mask_ele] * (max_len - len(x)), idx))


def batch_mask_by_len(lens: List[int], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(lens)
    return list(map(lambda x: [good_ele] * x + [mask_ele] * (max_len - x), lens))


def batch_append(idx: List[List], append_ele, before=False) -> List[List]:
    if not before:
        return list(map(lambda x: x + [append_ele], idx))
    else:
        return list(map(lambda x: [append_ele] + x, idx))


def batch_lens(idx: List[List]) -> List:
    return list(map(len, idx))


def as_batch(idx: List) -> List[List]:
    return [idx]


def flatten_lst(lst: List[List]) -> List:
    return [i for sub_lst in lst for i in sub_lst]
