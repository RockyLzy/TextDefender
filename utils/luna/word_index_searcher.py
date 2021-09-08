from typing import Union, Callable, Dict
from functools import lru_cache
from .searcher import Searcher


class WordIndexSearcher(Searcher):
    def __init__(
        self, 
        word_searcher, 
        word2idx: Union[Callable, Dict],
        idx2word: Union[Callable, Dict]
    ):
        self._word_searcher = word_searcher
        if isinstance(word2idx, dict):
            self.word2idx = word2idx.__getitem__
        else:
            self.word2idx = word2idx
        if isinstance(idx2word, dict):
            self.idx2word = idx2word.__getitem__
        else:
            self.idx2word = idx2word

    @lru_cache(maxsize=None)
    def search(self, idx):
        words = self._word_searcher.search(self.idx2word(idx))
        idxes = [self.word2idx(ele) for ele in words]
        assert 0 not in idxes, "Something must be wrong"
        return idxes

