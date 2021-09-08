import csv
from typing import Union, Callable, Dict
import json
from collections import defaultdict, Counter
from functools import lru_cache
from .searcher import Searcher
import numpy as np


class CachedWordSearcher(Searcher):
    """
        Load words from a json file
    """
    def __init__(
        self,
        file_name: str,
        vocab_list,
        second_order: bool = False,
    ):
        super().__init__()
        loaded = json.load(open(file_name))
        # filter by a given vocabulary
        if vocab_list:
            filtered = defaultdict(lambda: [], {})
            for k in loaded:
                if k in vocab_list:
                    for v in loaded[k]:
                        if v in vocab_list:
                            filtered[k].append(v)
            filtered = dict(filtered)
        else:
            filtered = loaded
        # add second order words
        if second_order:
            nbrs = defaultdict(lambda: [], {})
            for k in filtered:
                for v in filtered[k]:
                    nbrs[k].append(v)
                    # some neighbours have no neighbours
                    if v not in filtered:
                        continue
                    for vv in filtered[v]:
                        if vv != k and vv not in nbrs[k]:
                            nbrs[k].append(vv)
            nbrs = dict(nbrs)
        else:
            nbrs = filtered
        self.nbrs = nbrs
            
    def show_verbose(self):
        nbr_num = list(map(len, list(self.nbrs.values())))
        print(f"total word: {len(self.nbrs)}, ",
              f"mean: {round(np.mean(nbr_num), 2)}, ",
              f"median: {round(np.median(nbr_num), 2)}, "
              f"max: {np.max(nbr_num)}, ")
        print(Counter(nbr_num))

    def search(self, word):
        if word in self.nbrs:
            return self.nbrs[word]
        else:
            return []

