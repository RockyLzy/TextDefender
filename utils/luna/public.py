import functools
import warnings
import logging
import os
import time
from contextlib import contextmanager
import numpy as np
import re
import pickle
import random
import argparse
from typing import List, Dict, NamedTuple, Union, Iterable
from colorama import Fore, Back
import psutil
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support as sklearn_prf
import inspect
import arrow
import traceback
from sklearn.linear_model import LinearRegression
import torch
import logging

__saved_path__ = "saved/vars"

arg_required = object()
arg_optional = object()
arg_place_holder = object()


def shutdown_logging(repo_name):
    for key, logger in logging.root.manager.loggerDict.items():
        if isinstance(key, str) and key.startswith(repo_name):
            logging.getLogger(key).setLevel(logging.ERROR)


@contextmanager
def numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        

def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property


def deprecated(message: str = ''):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used first time and filter is set for show DeprecationWarning.
    """

    def decorator_wrapper(func):
        @functools.wraps(func)
        def function_wrapper(*args, **kwargs):
            current_call_source = '|'.join(
                traceback.format_stack(inspect.currentframe()))
            if current_call_source not in function_wrapper.last_call_source:
                warnings.warn("Function {} is now deprecated! {}".format(func.__name__, message),
                              category=DeprecationWarning, stacklevel=2)
                function_wrapper.last_call_source.add(current_call_source)

            return func(*args, **kwargs)

        function_wrapper.last_call_source = set()

        return function_wrapper

    return decorator_wrapper


def check_os(platform):
    if platform == 'win' and is_win():
        allow = True
    elif platform == 'unix' and is_unix():
        allow = True
    else:
        allow = False
    if allow:
        def inner(func):
            return func

        return inner
    else:
        raise Exception("only support {}".format(platform))


def is_win():
    return psutil.WINDOWS


def is_unix():
    return psutil.LINUX | psutil.MACOS


def time_stamp():
    return arrow.now().format('MMMDD_HH-mm-ss')


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


@check_os('unix')
def create_folder_for_file(file_path):
    splash_idx = file_path.rindex('/')
    create_folder(file_path[:splash_idx])


def show_mem():
    top = psutil.Process(os.getpid())
    info = top.memory_full_info()
    memory = info.uss / 1024. / 1024.
    print('Memory: {:.2f} MB'.format(memory))


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name,
                 var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def set_saved_path(path):
    global __saved_path__
    __saved_path__ = path


def save_var(variable, name, path=None):
    if path is None:
        path = __saved_path__
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    pickle.dump(variable, open("{}/{}.pkl".format(path, name), "wb"))


def load_var(name, path=None):
    if path is None:
        path = __saved_path__
    return pickle.load(open("{}/{}.pkl".format(path, name), "rb"))


def exist_var(name, path=None):
    if path is None:
        path = __saved_path__
    return os.path.exists("{}/{}.pkl".format(path, name))


def auto_create(name, func, cache=False, path=None):
    if path is None:
        path = __saved_path__
    if cache and exist_var(name, path):
        print("cache for {} exists".format(name))
        with time_record("*** load {} from cache".format(name)):
            obj = load_var(name, path)
    else:
        print("cache for {} does not exist".format(name))
        with time_record("*** create {} and save to cache".format(name)):
            obj = func()
            save_var(obj, name, path)
    return obj


@contextmanager
def time_record(sth=None):
    start = time.time()
    yield
    end = time.time()
    if sth:
        print(sth, "cost {:.3} seconds".format(end - start))
    else:
        print("cost {:.3} seconds".format(end - start))


def as_table(x):
    if isinstance(x, list):
        return tabulate([[retrieve_name(ele), ele] for ele in x])
    elif isinstance(x, dict):
        return tabulate([[k, v] for k, v in x.items()])


class ProgressManager:
    def __init__(self, total):
        self.__start = time.time()
        self.__prev_prev = time.time()
        self.__prev = time.time()
        self.__total = total
        self.__complete = 0

    def update(self, batch_num):
        self.__complete += batch_num
        self.__prev_prev = self.__prev
        self.__prev = time.time()

    @property
    def batch_time(self):
        return self.__prev - self.__prev_prev

    @property
    def cost_time(self):
        return self.__prev - self.__start

    @property
    def rest_time(self):
        return self.cost_time / self.__complete * (self.__total - self.__complete)

    @property
    def complete_num(self):
        return self.__complete

    @property
    def total_num(self):
        return self.__total


class DeprecatedDataSet:
    def __init__(self):
        self.data = []
        self.__next = 0

    def next_batch(self, batch_size,
                   fill_batch=True):
        if self.__next + batch_size > len(self.data):
            if fill_batch:
                ret = self.data[self.size - batch_size:self.size]
            else:
                ret = self.data[self.__next:self.size]
            self.__next = self.size
        else:
            ret = self.data[self.__next:self.__next + batch_size]
            self.__next += batch_size
        return ret

    @property
    def size(self):
        return len(self.data)

    @property
    def finished(self):
        return self.__next == self.size

    def reset(self, shuffle=True):
        self.__next = 0
        if shuffle:
            random.shuffle(self.data)


class ArgParser:
    def __init__(self):
        self.ap = argparse.ArgumentParser()

    def request(self, key, value):
        self.ap.add_argument('-{}'.format(key),
                             action='store',
                             default=value,
                             type=type(value),
                             dest=str(key))

    def parse(self):
        return self.ap.parse_args()


def hit(scores: List[List], gold: List, k: int):
    corr = 0
    total = len(gold)
    for score, label in zip(scores, gold):
        if label in list(reversed(np.argsort(score)))[:k]:
            corr += 1
    return corr / total


def get_prf(pred: List, gold: List) -> (List, List, List):
    precision, recall, f1, _ = sklearn_prf(gold, pred, beta=1, average=None)
    return precision.tolist(), recall.tolist(), f1.tolist()


# def show_prf(pred: List, gold: List, classes):
#     precision, recall, f1, _ = sklearn_prf(gold, pred, beta=1, labels=classes)
#     head = "{:4}|{:15}|{:10}|{:10}|{:10}"
#     content = "{:4}|{:15}|{:10f}|{:10f}|{:10f}"
#     print(Color.cyan(head.format("ID", "Class", "Precision", "Recall", "F1")))
#     for i in range(len(classes)):
#         print(Color.white(content.format(i, classes[i], precision[i], recall[i], f1[i])))


def score2rank(scores) -> list:
    return np.argmax(scores, 1).tolist()


def accuracy(scores: List[List], gold: List):
    return hit(scores, gold, 1)


def locate_chunk(num_total,  num_chunk, chunk_id):
    start = num_total // num_chunk * chunk_id
    end = num_total // num_chunk * (chunk_id + 1)
    if chunk_id == num_chunk - 1:
        end = num_total
    return start, end


def chunks(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


class CherryPicker:
    def __init__(self, lower_is_better, compare_fn=None):
        self.lower_is_better = lower_is_better
        self.history_values = []
        self.history_infos = []
        self.compare_fn = compare_fn

    def add(self, value, info):
        self.history_infos.append(info)
        self.history_values.append(value)

    @property
    def size(self):
        return len(self.history_values)

    def select_best_point(self):
        if self.size == 0:
            raise Exception("Nothing to pick.")
        # np.argmin selects the first occurrence of the min
        if self.compare_fn is None:
            if self.lower_is_better:
                chosen_id = int(np.argmin(self.history_values))
            else:
                chosen_id = int(np.argmax(self.history_values))
        else:
            chosen_id = len(self.history_values) - 1
            chosen_val = self.history_values[-1]
            for i in reversed(range(len(self.history_values))):
                if self.lower_is_better:
                    if self.compare_fn(self.history_values[i], chosen_val) <= 0:
                        chosen_id = i
                        chosen_val = self.history_values[chosen_id]
                else:
                    if self.compare_fn(self.history_values[i], chosen_val) >= 0:
                        chosen_id = i
                        chosen_val = self.history_values[chosen_id]
        return chosen_id, self.history_values[chosen_id], self.history_infos[chosen_id]


class TrainingStopObserver:
    def __init__(self,
                 lower_is_better,
                 can_stop_val=None,
                 must_stop_val=None,
                 min_epoch=None,
                 max_epoch=None,
                 epoch_num=None
                 ):
        self.history_values = []
        self.history_infos = []
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.epoch_num = epoch_num
        self.lower_is_better = lower_is_better
        self.can_stop_val = can_stop_val
        self.must_stop_val = must_stop_val

    def check_stop(self, value, info=None) -> bool:
        self.history_values.append(value)
        self.history_infos.append(info)
        if self.can_stop_val is not None:
            if self.lower_is_better and value > self.can_stop_val:
                return False
            if not self.lower_is_better and value < self.can_stop_val:
                return False
        if self.must_stop_val is not None:
            if self.lower_is_better and value < self.must_stop_val:
                return True
            if not self.lower_is_better and value > self.must_stop_val:
                return True
        if self.max_epoch is not None and len(self.history_values) > self.max_epoch:
            return True
        if self.min_epoch is not None and len(self.history_values) <= self.min_epoch:
            return False
        lower = value < np.mean(self.history_values[-(self.epoch_num + 1):-1])
        if self.lower_is_better:
            return False if lower else True
        else:
            return True if lower else False

    def select_best_point(self):
        if self.lower_is_better:
            chosen_id = int(np.argmin(self.history_values[self.min_epoch:]))
        else:
            chosen_id = int(np.argmax(self.history_values[self.min_epoch:]))
        return self.history_values[self.min_epoch + chosen_id], self.history_infos[self.min_epoch + chosen_id]


def cast_item(array):
    if isinstance(array, np.ndarray):
        array = array.tolist()
    while True:
        if isinstance(array, list):
            if len(array) != 1:
                raise Exception("More than one item!")
            array = array[0]
        else:
            break
    return array


# def cast_list(array):
#     if isinstance(array, list):
#         return cast_list(np.array(array))
#     if isinstance(array, np.ndarray):
#         return array.squeeze().tolist()


class Aggregator:
    """
    Usage:
        You may use an Aggregator to aggregate values any where,
        and reduce them through any way you like. The tool prevent you from
        writing many dirty code to track values in different places/iterations.
        Without an Aggregator:
            key1_list, key2_list, key3_list = [], [], []
            # In an iteration, you collect them as:
                key1_list.append(1)
                key2_list.append(2)
                key3_list.append(5)
            # while in another iteration,
                key1_list.append(2)
                key2_list.append(2)
                key3_list.append(4)
            # ...
        With an Aggregator:
            agg = Aggregator()
            # In an iteration, you collect them as:
                agg.aggregate((key1, 1), (key2, 2), (key3, 5) ...)
            # while in another iteration,
                agg.aggregate((key1, 3), (key2, 2), (key3, 5) ...)
                agg.aggregate((key1, 5), (key2, 4), (key3, 5) ...)
            ...
        And finally, you can reduce the values:
            agg.aggregated(key1)  --> [1, 3, 5]
            agg.aggregated(key1, 'mean')  --> 3
            agg.aggregated(key1, np.sum)  --> 9
            agg.mean(key1)  --> 3
    """

    def __init__(self):
        self.__kv_mode = False
        self.__keys = None
        self.__saved = None

    @property
    def size(self):
        return len(self.__saved[0])

    def has_key(self, key):
        if self.__keys is None:
            return False
        if not self.__kv_mode:
            return False
        return key in self.__keys

    def aggregate(self, *args):
        # First called, init the collector and decide the key mode
        if self.__saved is None:
            if Aggregator.__args_kv_mode(*args):
                self.__kv_mode = True
                self.__keys = list(map(lambda x: x[0], args))
            # else:
            #     self.keys = ['__{}' for i in range(len(args))]
            self.__saved = [[] for _ in range(len(args))]
        # Later called
        if Aggregator.__args_kv_mode(*args) != self.__kv_mode:
            raise Exception("you must always specify a key or not")
        for i in range(len(args)):
            if self.__kv_mode:
                saved_id = self.__keys.index(args[i][0])
                to_save = args[i][1]
            else:
                saved_id = i
                to_save = args[i]
            if isinstance(to_save, list):
                self.__saved[saved_id].extend(to_save)
            else:
                self.__saved[saved_id].append(to_save)

    @staticmethod
    def __args_kv_mode(*args):
        # print("args is {}".format(args))
        has_key_num = 0
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], str):
                has_key_num += 1
        if has_key_num == len(args):
            return True
        if has_key_num == 0:
            return False
        raise Exception("you must specify a key for all args or not")

    def mean(self, key):
        return self.aggregated(key, 'mean')

    def std(self, key):
        return self.aggregated(key, 'std')

    def sum(self, key):
        return self.aggregated(key, 'sum')

    def list(self, key):
        return self.aggregated(key)

    def aggregated(self, key=None, reduce: Union[str, callable] = 'no'):
        if reduce == 'no':
            def reduce_fn(x): return x
        elif reduce == 'mean':
            reduce_fn = np.mean
        elif reduce == 'sum':
            reduce_fn = np.sum
        elif reduce == 'std':
            reduce_fn = np.std
        elif inspect.isfunction(reduce):
            reduce_fn = reduce
        else:
            raise Exception(
                'reduce must be None, mean, sum, std or a function.')

        if key is None:
            if not self.__kv_mode:
                if len(self.__saved) == 1:
                    return reduce_fn(self.__saved[0])
                else:
                    return tuple(reduce_fn(x) for x in self.__saved)
            else:
                raise Exception("you must specify a key")
        elif key is not None:
            if self.__kv_mode:
                saved_id = self.__keys.index(key)
                return reduce_fn(self.__saved[saved_id])
            else:
                raise Exception("you cannot specify a key")


def analyze_length_count(length_count: dict):
    sorted_count = sorted(length_count.items(), key=lambda kv: kv[0])
    print("Analyze length count:")
    print("\tTotal Count:", *sorted_count)
    pivots = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.01]
    agg_num = []
    tmp_num = 0
    for k, v in sorted_count:
        tmp_num += v
        agg_num.append(tmp_num)
    print("\tTotal num: ", tmp_num)
    agg_ratio = list(map(lambda x: x / tmp_num, agg_num))
    print("\tRatio: ")
    for pivot in pivots:
        idx = sum(list(map(lambda x: x < pivot, agg_ratio))) - 1
        print("\t\t{} : {}".format(
            pivot, "-" if idx == -1 else sorted_count[idx][0]))


def analyze_vocab_count(vocab_count: dict):
    pivots = [0, 1, 2, 3, 4, 5, 10, 20, 30]
    vocab_size = []
    count = []
    for pivot in pivots:
        tmp = list(filter(lambda pair: pair[1] > pivot, vocab_count.items()))
        vocab_size.append(len(tmp))
        count.append(sum(map(lambda pair: pair[1], tmp)))
    print("Analyze vocab count:")
    print("\tTotal vocab size {}, count: {}".format(vocab_size[0], count[0]))
    print("\tRatio: ")
    for cid in range(len(pivots)):
        print("\t\t> {}: vocab size {}/{:.3f}, count {}/{:.3f}".format(
            pivots[cid],
            vocab_size[cid], vocab_size[cid] / vocab_size[0],
            count[cid], count[cid] / count[0]
        ))


def group_fields(lst: List[object],
                 keys: Union[str, List[str]] = None,
                 indices: Union[int, List[int]] = None):
    assert keys is None or indices is None
    is_single = False
    if keys:
        if not isinstance(keys, list):
            keys = [keys]
            is_single = True
        indices = []
        for key in keys:
            obj_type = type(lst[0])
            idx = obj_type._fields.index(key)
            indices.append(idx)
    else:
        if not isinstance(indices, list):
            indices = [indices]
            is_single = True
    rets = []
    for idx in indices:
        rets.append(list(map(lambda item: item[idx], lst)))
    if is_single:
        return rets[0]
    else:
        return rets


def show_num_list(lst):
    if isinstance(lst[0], float):
        for ele in lst:
            print("{:5.2f}".format(ele), end=' ')
    else:
        for ele in lst:
            print("{:5}".format(ele), end=' ')
    print()
