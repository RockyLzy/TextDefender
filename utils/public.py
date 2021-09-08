import os
import time
import pickle
import hashlib
import torch
import random
import numpy as np
import logging
from contextlib import contextmanager
__saved_path__ = "saved/vars"
logging.getLogger('jieba').setLevel(logging.CRITICAL)


def softmax1d(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

@contextmanager
def time_recorder(operation=None):
    start = time.time()
    yield
    end = time.time()
    logging.info("{} cost {:.3} seconds".format(operation if operation else '', end - start))


def check_and_create_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def func_cache(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    def cache_decorator(func):
        def wrapper(*args, **kwargs):
            file_name_before_hash = '{}.{}'.format(func.__name__, '_'.join([str(arg) for arg in args]))
            cache_file_name = hashlib.md5(file_name_before_hash.encode('utf8')).hexdigest() + '.pkl'
            cache_file_path = os.path.join(cache_dir, cache_file_name)
            if os.path.exists(cache_file_path) and os.path.isfile(cache_file_path):
                with open(cache_file_path, 'rb') as cache:
                    logging.info('Loading cached result from {}'.format(cache_file_path))
                    return pickle.load(cache)

            result = func(*args, **kwargs)

            with open(cache_file_path, 'wb') as cache:
                pickle.dump(result, cache)
                logging.info('Saving result to cache {}'.format(cache_file_path))
            return result

        return wrapper

    return cache_decorator


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
        logging.info("cache for {} exists".format(name))
        with time_recorder("*** load {} from cache".format(name)):
            obj = load_var(name, path)
    else:
        logging.info("cache for {} does not exist".format(name))
        with time_recorder("*** create {} and save to cache".format(name)):
            obj = func()
            save_var(obj, name, path)
    return obj

def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True