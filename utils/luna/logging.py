import os
from typing import List
import arrow
from inspect import isfunction

__log_path__ = "logs"
globals()["__default_target__"] = 'c'


def log_config(filename,
               default_target,
               log_path=__log_path__,
               append=False,
               ):
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_time = arrow.now().format('MMMDD_HH-mm-ss')
    
    def __lazy():
        return open("{}/{}.{}.txt".format(log_path, filename, log_time),
                    "a" if append else "w")
    logger = __lazy
    globals()["__logger__"] = logger
    globals()["__default_target__"] = default_target


def log(*info, target=None, color=None):
    if target is None:
        target = globals()["__default_target__"]
    assert target in ['c', 'f', 'cf', 'fc']
    if len(info) == 1:
        info_str = str(info[0])
    else:
        info = list(map(str, info))
        info_str = " ".join(info)
    if 'c' in target:
        if isfunction(color):
            print(color(info_str))
        else:
            print(info_str)
    if 'f' in target:
        logger = globals()["__logger__"]
        if isfunction(logger):
            logger = logger()
            globals()["__logger__"] = logger
        logger.write("{}\n".format(info_str))
        logger.flush()


log_buffer = []  # type:List


def log_to_buffer(*info):
    for ele in info:
        log_buffer.append(ele)


def log_flush_buffer(target=None):
    log("\n".join(log_buffer), target=target)
    log_buffer.clear()


