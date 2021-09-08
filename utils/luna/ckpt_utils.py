import re
import os


def checkpoint_paths(path, pattern=r'checkpoint@(\d+)\.pt'):
    """Retrieves all checkpoints found in `path` directory.
    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


# model_name = /aaa/bbb/ccc/model
# find:
#    /aaa/bbb/ccc/model.1
#    /aaa/bbb/ccc/model.2
#  * /aaa/bbb/ccc/model.best

def fetch_best_ckpt_name(model_path):
    model_name = model_path + '.best'
    if os.path.exists(model_name):
        print("Found checkpoint {}".format(model_name))
    else:
        model_name = fetch_last_ckpt_name(model_path)
        print("Best checkpoint not found, use latest {} instead".format(model_name))
    return model_name


def fetch_last_ckpt_name(model_path):
    splash_index = model_path.rindex('/')
    model_folder = model_path[:splash_index]
    model_file = model_path[splash_index+1:]
    files = checkpoint_paths(model_folder,r'{}.(\d+)'.format(model_file))
    return files[0]



# print(fetch_last_ckpt_name("/disks/sdb/zjiehang/zhou_data/saved_models/word_tag/lzynb"))
