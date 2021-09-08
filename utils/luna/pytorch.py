from .public import *
import torch
import random
import numpy as np
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from .logging import log

__model_path__ = "saved/models"


def get_device(model):
    return next(model.parameters()).device
    

def cast_list(array, squeeze=True):
    if isinstance(array, torch.Tensor):
        return cast_list(array.detach().cpu().numpy(),squeeze)
    if isinstance(array, list):
        return cast_list(np.array(array),squeeze)
    if isinstance(array, np.ndarray):
        if squeeze:
            array = array.squeeze().tolist()
            return array if isinstance(array, list) else [array]
        else:
            return array.tolist()


def flt2str(flt, fmt=":6.2f", cat=None):
    fmter = "{{{}}}".format(fmt)
    if isinstance(flt, (float, int)):
        return fmter.format(flt)
    elif isinstance(flt, list):
        str_lst = [fmter.format(ele) for ele in flt]
        if cat is None:
            return str_lst
        else:
            return cat.join(str_lst)
    elif isinstance(flt, (torch.Tensor, np.ndarray)):
        return flt2str(cast_list(flt), fmt, cat)
    else:
        raise Exception('WTF objects are you passing?')


def allocate_cuda_device(cuda_idx) -> torch.device:
    if torch.cuda.is_available() and cuda_idx >= 0:
        device = torch.device("cuda:{}".format(cuda_idx))
    else:
        device = torch.device("cpu")
    return device


def set_gpu_device(device_id):
    torch.cuda.set_device(device_id)


def gpu(*x):
    if torch.cuda.is_available():
        if len(x) == 1:
            return x[0].cuda()
        else:
            return map(lambda m: m.cuda(), x)
    else:
        if len(x) == 1:
            return x[0]
        else:
            return x

def exist_model(saved_model_name):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    for file in os.listdir(__model_path__):
        file = file[:-5]
        name = file.split('@')[0]
        ckpt = int(file.split('@')[1])
        if name == saved_model_name:
            return True
    return False

def load_model(model, saved_model_name, checkpoint=-1):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        for file in os.listdir(__model_path__):
            if file[-5:] == '.ckpt':
                file = file[:-5]
                name = file.split('@')[0]
                ckpt = int(file.split('@')[1])
                if name == saved_model_name and ckpt > checkpoint:
                    checkpoint = ckpt
    path = "{}/{}@{}.ckpt".format(__model_path__, saved_model_name, checkpoint)
    if not os.path.exists(path):
        log("Checkpoint {} not found.".format(path))
    else:
        log("Restore model from checkpoint {}.".format(path))
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path))
    return checkpoint


def save_model(model, saved_model_name, checkpoint=-1):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        checkpoint = 0
    path = "{}/{}@{}.ckpt".format(__model_path__, saved_model_name, checkpoint)
    torch.save(model.state_dict(), path)
    log("Model saved to {}.".format(path))
    return checkpoint + 1


class ModelManager:
    def __init__(self, model, model_name, seconds=-1, init_ckpt=-1):
        self.model = model
        self.model_name = model_name
        self.seconds = seconds
        self.last_time = time.time()
        self.ckpt = load_model(model=self.model,
                               saved_model_name=self.model_name,
                               checkpoint=init_ckpt)

    def save(self):
        curr_time = time.time()
        if curr_time - self.last_time > self.seconds:
            save_model(model=self.model,
                       saved_model_name=self.model_name,
                       checkpoint=self.ckpt)
            self.last_time = curr_time

    def next_ckpt(self):
        self.ckpt += 1


@deprecated
def ten2var(x):
    return gpu(torch.autograd.Variable(x))


@deprecated
def long2var(x):
    return gpu(torch.autograd.Variable(torch.LongTensor(x)))


@deprecated
def float2var(x):
    return gpu(torch.autograd.Variable(torch.FloatTensor(x)))


def var2list(x):
    return x.cpu().data.numpy().tolist()


def var2num(x):
    return x.cpu().data[0]


def load_word2vec(embedding: torch.nn.Embedding,
                  word_dict: Dict[str, int],
                  word2vec_path,
                  norm=True,
                  cached_name=None):
    cache = "{}{}".format(cached_name, ".norm" if norm else "")
    if cached_name and exist_var(cache):
        log("Load word2vec from cache {}".format(cache))
        pre_embedding = load_var(cache)
    else:
        log("Load word2vec from {}".format(word2vec_path))
        pre_embedding = np.random.normal(0, 1, embedding.weight.size())
        word2vec_file = open(word2vec_path, errors='ignore')
        # x = 0
        found = 0
        emb_size = -1
        error_num = 0
        for line in word2vec_file.readlines():
            # x += 1
            # log("Process line {} in file {}".format(x, word2vec_path))
            split = re.split(r"\s+", line.strip())
            if emb_size == -1:
                emb_size = len(split) - 1
            if len(split) != emb_size + 1 or len(split) < 10:
                error_num += 1
                continue
            word = split[0]
            if word in word_dict:
                found += 1
                pre_embedding[word_dict[word]] = np.array(list(map(float, split[1:])))
        log("Error line: ", error_num)
        log("Pre_train match case: {:.4f}".format(found / len(word_dict)))
        if norm:
            pre_embedding = pre_embedding / np.std(pre_embedding)
        if cached_name:
            save_var(pre_embedding, cache)
    embedding.weight.data.copy_(torch.from_numpy(pre_embedding))


def show_mean_std(tensor, name=""):
    print("[INFO] {} Mean {:.4f} Std {:.4f} Max {:.4f}".format(
        name,
        tensor.mean().item(),
        tensor.std().item(),
        tensor.max().item()
    ))


def idx_to_msk(idx, num_classes):
    """
    idx: [1, 2, 3]
    msk: [[0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
    """
    assert idx.dim() == 1, "the dimension of idx must be 1, e.g. [1, 2, 3]"
    msk = torch.zeros((idx.size(0), num_classes), device=idx.device)
    msk.scatter_(1, idx.view(-1, 1), 1)
    msk = msk.byte()
    return msk


def msk_to_idx(msk):
    assert msk.sum() == msk.size(0), \
        "only one element is allowed to be 1 in each row"
    return msk.nonzero()[:, 1].flatten()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def flip_by_length(inputs, lengths):
#     rev_inputs = []
#     for it_id, it_input in enumerate(inputs):
#         it_len = lengths[it_id]
#         rev_input = torch.cat([
#             it_input.index_select(0, torch.tensor(list(reversed(range(it_len)))).to(inputs.device)),
#             torch.zeros_like(it_input[it_len:]).to(inputs.device)
#         ])
#         rev_inputs.append(rev_input)
#     rev_inputs = torch.stack(rev_inputs)
#     return rev_inputs


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# def focal_loss(inputs,
#                targets,
#                gamma=2, alpha=None, reduction="mean"):
#     batch_size = inputs.size(0)
#     num_classes = inputs.size(1)
#     prob = F.softmax(inputs, dim=1).clamp(1e-10, 1.)
#     # prob = inputs.exp()
#
#     class_mask = inputs.data.new(batch_size, num_classes).fill_(0)
#     ids = targets.view(-1, 1)
#     class_mask.scatter_(1, ids.data, 1.)
#     if alpha is None:
#         alpha = torch.ones(num_classes, 1).to(inputs.device)
#     alpha = alpha[ids.data.view(-1)]
#
#     probs = (prob * class_mask).sum(1).view(-1, 1)
#
#     log_p = probs.log()
#
#     batch_loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p
#
#     if reduction == "mean":
#         loss = batch_loss.mean()
#     elif reduction == "sum":
#         loss = batch_loss.sum()
#     elif reduction == "zheng":
#         pred = torch.argmax(inputs, dim=1)
#         ce_mask = pred != targets
#         loss = torch.mean(batch_loss * ce_mask)
#     elif reduction == "none":
#         loss = batch_loss
#     else:
#         raise Exception()
#     return loss


# class NonLinearLayerWithRes(torch.nn.Module):
#     def __init__(self, d_in, d_hidden, dropout):
#         super(NonLinearLayerWithRes, self).__init__()
#         self.fc1 = torch.nn.Linear(d_in, d_hidden)
#         self.fc2 = torch.nn.Linear(d_hidden, d_in)
#         self.drop = torch.nn.Dropout(dropout)
#
#     def forward(self, x):
#         out = self.fc2(F.relu(self.fc1(x)))
#         out += x
#         out = self.drop(out)
#         # out = torch.nn.LayerNorm(out)
#         return out
