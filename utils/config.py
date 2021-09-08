from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer


MODEL_CLASSES = {
    # Note: there may be some bug in `dcnn` modeling, if you want to pretraining.
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'electra': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
}

DATASET_LABEL_NUM = {
    'sst2': 2,
    'agnews': 4,
    'imdb': 2,
    'mr': 2,
    'onlineshopping': 2,
    'snli': 3,
}

LABEL_MAP = {
    'nli': {'entailment': 0, 'contradiction': 1, 'neutral': 2},
    'agnews': {'0': 0, '1': 1, '2': 2, '3': 3},
    'binary': {'0': 0, '1': 1}
}

GLOVE_CONFIGS = {
    '6B.50d': {'size': 50, 'lines': 400000},
    '840B.300d': {'size': 300, 'lines': 2196017}
}
