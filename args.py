import os
import torch
import logging
import argparse
from overrides import overrides
from utils.config import MODEL_CLASSES, DATASET_LABEL_NUM
from trainer import TRAINER_REGISTRY

from utils.public import set_seed, check_and_create_path


def string_to_bool(string_val):
    return True if string_val.lower() == 'true' else False



class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.mode = 'train'  # in ['train', 'attack', 'evaluate', 'augment', 'textattack_augment', 'dev_augment']
        self.model_type = 'bert'
        self.dataset_name = 'agnews'
        self.keep_sentiment_word = 'False'
        self.model_name_or_path = 'bert-base-uncased'
        self.evaluation_data_type = 'test'
        self.training_type = 'base'

        # attack parameters
        self.attack_method = 'bae'
        self.attack_times = 1
        self.attack_numbers = 1000
        # attack constraint args defined by us
        self.modify_ratio = 0.3
        self.neighbour_vocab_size = 50
        self.sentence_similarity = 0.840845057
        self.query_budget_size = self.neighbour_vocab_size

        # path parameters
        self.workspace = '/disks/sdb/lzy/workspace'
        self.dataset_path = self.workspace + '/dataset/' + self.dataset_name
        # self.log_path = self.workspace + '/log/' + self.dataset_name + '_' + self.model_type
        self.cache_path = self.workspace + '/cache'
        # self.saved_path = self.workspace + '/saved_models/' + self.dataset_name
        self.sentiment_path = self.workspace + '/dataset/sentiment_word/sentiment-words.txt'
        self.log_path = self.workspace + "/log"
        self.tensorboard = None

        # augment parameters
        self.use_aug = 'False'
        self.aug_ratio = 0.5
        self.aug_attacker = 'pwws'

        self.dev_aug_ratio = 0.5
        self.dev_aug_attacker = 'textfooler'
        self.use_dev_aug = 'False'

        # text_attack augment parameters
        self.split_num = 3
        self.start_idx = 0

        # model ensemble num in predicting (if needed)
        self.ensemble = 'False'
        self.ensemble_num = 100
        self.ensemble_method = 'logits' # in ['logits', 'votes']

        # base training hyper-parameters, if need other, define in subclass
        self.epochs = 10  # training epochs
        if string_to_bool(self.use_aug) and self.aug_ratio == 0.5:
            self.batch_size = 24
        else:
            self.batch_size = 32  # batch size
        # self.gradient_accumulation_steps = 1  # Number of updates steps to accumulate before performing a backward/update pass.
        # self.learning_rate = 5e-5  # The initial learning rate for Adam.
        # self.weight_decay = 1e-6  # weight decay
        # self.adam_epsilon = 1e-8  # epsilon for Adam optimizer
        # self.max_grad_norm = 1.0  # max gradient norm
        # self.learning_rate_decay = 0.1  # Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training

        # read dataset parameter
        if self.dataset_name != 'imdb':
            self.max_seq_len = 128
        else:
            self.max_seq_len = 256
        self.shuffle = 'True'

        # unchanged args
        self.type_accept_instance_as_input = ['mask', 'safer']
        # self.imdb_dir = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/aclImdb'
        # self.imdb_lm_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/lm_scores/imdb_all.txt'
        # self.counter_fitted_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/counter-fitted-vectors.txt'
        # self.snli_dir = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/snli'
        # self.snli_lm_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/lm_scores/snli_all.txt'
        # self.neighbor_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/counterfitted_neighbors.json'
        # self.glove_dir = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/glove'
        self.do_lower_case = 'True'
        # for lstm
        self.hidden_size = 100
        self.glove_name = '840B.300d'
        self.use_lm = 'False'

        # saving args
        self.saving_step = 1
        self.saving_last_epoch = 'False'
        self.compare_key = '+accuracy'
        self.file_name = None
        self.seed = 42
        self.remove_attack_constrainst = 'False'

    def build_environment(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.seed)

    def build_dataset_dir(self):
        testing_file = ['train.json', 'train.txt', 'train.csv', 'train.tsv']
        for file in testing_file:
            train_file_path = os.path.join(self.dataset_dir, file)
            if os.path.exists(train_file_path) and os.path.isfile(train_file_path):
                return
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)
        for file in testing_file:
            train_file_path = os.path.join(self.dataset_dir, file)
            if os.path.exists(train_file_path) and os.path.isfile(train_file_path):
                return
        raise FileNotFoundError("Dataset file cannot be found in dir {}".format(self.dataset_dir))

    # setting new saving path
    # the new saving path is combined by
    # args.saving_dir = args.saving_dir/${data}_${model}
    def build_saving_dir(self):
        self.saving_dir = os.path.join(self.saving_dir, "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.saving_dir)

    def build_logging_dir(self):
        self.log_path = os.path.join(self.log_path, "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.log_path)

    def build_caching_dir(self):
        # build safer perturbation set path
        if self.safer_perturbation_set is not None:
            self.safer_perturbation_set = os.path.join(self.caching_dir,
                                                       os.path.join(self.dataset_name, self.safer_perturbation_set))
        self.caching_dir = os.path.join(self.caching_dir, "{}_{}".format(self.dataset_name, self.model_type))
        check_and_create_path(self.caching_dir)

    def build_logging(self, **kwargs):
        self.log_path = os.path.join(self.log_path, f"{self.dataset_name}_{self.model_type}")
        logging_file_path = self.build_logging_file()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=logging_file_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

    def build_saving_file_name(self, description: str = None):
        '''
        build hyper-parameter for saving and loading model, some important hyper-parameters are set to define the saving file name
        :param args:
        :param description:
        :return:
        '''
        file_name = self.training_type
        if self.file_name is not None:
            file_name = "{}{}".format(file_name if file_name == "" else file_name + "_", self.file_name)
        hyper_parameter_dict = {'len': self.max_seq_len, 'epo': self.epochs, 'batch': self.batch_size}
        if self.training_type == 'freelb' or self.training_type == 'pgd' or self.training_type == 'tavat' or self.training_type == 'infobert':
            hyper_parameter_dict['advstep'] = self.adv_steps
            hyper_parameter_dict['advlr'] = self.adv_learning_rate
            hyper_parameter_dict['norm'] = self.adv_max_norm
        elif self.training_type == 'advhotflip':
            hyper_parameter_dict['rate'] = self.adv_change_rate
            hyper_parameter_dict['advstep'] = self.adv_steps
        elif self.training_type == 'ibp':
            hyper_parameter_dict['certfrac'] = self.cert_frac
            hyper_parameter_dict['certeps'] = self.cert_eps
        # elif self.training_type == 'metric' or self.training_type == 'metric_token':
        #     hyper_parameter_dict['rate'] = self.attack_max_rate_for_training
        #     hyper_parameter_dict['step'] = self.adv_steps
        #     hyper_parameter_dict['alpha'] = self.metric_learning_alpha
        #     hyper_parameter_dict['margin'] = self.metric_learning_margin
        if self.training_type == 'mask':
            hyper_parameter_dict['rate'] = self.mask_rate

        # if self.learning_rate != 5e-5:
        #     hyper_parameter_dict['lrate'] = self.learning_rate

        if file_name == "":
            file_name = '{}'.format(
                "-".join(["{}{}".format(key, value) for key, value in hyper_parameter_dict.items()]))
        else:
            file_name = '{}-{}'.format(file_name, "-".join(
                ["{}{}".format(key, value) for key, value in hyper_parameter_dict.items()]))

        if description is not None:
            file_name = '{}-{}'.format(file_name, description)
        return file_name

    def build_logging_path(self):
        if self.mode is None:
            return self.build_saving_file_name()
        elif self.mode == 'attack':
            if self.use_dev_aug == 'True':
                if self.training_type in ['mask', 'safer']:
                    logging_path = "{}-dev-{}-{}".format(self.mode, self.build_saving_file_name(), self.ensemble_method)
                    # if self.with_lm:
                    #     logging_path = "{}-{}".format(logging_path, 'lm')
                else:
                    logging_path = "{}-dev-{}".format(self.mode, self.build_saving_file_name())
            if self.training_type in ['mask', 'safer']:
                logging_path = "{}-{}-{}".format(self.mode, self.build_saving_file_name(), self.ensemble_method)
                # if self.with_lm:
                #     logging_path = "{}-{}".format(logging_path, 'lm')
            else:
                logging_path = "{}-{}".format(self.mode, self.build_saving_file_name())
            return logging_path
        else:
            return '{}-{}'.format(self.mode, self.build_saving_file_name())

    def build_logging_file(self):
        logging_path = self.build_logging_path()
        logging_path = os.path.join(self.log_path, logging_path)
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        if self.mode == 'attack':
            return os.path.join(logging_path, 'running.log')
        else:
            return os.path.join(self.log_path, '{}.log'.format(self.build_logging_path()))


    @staticmethod
    def parse(verbose=False) -> "ProgramArgs":
        parser = argparse.ArgumentParser()
        default_args = ProgramArgs()
        for key, value in default_args.__dict__.items():
            if type(value) == bool:
                raise Exception("Bool value is not supported!!!")
            parser.add_argument('--{}'.format(key),
                                action='store',
                                default=value,
                                type=type(value),
                                dest=str(key))
        parsed_args, _ = parser.parse_known_args(namespace=default_args)
        # if parsed_args.mode == 'train':
        if parsed_args.training_type in TRAINER_REGISTRY:
            TRAINER_REGISTRY[parsed_args.training_type].add_args(parser)
            parsed_args, _ = parser.parse_known_args(namespace=default_args)
        else:
            TRAINER_REGISTRY['base'].add_args(parser)
            parsed_args, _ = parser.parse_known_args(namespace=default_args)
        parsed_args.query_budget_size = parsed_args.neighbour_vocab_size
        if verbose:
            print("Args:")
            for key, value in parsed_args.__dict__.items():
                print("\t--{}={}".format(key, value))
        assert isinstance(parsed_args, ProgramArgs)
        return parsed_args  # type: ProgramArgs

    def __repr__(self):
        basic_ret = "\n"
        for key, value in self.__dict__.items():
            basic_ret += "\t--{}={}\n".format(key, value)
        return basic_ret

    __str__ = __repr__
