import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from args import ProgramArgs, string_to_bool
args = ProgramArgs.parse(True)

args.build_environment()
args.build_logging()


# import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# _ = [tf.config.experimental.set_memory_growth(val, True) for val in gpus]

import logging
from typing import Union, Set

from textattack.models.tokenizers import GloveTokenizer
from textattack.models.helpers import GloveEmbeddingLayer
from tqdm import tqdm
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import ConcatDataset
import numpy as np

from data.reader import ClassificationReader, ClassificationReaderSpacy
from trainer import (
    BaseTrainer,
    FreeLBTrainer,
    HotflipTrainer,
    PGDTrainer,
    IBPTrainer,
    TokenAwareVirtualAdversarialTrainer,
    InfoBertTrainer,
    DNETrainer,
    MixUpTrainer,
    SAFERTrainer,
    MaskTrainer,
    ASCCTrainer
)
from utils.config import MODEL_CLASSES, DATASET_LABEL_NUM, GLOVE_CONFIGS
from utils.metrics import Metric, ClassificationMetric, SimplifidResult
from utils.my_utils import convert_batch_to_bert_input_dict
from utils.public import auto_create, check_and_create_path
from utils.textattack_utils import build_english_attacker, CustomTextAttackDataset
from utils.dne_utils import DecayAlphaHull, get_bert_vocab, WeightedEmbedding
from utils.ascc_utils import WarmupMultiStepLR
from data.instance import InputInstance

from textattack.loggers import AttackLogManager
from textattack.models.wrappers import HuggingFaceModelWrapper, PyTorchModelWrapper, HuggingFaceModelEnsembleWrapper
from textattack.augmentation.augmenter import Augmenter
from textattack.augmentation.faster_augmentor import FasterAugmenter
from textattack.transformations import WordSwapWordNet, WordSwapEmbedding, WordSwapMaskedLM
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from model.models import LSTMModel, MixText, ASCCModel
from utils.certified import attacks, vocabulary, data_util


class AttackBenchmarkTask(object):
    def __init__(self, args: ProgramArgs):
        self.methods = {'train': self.train,
                        'evaluate': self.evaluate,
                        # 'predict': self.predict,
                        'attack': self.attack,
                        'augment': self.augment,
                        'textattack_augment': self.textattack_augment,
                        'dev_augment': self.dev_augment,
                        'dev_eval': self.dev_aug_evaluate,
                        }
        assert args.mode in self.methods, 'mode {} not found'.format(args.mode)

        self.tensor_input = False if args.training_type in args.type_accept_instance_as_input and args.mode == 'train' else True

        if args.model_type != 'lstm':
            self.tokenizer = self._build_tokenizer(args)
            self.dataset_reader = ClassificationReader(model_type=args.model_type, max_seq_len=args.max_seq_len)
        else:
            self.dataset_reader = ClassificationReaderSpacy(model_type=args.model_type,
                                                            max_seq_len=args.max_seq_len)
            # build attack surface
            if string_to_bool(args.use_lm):
                if args.dataset_name == 'imdb':
                    lm_file = args.imdb_lm_file
                elif args.dataset_name == 'snli':
                    lm_file = args.snli_lm_file
                else:
                    raise NotImplementedError
                self.attack_surface = auto_create(
                    f'{args.dataset_name}_attack_surface_lm',
                    lambda: attacks.LMConstrainedAttackSurface.from_files(args.neighbor_file, lm_file),
                    True, path=args.cache_path
                )
            else:
                self.attack_surface = auto_create(
                    'attack_surface_cf',
                    lambda: attacks.WordSubstitutionAttackSurface.from_file(args.neighbor_file),
                    True, path=args.cache_path
                )

        if args.use_dev_aug == 'False':
            self.train_raw, self.eval_raw, self.test_raw = auto_create(
                f'{args.dataset_name}_raw_datasets', lambda: self._build_raw_dataset(args),
                True, path=args.cache_path
            )
            self.train_dataset, self.eval_dataset, self.test_dataset = auto_create(
                    f'{args.dataset_name}_tokenized_datasets', lambda: self._build_tokenized_dataset(args),
                    True, path=args.cache_path
            )
        else:
            self.train_raw, self.eval_raw, self.test_raw = auto_create(
                f'{args.dataset_name}_dev_{args.dev_aug_attacker}_{args.dev_aug_ratio}_datasets', lambda: self._build_raw_dataset(args),
                True, path=args.cache_path
            )
            self.train_dataset, self.eval_dataset, self.test_dataset = auto_create(
                    f'{args.dataset_name}_dev_{args.dev_aug_attacker}_{args.dev_aug_ratio}_tokenized_datasets', lambda: self._build_tokenized_dataset(args),
                    True, path=args.cache_path
            )
        if not self.tensor_input:
            self.train_dataset = self.train_raw

        if args.model_type == 'lstm':
            word_set = self.dataset_reader.get_word_set(self.train_raw + self.eval_raw,
                                                        args.counter_fitted_file, self.attack_surface)
            self.vocab, self.word_mat = auto_create(
                f'{args.dataset_name}_glove_vocab_emb',
                lambda: vocabulary.Vocabulary.read_word_vecs(word_set, args.glove_dir, args.glove_name,
                                                             args.device, prepend_null=True),
                True,
                path=args.cache_path
            )

        self.data_loader, self.eval_data_loader, self.test_data_loader = self._build_dataloader(args)
        self.model = self._build_model(args)
        self.forbidden_words = self._build_forbidden_words(args.sentiment_path) if string_to_bool(
            args.keep_sentiment_word) else None
        self.loss_function = self._build_criterion(args)

    def train(self, args: ProgramArgs):
        self.optimizer = self._build_optimizer(args)
        self.lr_scheduler = self._build_lr_scheduler(args)
        self.writer = self._build_writer(args)
        trainer = self._build_trainer(args)
        best_metric = None
        epoch_now = self._check_training_epoch(args)
        for epoch_time in range(epoch_now, args.epochs):
            if args.training_type == 'ibp':
                trainer.set_epoch(epoch_time)
            trainer.train_epoch(args, epoch_time)

            # saving model according to epoch_time
            self._saving_model_by_epoch(args, epoch_time)

            # evaluate model according to epoch_time
            metric = self.evaluate(args, is_training=True)

            # update best metric
            # if best_metric is None, update it with epoch metric directly, otherwise compare it with epoch_metric
            if best_metric is None or metric > best_metric:
                best_metric = metric
                self._save_model_to_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description='best'))

        self.evaluate(args)

    def dev_aug_evaluate(self, args: ProgramArgs):
        self.optimizer = self._build_optimizer(args)
        self.lr_scheduler = self._build_lr_scheduler(args)
        self.writer = self._build_writer(args)
        best_metric = None
        for epoch_time in range(args.epochs):
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description='epoch{}'.format(epoch_time)))
            metric = self.evaluate(args, is_training=True)
            if best_metric is None or metric > best_metric:
                best_metric = metric
                self._save_model_to_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description=f"best_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
        self.evaluate(args)


    @torch.no_grad()
    def evaluate(self, args: ProgramArgs, is_training: bool = False) -> Metric:
        if is_training:
            logging.info('Using current modeling parameter to evaluate')
            epoch_iterator = tqdm(self.eval_data_loader)
        else:
            if args.use_dev_aug == 'False':
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                          args.build_saving_file_name(description='best'))
            else:
                self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description=f"best_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
            epoch_iterator = tqdm(self.test_data_loader)
        self.model.eval()

        metric = ClassificationMetric(compare_key=args.compare_key)
        for step, batch in enumerate(epoch_iterator):
            if args.model_type == 'lstm':
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3].long().t().squeeze()
                logits = self.model.forward(batch, compute_bounds=False)
                losses = self.loss_function(logits, golds)
            else:
                assert isinstance(batch[0], torch.Tensor)
                batch = tuple(t.to(args.device) for t in batch)
                golds = batch[3]
                inputs = convert_batch_to_bert_input_dict(batch, args.model_type)
                logits = self.model.forward(**inputs)[0]
                losses = self.loss_function(logits.view(-1, DATASET_LABEL_NUM[args.dataset_name]), golds.view(-1))
                epoch_iterator.set_description('loss: {:.4f}'.format(torch.mean(losses)))
            metric(losses, logits, golds)

        print(metric)
        logging.info(metric)
        return metric

    def attack(self, args: ProgramArgs):
        if args.use_dev_aug == 'False':
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                      args.build_saving_file_name(description='best'))
        else:
            self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                      args.build_saving_file_name(description=f"best_{args.dev_aug_attacker}_{args.dev_aug_ratio}"))
        self.model.eval()
        attacker = self._build_attacker(args)

        if args.evaluation_data_type == 'dev':
            dataset = self.eval_raw
        else:
            dataset = self.test_raw
        test_instances = dataset

        attacker_log_path = '{}'.format(args.build_logging_path())
        attacker_log_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
                                         attacker_log_path)
        attacker_log_manager = AttackLogManager()
        # attacker_log_manager.enable_stdout()
        attacker_log_manager.add_output_file(
            os.path.join(attacker_log_path, f'{args.attack_method}_{args.neighbour_vocab_size}_{args.modify_ratio}.txt'))
        test_instances = [x for x in test_instances if len(x.text_a.split(' ')) > 4]
        # attack multiple times for average success rate
        for i in range(args.attack_times):
            print("Attack time {}".format(i))
            choice_instances = np.random.choice(test_instances, size=(args.attack_numbers,), replace=False)
            dataset = CustomTextAttackDataset.from_instances(args.dataset_name, choice_instances,
                                                             self.dataset_reader.get_labels())
            results_iterable = attacker.attack_dataset(dataset)
            description = tqdm(results_iterable, total=len(choice_instances))
            result_statistics = SimplifidResult()
            for result in description:
                try:
                    attacker_log_manager.log_result(result)
                    result_statistics(result)
                    description.set_description(result_statistics.__str__())
                except Exception as e:
                    print('error in process')
                    continue

        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def augment(self, args: ProgramArgs):
        self._loading_model_from_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                      args.build_saving_file_name(description='best'))
        self.model.eval()
        attacker = self._build_attacker(args)
        training_instance = [instance for instance in self.train_raw if instance.length() > 4]
        training_len = len(training_instance)
        print('Training Set: {} sentences. '.format(training_len))
        attacker_log_manager = AttackLogManager()
        dataset = CustomTextAttackDataset.from_instances(f'{args.dataset_name}_aug', training_instance,
                                                         self.dataset_reader.get_labels())
        results_iterable = attacker.attack_dataset(dataset)
        aug_instances = []
        for (result, instance) in tqdm(zip(results_iterable, training_instance), total=training_len):
            try:
                adv_sentence = result.perturbed_text()
                aug_instances.append(InputInstance.create_instance_with_perturbed_sentence(instance, adv_sentence))
            except:
                continue
        self.dataset_reader.saving_instances(aug_instances, args.dataset_path, 'aug_{}'.format(args.attack_method))
        print(f'Augmented {len(aug_instances)} sentences. ')
        attacker_log_manager.enable_stdout()
        attacker_log_manager.log_summary()

    def dev_augment(self, args: ProgramArgs):
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        _ = [tf.config.experimental.set_memory_growth(val, True) for val in gpus]
        from utils.augmentor import TEXTFOOLER_SET
        transformation = WordSwapEmbedding(max_candidates=50)
        stopwords = TEXTFOOLER_SET
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        
        augmenter = FasterAugmenter(transformation, constraints, pct_words_to_swap=args.modify_ratio)
        eval_instances = [instance for instance in self.eval_raw if instance.length() > 4]
        aug_num = int(args.dev_aug_ratio * len(eval_instances))
        instances = np.random.choice(eval_instances, size=(aug_num,), replace=False)
        aug_len = len(instances)
        print('Dev Set: {} sentences. '.format(aug_len))
        aug_instances = []
        is_nli = instances[0].is_nli()
        if is_nli:
            for instance in tqdm(instances):
                aug_instances.append((instance.text_a, augmenter.augment(instance.text_b), instance.label))
        else:
            for instance in tqdm(instances):
                aug_instances.append((augmenter.augment(instance.text_a), instance.label))

        with open(f'{args.dataset_path}/dev_{args.dev_aug_attacker}_{args.dev_aug_ratio}.txt', 'w', encoding='utf-8') as fout:
            for instance in eval_instances:
                if is_nli:
                    fout.write(f'{instance.label}\t{instance.text_a}\t{instance.text_b}\n')
                else:
                    fout.write(f'{instance.text_a}\t{instance.label}\n')
            for instance in aug_instances:
                if is_nli:
                    fout.write(f'{instance[2]}\t{instance[0]}\t{instance[1]}\n')
                else:
                    fout.write(f'{instance[0][0]}\t{instance[1]}\n')
        print(f'Augmented {len(aug_instances)} sentences.')
        pass


    def textattack_augment(self, args: ProgramArgs):
        if args.attack_method == 'pwws':
            transformation = WordSwapWordNet()
            constraints = [RepeatModification(), StopwordModification()]
        elif args.attack_method == 'textfooler':
            transformation = WordSwapEmbedding(max_candidates=50)
            stopwords = set(
                ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost",
                 "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another",
                 "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as",
                 "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
                 "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn",
                 "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere",
                 "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for",
                 "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence",
                 "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
                 "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's",
                 "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn",
                 "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself",
                 "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none",
                 "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only",
                 "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per",
                 "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't",
                 "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the",
                 "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
                 "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru",
                 "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve",
                 "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence",
                 "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
                 "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within",
                 "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]
            )
            constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
            constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
            constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
            use_constraint = UniversalSentenceEncoder(
                threshold=0.840845057,
                metric="angular",
                compare_against_original=False,
                window_size=15,
                skip_text_shorter_than_window=True,
            )
            constraints.append(use_constraint)
        elif args.attack_method == 'bae':
            transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)
            constraints = [RepeatModification(), StopwordModification()]
            constraints.append(MaxWordsPerturbed(max_percent=0.4))
            use_constraint = UniversalSentenceEncoder(
                threshold=0.2,
                metric="cosine",
                compare_against_original=True,
                window_size=None,
            )
            constraints.append(use_constraint)

        # augmenter = Augmenter(transformation, constraints, pct_words_to_swap=args.modify_ratio)
        augmenter = FasterAugmenter(transformation, constraints, pct_words_to_swap=args.modify_ratio)
        training_instance = [instance for instance in self.train_raw if instance.length() > 4][30000:90000]
        # split_size = int(len(training_instance) / args.split_num)
        # training_instance = training_instance[args.start_idx * split_size: (args.start_idx + 1) * split_size]
        training_len = len(training_instance)
        print('Training Set: {} sentences. '.format(training_len))
        aug_instances = []
        is_nli = training_instance[0].is_nli()
        if is_nli:
            for instance in tqdm(training_instance):
                aug_instances.append((instance.text_a, augmenter.augment(instance.text_b), instance.label))
        else:
            for instance in tqdm(training_instance):
                aug_instances.append((augmenter.augment(instance.text_a), instance.label))

        with open(f'{args.dataset_path}/aug_{args.attack_method}.txt', 'w', encoding='utf-8') as fout:
            for instance in aug_instances:
                if is_nli:
                    fout.write(f'{instance[2]}\t{instance[0]}\t{instance[1]}\n')
                else:
                    fout.write(f'{instance[0]}\t{instance[1]}\n')
        print(f'Augmented {len(aug_instances)} sentences.')

    def _save_model_to_file(self, save_dir: str, file_name: str):
        save_file_name = '{}.pth'.format(file_name)
        check_and_create_path(save_dir)
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.model.state_dict(), save_path)
        logging.info('Saving model to {}'.format(save_path))

    def _saving_model_by_epoch(self, args: ProgramArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                self._save_model_to_file(f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}",
                                         args.build_saving_file_name(description='epoch{}'.format(epoch)))
    
    def _check_training_epoch(self, args: ProgramArgs):
        epoch_now = 0
        save_dir = f"{args.workspace}/saved_models/{args.dataset_name}_{args.model_type}"
        for epoch in range(args.epochs):
            file_name = args.build_saving_file_name(description='epoch{}'.format(epoch))
            save_file_name = '{}.pth'.format(file_name)
            check_and_create_path(save_dir)
            save_path = os.path.join(save_dir, save_file_name)
            if os.path.exists(save_path) and os.path.isfile(save_path):
                epoch_now = epoch + 1
                continue
            else:
                if epoch_now != 0:
                    file_name = args.build_saving_file_name(description='epoch{}'.format(epoch-1))
                    self._loading_model_from_file(save_dir, file_name)
                break
        return epoch_now

    def _loading_model_from_file(self, save_dir: str, file_name: str):
        load_file_name = '{}.pth'.format(file_name)
        load_path = os.path.join(save_dir, load_file_name)
        assert os.path.exists(load_path) and os.path.isfile(load_path), '{} not exits'.format(load_path)
        self.model.load_state_dict(torch.load(load_path), strict=False)
        logging.info('Loading model from {}'.format(load_path))

    def _build_trainer(self, args: ProgramArgs):
        trainer = BaseTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                              self.writer)
        if args.training_type == 'freelb':
            trainer = FreeLBTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                                    self.writer)
        elif args.training_type == 'pgd':
            trainer = PGDTrainer(self.data_loader, self.model, self.loss_function, self.optimizer, self.lr_scheduler,
                                 self.writer)
        elif args.training_type == 'advhotflip':
            trainer = HotflipTrainer(args, self.tokenizer, self.data_loader, self.model, self.loss_function,
                                     self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'ibp':
            trainer = IBPTrainer(args, self.data_loader, self.model, self.loss_function, self.optimizer,
                                 self.lr_scheduler, self.writer)
        elif args.training_type == 'tavat':
            trainer = TokenAwareVirtualAdversarialTrainer(args, self.data_loader, self.model, self.loss_function,
                                                          self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'infobert':
            trainer = InfoBertTrainer(args, self.data_loader, self.model, self.loss_function,
                                      self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'dne':
            trainer = DNETrainer(args, self.data_loader, self.model, self.loss_function,
                                 self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'mixup':
            trainer = MixUpTrainer(args, self.data_loader, self.model, self.loss_function,
                                   self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'safer':
            trainer = SAFERTrainer(args, self.tokenizer, self.dataset_reader, self.data_loader, self.model,
                                   self.loss_function, self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'mask':
            trainer = MaskTrainer(args, self.tokenizer, self.dataset_reader, self.data_loader, self.model,
                                   self.loss_function, self.optimizer, self.lr_scheduler, self.writer)
        elif args.training_type == 'ascc':
            trainer = ASCCTrainer(args, self.data_loader, self.model, self.loss_function,
                                 self.optimizer, self.lr_scheduler, self.writer)

        return trainer

    def _build_optimizer(self, args: ProgramArgs, **kwargs):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        return optimizer

    def _build_model(self, args: ProgramArgs):
        if args.model_type == 'lstm':
            model = LSTMModel(
                GLOVE_CONFIGS[args.glove_name]['size'], args.hidden_size,
                self.word_mat, args.device,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                pool=args.pool,
                dropout=args.dropout_prob,
                no_wordvec_layer=args.no_wordvec_layer).to(args.device)
        elif args.training_type == 'mixup':
            config_class, _, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = MixText.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
        elif args.training_type == 'dne':
            config_class, model_class, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
            bert_vocab = get_bert_vocab()
            hull = DecayAlphaHull.build(
                alpha=args.dir_alpha,
                decay=args.dir_decay,
                nbr_file=args.nbr_file,
                vocab=bert_vocab,
                nbr_num=args.nbr_num,
                second_order=True
            )
            # here we just focus on bert model
            model.bert.embeddings.word_embeddings = WeightedEmbedding(
                num_embeddings=bert_vocab.get_vocab_size('tokens'),
                embedding_dim=768,
                padding_idx=model.bert.embeddings.word_embeddings.padding_idx,
                _weight=model.bert.embeddings.word_embeddings.weight,
                hull=hull,
                sparse=False)
        elif args.training_type == 'ascc':
            config_class, _, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = ASCCModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
            bert_vocab = get_bert_vocab()
            model.build_nbrs(args.nbr_file, bert_vocab, args.alpha, args.num_steps)
        else:
            config_class, model_class, _ = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(
                args.model_name_or_path,
                num_labels=DATASET_LABEL_NUM[args.dataset_name],
                finetuning_task=args.dataset_name,
                output_hidden_states=True,
            )
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('ckpt' in args.model_name_or_path),
                config=config
            ).to(args.device)
        return model

    def _build_tokenizer(self, args: ProgramArgs):
        _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=string_to_bool(args.do_lower_case)
        )
        return tokenizer

    def _build_raw_dataset(self, args: ProgramArgs):
        train_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                       split='train')
        if args.use_dev_aug == 'False':
            eval_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                      split='dev')
        else:
            eval_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                       split=f"dev_{args.dev_aug_attacker}_{args.dev_aug_ratio}")
        test_raw = self.dataset_reader.read_from_file(file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                                                      split='test')

        return train_raw, eval_raw, test_raw

    def _build_tokenized_dataset(self, args: ProgramArgs):
        assert isinstance(self.dataset_reader, ClassificationReader)
        train_dataset = self.dataset_reader.get_dataset(self.train_raw, self.tokenizer)
        eval_dataset = self.dataset_reader.get_dataset(self.eval_raw, self.tokenizer)
        test_dataset = self.dataset_reader.get_dataset(self.test_raw, self.tokenizer)

        return train_dataset, eval_dataset, test_dataset

    def _build_dataloader(self, args: ProgramArgs):
        if args.model_type == 'lstm':
            train_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.train_dataset,
                                                                       batch_size=args.batch_size,
                                                                       vocab=self.vocab)
            eval_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.eval_dataset,
                                                                      batch_size=args.batch_size,
                                                                      vocab=self.vocab)
            test_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.test_dataset,
                                                                      batch_size=args.batch_size,
                                                                      vocab=self.vocab)
        else:
            assert isinstance(self.dataset_reader, ClassificationReader)
            if string_to_bool(args.use_aug):
                aug_raw = auto_create(
                    f'{args.dataset_name}_raw_aug_{args.aug_attacker}',
                    lambda: self.dataset_reader.read_from_file(
                        file_path=f"{args.workspace}/dataset/{args.dataset_name}",
                        split=f'aug_{args.aug_attacker}'),
                    True,
                    path=args.cache_path
                )
                aug_dataset = auto_create(
                    f'{args.dataset_name}_tokenized_aug_{args.aug_attacker}',
                    lambda: self.dataset_reader.get_dataset(aug_raw, self.tokenizer),
                    True,
                    path=args.cache_path
                )

                if args.aug_ratio == 1.0:
                    train_data_loader = self.dataset_reader.get_dataset_loader(dataset=aug_dataset,
                                                                               tokenized=True,
                                                                               batch_size=args.batch_size,
                                                                               shuffle=string_to_bool(args.shuffle))
                elif args.aug_ratio == 0.5:
                    train_data_loader = self.dataset_reader.get_dataset_loader(
                        dataset=ConcatDataset([self.train_dataset, aug_dataset]),
                        tokenized=True,
                        batch_size=args.batch_size,
                        shuffle=string_to_bool(args.shuffle))
                else:
                    train_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.train_dataset,
                                                                               tokenized=True,
                                                                               batch_size=args.batch_size,
                                                                               shuffle=string_to_bool(args.shuffle))
            else:
                train_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.train_dataset,
                                                                           tokenized=self.tensor_input,
                                                                           batch_size=args.batch_size,
                                                                           shuffle=string_to_bool(args.shuffle))
            eval_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.eval_dataset,
                                                                      tokenized=True,
                                                                      batch_size=args.batch_size,
                                                                      shuffle=string_to_bool(args.shuffle))
            test_data_loader = self.dataset_reader.get_dataset_loader(dataset=self.test_dataset,
                                                                      tokenized=True,
                                                                      batch_size=args.batch_size,
                                                                      shuffle=string_to_bool(args.shuffle))
        return train_data_loader, eval_data_loader, test_data_loader

    def _build_criterion(self, args: ProgramArgs):
        # if args.model_type == 'lstm':
        #     return BCEWithLogitsLoss(reduction='mean')
        return CrossEntropyLoss(reduction='none')

    def _build_lr_scheduler(self, args: ProgramArgs):
        if args.training_type == 'ascc':
            return WarmupMultiStepLR(self.optimizer, (40, 80), 0.1, 1.0 / 10.0, 2, 'linear')
        return CosineAnnealingLR(self.optimizer, len(self.train_dataset) // args.batch_size * args.epochs)

    def _build_writer(self, args: ProgramArgs, **kwargs) -> Union[SummaryWriter, None]:
        writer = None
        if args.tensorboard == 'yes':
            tensorboard_file_name = '{}-tensorboard'.format(args.build_logging_path())
            tensorboard_path = os.path.join(f"{args.workspace}/log/{args.dataset_name}_{args.model_type}",
                                            tensorboard_file_name)
            writer = SummaryWriter(tensorboard_path)
        return writer

    def _build_forbidden_words(self, file_path: str) -> Set[str]:
        sentiment_words_set = set()
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file.readlines():
                sentiment_words_set.add(line.strip())
        return sentiment_words_set

    def _build_attacker(self, args: ProgramArgs):
        if args.training_type in ['dne', 'safer', 'mask']:
           model_wrapper = HuggingFaceModelEnsembleWrapper(args, self.model, self.tokenizer)
        elif args.model_type != 'lstm':
            model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, batch_size=args.batch_size)
        else:
            tokenizer = GloveTokenizer(word_id_map=self.vocab.word2index,
                                       unk_token_id=0,
                                       pad_token_id=1,
                                       max_length=args.max_seq_len
                                       )
            model_wrapper = PyTorchModelWrapper(self.model, tokenizer, batch_size=args.batch_size)

        attacker = build_english_attacker(args, model_wrapper)
        return attacker


if __name__ == '__main__':
    logging.info(args)
    test = AttackBenchmarkTask(args)

    test.methods[args.mode](args)
