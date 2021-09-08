# -*- coding: utf-8 -*-
import torch
import scipy
import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod
from functools import lru_cache
from overrides import overrides
from sklearn.metrics import f1_score, classification_report
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult

from textattack.attack_results.attack_result import AttackResult

ABSTAIN_FLAG = -1
BASE_FOR_CLASSIFICATION = ['loss', 'accuracy', 'f1']

class Metric(ABC):
    def __init__(self, compare_key='-loss'):
        compare_key = compare_key.lower()
        if not compare_key.startswith('-') and compare_key[0].isalnum():
            compare_key = "+{}".format(compare_key)
        self.compare_key = compare_key

    def __str__(self):
        return ', '.join(['{}: {:.4f}'.format(key, value) for (key, value) in self.get_metric().items()])

    @abstractmethod
    def __call__(self, ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset: bool = False) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def __gt__(self, other: "Metric"):
        is_large = self.compare_key.startswith('+')
        key = self.compare_key[1:]
        assert key in self.get_metric()

        if is_large:
            return self.get_metric()[key] > other.get_metric()[key]
        else:
            return self.get_metric()[key] < other.get_metric()[key]

    def __ge__(self, other: "Metric"):
        is_large = self.compare_key.startswith('+')
        key = self.compare_key[1:]
        assert key in self.get_metric()

        if is_large:
            return self.get_metric()[key] >= other.get_metric()[key]
        else:
            return self.get_metric()[key] <= other.get_metric()[key]


class ClassificationMetric(Metric):
    def __init__(self, compare_key='-loss'):
        super().__init__(compare_key)
        self._all_losses = torch.FloatTensor()
        self._all_predictions = torch.LongTensor()
        self._all_gold_labels = torch.LongTensor()


    @overrides
    def __call__(self,
                 losses: torch.Tensor,
                 logits: torch.FloatTensor,
                 gold_labels: torch.LongTensor
                 ) -> None:
        self._all_losses = torch.cat([self._all_losses, losses.to(self._all_losses.device)], dim=0)
        predictions = logits.argmax(-1).to(self._all_predictions.device)
        self._all_predictions = torch.cat([self._all_predictions, predictions], dim=0)
        self._all_gold_labels = torch.cat([self._all_gold_labels, gold_labels.to(self._all_gold_labels.device)],dim=0)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict:
        loss = torch.mean(self._all_losses).item()
        total_num = self._all_gold_labels.shape[0]
        accuracy = torch.sum(self._all_gold_labels == self._all_predictions).item() / total_num
        f1 = f1_score(y_true=self._all_gold_labels.numpy(), y_pred=self._all_predictions.numpy(), average='macro')
        result = {'loss': loss,
                  'accuracy': accuracy,
                  'f1': f1}
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self) -> None:
        self._all_losses = torch.FloatTensor()
        self._all_predictions = torch.LongTensor()
        self._all_gold_labels = torch.LongTensor()


class RandomSmoothAccuracyMetrics(Metric):
    def __init__(self, compare_key='+accuracy'):
        super().__init__(compare_key)
        self._all_numbers = 0
        self._abstain_numbers = 0
        self._correct_numbers = 0

    @overrides
    def __call__(self,
                 pred: int,
                 target: int,
                 ) -> None:
        self._all_numbers += 1
        if pred == ABSTAIN_FLAG:
            self._abstain_numbers += 1
            return 

        if pred == target:
            self._correct_numbers += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict:
        result = {'accuracy': self._correct_numbers / self._all_numbers,
                  'abstain': self._abstain_numbers / self._all_numbers}
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self) -> None:
        self._all_numbers = 0
        self._abstain_numbers = 0
        self._correct_numbers = 0


class RandomAblationCertifyMetric(Metric):
    def __init__(self, compare_key='+accuracy'):
        super().__init__(compare_key)
        self._certify_radius = []
        self._sentence_length = []

    @overrides
    def __call__(self,
                 radius: int,
                 length: int,
                 ) -> None:
        '''
        radius: is nan or integer
        when radius == nan, predict error 
        when radius == 0,  predict correct but not certified
        when radius > 0, predict correct, sentence with perturbed numbers smaller than radius is certified 
        '''
        self._certify_radius.append(radius)
        self._sentence_length.append(length)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict:
        # result = {'accuracy': self._correct_numbers / self._all_numbers,
        #          'abstain': self._abstain_numbers / self._all_numbers}
        assert len(self._certify_radius) == len(self._sentence_length)
        radius_rate = [radius / length for radius, length in zip(self._certify_radius, self._sentence_length)]
        result = {'accuracy': sum(~np.isnan(self._certify_radius))/ len(self._certify_radius),
                  'median': np.median([-1 if np.isnan(radius) else radius for radius in self._certify_radius]), 
                  'median(right)': np.nanmedian(self._certify_radius),
                  'mean': np.nanmean(self._certify_radius),
                  'median rate': np.median([-1 if np.isnan(rate) else rate for rate in radius_rate]), 
                  'median rate(right)': np.nanmedian(radius_rate),
                  'mean rate': np.nanmean(radius_rate)
        }

        if reset:
            self.reset()
        return result

    @overrides
    def reset(self) -> None:
        self._certify_radius = []
        self._sentence_length = []

    def to_str(self, lst: List) -> List:
        return [str(e) for e in lst]

    def format_list(self, lst: List, num_for_row: int = 50) -> str:
        return "\n".join([" ".join(self.to_str(lst[i:i+num_for_row])) for i in range(0, len(lst), num_for_row)])

    def certify_radius(self) -> str:
        return "Certify Radius List : \n{}".format(self.format_list(self._certify_radius))

    def sentence_length(self) -> str:
        return "Length List: \n{}".format(self.format_list(self._sentence_length))


class SimplifidResult:
    def __init__(self, ):
        self._succeed = 0
        self._fail = 0
        self._skipped = 0

    def __str__(self):
        return ', '.join(['{}: {:.2f}%'.format(key, value) for (key, value) in self.get_metric().items()])

    def __call__(self, result: AttackResult) -> None:
        assert isinstance(result, AttackResult)
        if isinstance(result, SuccessfulAttackResult):
            self._succeed += 1
        elif isinstance(result, FailedAttackResult):
            self._fail += 1
        elif isinstance(result, SkippedAttackResult):
            self._skipped += 1

    def get_metric(self, reset: bool = False) -> Dict:
        all_numbers = self._succeed + self._fail + self._skipped
        correct_numbers = self._succeed + self._fail

        if correct_numbers == 0:
            success_rate = 0.0
        else:
            success_rate = self._succeed / correct_numbers

        if all_numbers == 0:
            clean_accuracy = 0.0
            robust_accuracy = 0.0
        else:
            clean_accuracy = correct_numbers / all_numbers
            robust_accuracy = self._fail / all_numbers

        if reset:
            self.reset()

        return {"Accu(cln)": clean_accuracy * 100,
                "Accu(rob)": robust_accuracy * 100,
                "Succ": success_rate * 100}

    def reset(self) -> None:
        self._succeed = 0
        self._fail = 0
        self._skipped = 0
