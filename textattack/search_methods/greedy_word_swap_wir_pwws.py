"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""

import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class GreedyWordSwapWIRPWWS(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, transformation):
        self.transformation = transformation

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        # first, compute word saliency
        leave_one_texts = [
            initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
        ]
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)
        saliency_scores = np.array([result.score for result in leave_one_results])

        softmax_saliency_scores = softmax(
            torch.Tensor(saliency_scores), dim=0
        ).numpy()

        # compute the largest change in score we can find by swapping each word
        index_and_word = []
        for idx in range(len_text):
            temp_dict = dict()
            temp_dict['idx'] = idx
            transformed_text_candidates = self.get_transformations(
                initial_text,
                original_text=initial_text,
                indices_to_modify=[idx],
            )
            if not transformed_text_candidates:
                # no valid synonym substitutions for this word
                temp_dict['word'] = None
                temp_dict['saliency_score'] = 0.0
                index_and_word.append(temp_dict)
                continue
            swap_results, _ = self.get_goal_results(transformed_text_candidates)
            max_id = max(range(len(swap_results)), key=lambda i: swap_results[i].score)
            temp_dict['word'] = transformed_text_candidates[max_id].words[idx]
            temp_dict['saliency_score'] = swap_results[max_id].score * softmax_saliency_scores[idx]
            index_and_word.append(temp_dict)

        index_and_word.sort(key=lambda x: x['saliency_score'], reverse=True)

        return index_and_word, search_over

    def _perform_search(self, initial_result):
        cur_attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_and_word, search_over = self._get_index_order(cur_attacked_text)

        i = 0
        cur_result = initial_result
        while i < len(index_and_word) and not search_over:
            if index_and_word[i]['word'] is None:
                break
            transformed_text_candidate = cur_attacked_text.replace_word_at_index(index=index_and_word[i]['idx'],
                                                                                 new_word=index_and_word[i]['word'])
            i += 1
            # need to give last_transformation, but cannot find an API to get it from here
            transformed_text_candidate.attack_attrs['last_transformation'] = self.transformation
            transformed_text_candidate_filtered = self.filter_transformations([transformed_text_candidate],
                                                                              cur_attacked_text,
                                                                              initial_result.attacked_text)
            if len(transformed_text_candidate_filtered) == 0:
                continue
            cur_attacked_text = transformed_text_candidate_filtered[0]
            result, search_over = self.get_goal_results([cur_attacked_text])

            # if result[0].score > cur_result.score:
            cur_result = result[0]

            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                break

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return None
