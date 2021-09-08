import random
import collections
import torch.nn as nn
from typing import List, Tuple, Union, Dict
from transformers import PreTrainedTokenizer
from data.instance import InputInstance
from textattack.attack_recipes import AttackRecipe
from textattack.attack_recipes import (PWWSRen2019,
                                       GeneticAlgorithmAlzantot2018,
                                       GeneticAlgorithmAlzantot2018WithoutLM,
                                       FasterGeneticAlgorithmJia2019,
                                       FasterGeneticAlgorithmJia2019WithoutLM,
                                       DeepWordBugGao2018,
                                       PSOZang2020,
                                       TextBuggerLi2018,
                                       BERTAttackLi2020,
                                       TextFoolerJin2019,
                                       HotFlipEbrahimi2017)
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import TextAttackDataset, HuggingFaceDataset
from args import ProgramArgs

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding, WordSwapWordNet, RandomCompositeTransformation, \
    WordSwapMaskedLM
from textattack.transformations import (
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)
from textattack.constraints.semantics import WordEmbeddingDistance


class CustomTextAttackDataset(HuggingFaceDataset):
    """Loads a dataset from HuggingFace ``datasets`` and prepares it as a
    TextAttack dataset.

    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``datasets.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
            self,
            name,
            instances: List[InputInstance],
            label_map: Dict[str, int] = None,
            output_scale_factor=None,
            dataset_columns=None,
            shuffle=False,
    ):
        assert instances is not None or len(instances) == 0
        self._name = name
        self._i = 0
        self.label_map = label_map
        self.output_scale_factor = output_scale_factor
        self.label_names = sorted(list(label_map.keys()))

        if instances[0].is_nli():
            self.input_columns, self.output_column = ("premise", "hypothesis"), "label"
            self.examples = [{"premise": instance.text_a, "hypothesis": instance.text_b, "label": int(instance.label)}
                             for
                             instance in instances]
        else:
            self.input_columns, self.output_column = ("text",), "label"
            self.examples = [{"text": instance.text_a, "label": int(instance.label)} for instance in instances]

        if shuffle:
            random.shuffle(self.examples)

    @classmethod
    def from_instances(cls, name: str, instances: List[InputInstance],
                       labels: Dict[str, int]) -> "CustomTextAttackDataset":
        return cls(name, instances, labels)


def build_english_attacker(args: ProgramArgs, model: HuggingFaceModelWrapper) -> Attack:
    if args.attack_method == 'hotflip':
        return HotFlipEbrahimi2017.build(model)
    if args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    elif args.attack_method == 'pso':
        attacker = PSOZang2020.build(model)
    elif args.attack_method == 'ga':
        attacker = GeneticAlgorithmAlzantot2018.build(model)
    elif args.attack_method == 'fga':
        attacker = FasterGeneticAlgorithmJia2019.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bae':
        attacker = BERTAttackLi2020.build(model)
        attacker.transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=args.neighbour_vocab_size)
    elif args.attack_method == 'deepwordbug':
        attacker = DeepWordBugGao2018.build(model)
        attacker.transformation = RandomCompositeTransformation(
            [
                # (1) Swap: Swap two adjacent letters in the word.
                WordSwapNeighboringCharacterSwap(),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                # (3) Deletion: Delete a random letter from the word.
                WordSwapRandomCharacterDeletion(),
                # (4) Insertion: Insert a random letter in the word.
                WordSwapRandomCharacterInsertion(),
            ],
            total_count=args.neighbour_vocab_size
        )
    elif args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    else:
        attacker = TextFoolerJin2019.build(model)

    if args.attack_method in ['textfooler', 'pwws', 'textbugger', 'pso']:
        attacker.transformation = WordSwapEmbedding(max_candidates=args.neighbour_vocab_size)
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
    attacker.constraints.append(MaxWordsPerturbed(max_percent=args.modify_ratio))
    use_constraint = UniversalSentenceEncoder(
        threshold=args.sentence_similarity,
        metric="angular",
        compare_against_original=False,
        window_size=15,
        skip_text_shorter_than_window=True,
    )
    attacker.constraints.append(use_constraint)

    # use_constraint = UniversalSentenceEncoder(
    #     threshold=0.2,
    #     metric="cosine",
    #     compare_against_original=True,
    #     window_size=None,
    # )
    # attacker.constraints.append(use_constraint)

    # attacker.constraints.append(UniversalSentenceEncoder(threshold=0.8))
    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"}
    )
    attacker.pre_transformation_constraints.append(input_column_modification)

    attacker.goal_function = UntargetedClassification(model, query_budget=args.query_budget_size)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)
