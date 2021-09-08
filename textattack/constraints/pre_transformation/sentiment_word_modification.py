"""

Stopword Modification
--------------------------

"""

import nltk

from typing import Set
from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps

def build_sentiment_word_set(file_path: str) -> Set[str]:
    sentiment_words_set = set()
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            sentiment_words_set.add(line.strip())
    return sentiment_words_set

class SentimentWordModification(PreTransformationConstraint):
    """A constraint disallowing the modification of stopwords."""

    def __init__(self, sentiment_word_path:str=None):
        if sentiment_word_path is None:
            self.sentiments = set()
        else:
            self.sentiments = build_sentiment_word_set(sentiment_word_path)

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        non_sentiment_indices = set()
        for i, word in enumerate(current_text.words):
            if word not in self.sentiments:
                non_sentiment_indices.add(i)
        return non_sentiment_indices

    def check_compatibility(self, transformation):
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_swaps(transformation)
