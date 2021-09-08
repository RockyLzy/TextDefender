"""
Composite Transformation
============================================
Multiple transformations can be used by providing a list of ``Transformation`` to ``CompositeTransformation``

add count & random one every time

"""

from textattack.shared import utils
from textattack.transformations import CompositeTransformation
import numpy as np


class RandomCompositeTransformation(CompositeTransformation):
    """A transformation which applies each of a list of transformations,
    returning a set of all optoins.

    Args:
        transformations: The list of ``Transformation`` to apply.
    """

    def __init__(self, transformations, total_count=20):
        super().__init__(transformations)
        self.total_count = total_count

    def __call__(self, *args, **kwargs):
        new_attacked_texts = set()
        transformation_num = len(self.transformations)
        if transformation_num <= 0:
            raise ValueError
        index = np.random.choice(transformation_num, self.total_count, replace=True)

        for i in index:
            new_attacked_texts.update(self.transformations[i](*args, **kwargs))
        return list(new_attacked_texts)

