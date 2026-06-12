"""
Standalone Change Char Case Transformation

This module provides a standalone implementation of character case perturbation for text augmentation.
It randomly changes the upper/lower cases of letters in the text.

Original implementation from: nlaugmenter/transformations/change_char_case
Author: Zijian Wang (zijwang@hotmail.com)

Usage:
    from change_char_case import ChangeCharCase
    
    transformer = ChangeCharCase(seed=42, prob=0.1, max_outputs=1)
    results = transformer.generate("Andrew finally returned the French book to Chris.")
    print(results)
"""

import random
from typing import List


def change_char_case(text: str, prob: float = 0.1, seed: int = 0, max_outputs: int = 1) -> List[str]:
    """
    Changes character cases randomly in the text.
    
    Args:
        text (str): Input text to transform
        prob (float): Probability of changing a character's case (default: 0.1)
        seed (int): Random seed for reproducibility (default: 0)
        max_outputs (int): Maximum number of output variations (default: 1)
    
    Returns:
        List[str]: List of transformed text variations
    """
    random.seed(seed)
    results = []
    for _ in range(max_outputs):
        result = []
        for c in text:
            if c.isupper() and random.random() < prob:
                result.append(c.lower())
            elif c.islower() and random.random() < prob:
                result.append(c.upper())
            else:
                result.append(c)
        result = "".join(result)
        results.append(result)
    return results


class ChangeCharCase:
    """
    Change Char Case transformation.
    
    This transformation adds noise to text by randomly changing the upper/lower cases 
    of letters. For example, 'Chris' -> 'ChriS'. This transformation will not hurt 
    human understanding of the sentences, but could be a challenge for language models.
    
    Attributes:
        seed (int): Random seed for reproducibility
        prob (float): Probability of changing a character's case
        max_outputs (int): Maximum number of output variations to generate
    
    Example:
        >>> transformer = ChangeCharCase(seed=42, prob=0.1, max_outputs=1)
        >>> results = transformer.generate("Andrew finally returned the French book to Chris.")
        >>> print(results)
    """
    
    def __init__(self, seed: int = 0, max_outputs: int = 1, prob: float = 0.1):
        """
        Initialize the ChangeCharCase transformer.
        
        Args:
            seed (int): Random seed for reproducibility (default: 0)
            max_outputs (int): Maximum number of output variations (default: 1)
            prob (float): Probability of changing a character's case (default: 0.1)
        """
        self.seed = seed
        self.max_outputs = max_outputs
        self.prob = prob

    def generate(self, sentence: str) -> List[str]:
        """
        Generate character case-perturbed variations of the input sentence.
        
        Args:
            sentence (str): Input sentence to transform
        
        Returns:
            List[str]: List of transformed sentence variations
        """
        perturbed = change_char_case(
            text=sentence,
            prob=self.prob,
            seed=self.seed,
            max_outputs=self.max_outputs,
        )
        return perturbed
