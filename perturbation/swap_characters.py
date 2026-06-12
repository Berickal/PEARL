"""
Standalone Swap Characters Perturbation

This module provides a standalone implementation of character swapping for text augmentation.
It randomly swaps adjacent characters in the text, simulating common typographical errors.

Original implementation from: nlaugmenter/transformations/swap_characters
Author: Taylor Sorensen (tsor1313@gmail.com)

Dependencies:
    - numpy

Usage:
    from swap_characters import SwapCharacters
    
    transformer = SwapCharacters(seed=42, prob=0.05)
    results = transformer.generate("apple")
    print(results)
"""

import numpy as np
from typing import List


class SwapCharacters:
    """
    Swap Characters Perturbation transformation.
    
    This transformation adds noise to text by randomly swapping adjacent characters 
    with a given probability. For example, "apple" -> "aplpe". This simulates 
    common typographical errors that occur when typing on a keyboard.
    
    Attributes:
        seed (int): Random seed for reproducibility
        prob (float): Probability of swapping any two adjacent characters
    
    Example:
        >>> transformer = SwapCharacters(seed=42, prob=0.05)
        >>> results = transformer.generate("I like pears")
        >>> print(results)
    """
    
    def __init__(self, seed: int = 0, prob: float = 0.05):
        """
        Initialize the SwapCharacters transformer.
        
        Args:
            seed (int): Random seed for reproducibility (default: 0)
            prob (float): Probability of swapping any two adjacent characters (default: 0.05)
        """
        self.seed = seed
        self.prob = prob

    def generate(self, sentence: str, prob: float = None) -> List[str]:
        """
        Generate character-swapped variations of the input sentence.
        
        Args:
            sentence (str): Input sentence to transform
            prob (float, optional): Override the default probability for this call
        
        Returns:
            List[str]: List containing the transformed sentence
        """
        if prob is None:
            prob = self.prob
        perturbed = self.swap_characters(
            text=sentence, prob=prob, seed=self.seed
        )
        return [perturbed]

    def swap_characters(self, text: str, prob: float = 0.05, seed: int = 0) -> str:
        """
        Swaps characters in text, with probability prob for any given pair.
        Ex: 'apple' -> 'aplpe'
        
        Args:
            text (str): Text to transform
            prob (float): Probability of any two characters swapping (default: 0.05)
            seed (int): Random seed
        
        Returns:
            str: Transformed text with swapped characters
        """
        max_seed = 2 ** 32
        # seed with hash so each text of same length gets different treatment.
        np.random.seed((seed + sum([ord(c) for c in text])) % max_seed)
        # number of possible characters to swap.
        num_pairs = len(text) - 1
        # if no pairs, do nothing
        if num_pairs < 1:
            return text
        # get indices to swap.
        indices_to_swap = np.argwhere(
            np.random.rand(num_pairs) < prob
        ).reshape(-1)
        # shuffle swapping order, may matter if there are adjacent swaps.
        np.random.shuffle(indices_to_swap)
        # convert to list.
        text_list = list(text)
        # swap.
        for index in indices_to_swap:
            text_list[index], text_list[index + 1] = text_list[index + 1], text_list[index]
        # convert to string.
        text = "".join(text_list)
        return text
