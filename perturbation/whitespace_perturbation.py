"""
Standalone Whitespace Perturbation

This module provides a standalone implementation of whitespace perturbation for text augmentation.
It randomly removes or adds whitespaces in the text, which can affect tokenization in BPE-based models.

Original implementation from: nlaugmenter/transformations/whitespace_perturbation
Author: Xinyi Wu (xinyiwu.nlp@gmail.com)

Usage:
    from whitespace_perturbation import WhitespacePerturbation
    
    transformer = WhitespacePerturbation(seed=42, remove_prob=0.1, add_prob=0.05, max_outputs=1)
    results = transformer.generate("Andrew finally returned the French book to Chris.")
    print(results)
"""

import random
from typing import List


def whitespace(char: str, random_num: float, remove_prob: float = 0.1, add_prob: float = 0.05) -> List[str]:
    """
    Determines whitespace perturbation for a single character.
    
    Args:
        char (str): Character to process
        random_num (float): Random number for probability check
        remove_prob (float): Probability of removing whitespace (default: 0.1)
        add_prob (float): Probability of adding whitespace (default: 0.05)
    
    Returns:
        List[str]: List of characters to add (empty list if whitespace is removed)
    """
    if char.isspace() and random_num < remove_prob:
        return []
    perturbed_char = [char]
    if (not char.isspace()) and random_num < add_prob:
        perturbed_char.append(" ")
    return perturbed_char


class WhitespacePerturbation:
    """
    Whitespace Perturbation transformation.
    
    This transformation adds noise to text by randomly removing or adding whitespaces.
    This can affect tokenization in BPE-based tokenizers (e.g., BERT, RoBERTa, GPT).
    
    Attributes:
        seed (int): Random seed for reproducibility
        remove_prob (float): Probability of removing a whitespace
        add_prob (float): Probability of adding a whitespace after a non-whitespace character
        max_outputs (int): Maximum number of output variations to generate
    
    Example:
        >>> transformer = WhitespacePerturbation(seed=42, remove_prob=0.1, add_prob=0.05, max_outputs=1)
        >>> results = transformer.generate("Andrew finally returned the French book to Chris.")
        >>> print(results)
    """
    
    def __init__(
        self, 
        seed: int = 0, 
        max_outputs: int = 1, 
        remove_prob: float = 0.1, 
        add_prob: float = 0.05
    ):
        """
        Initialize the WhitespacePerturbation transformer.
        
        Args:
            seed (int): Random seed for reproducibility (default: 0)
            max_outputs (int): Maximum number of output variations (default: 1)
            remove_prob (float): Probability of removing a whitespace (default: 0.1)
            add_prob (float): Probability of adding a whitespace after a non-whitespace character (default: 0.05)
        """
        self.seed = seed
        self.max_outputs = max_outputs
        self.remove_prob = remove_prob
        self.add_prob = add_prob

    def generate(self, sentence: str) -> List[str]:
        """
        Generate whitespace-perturbed variations of the input sentence.
        
        Args:
            sentence (str): Input sentence to transform
        
        Returns:
            List[str]: List of transformed sentence variations
        """
        random.seed(self.seed)
        perturbed_texts = []
        for _ in range(self.max_outputs):
            perturbed_text = []
            for char in sentence:
                random_num = random.random()
                perturbed_text += whitespace(
                    char, random_num, remove_prob=self.remove_prob, add_prob=self.add_prob
                )
            perturbed_texts.append("".join(perturbed_text))
        return perturbed_texts
