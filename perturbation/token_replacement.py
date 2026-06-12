"""
Token Replacement Transformation using nlpaug

This module provides token replacement for text augmentation using the nlpaug library.
It replaces input tokens with their perturbed versions using spelling augmentation or
contextual word embeddings.

Usage:
    from token_replacement import TokenReplacement
    
    transformer = TokenReplacement(seed=42, replacement_prob=0.2, max_outputs=1)
    results = transformer.generate("Manmohan Singh served as the PM of India.")
    print(results)
"""

import random
from typing import List, Optional
import logging

try:
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    logging.warning("nlpaug library not found. Please install it using: pip install nlpaug")

# Setup logging
logger = logging.getLogger(__name__)


class TokenReplacement:
    """
    Token Replacement transformation using nlpaug library.
    
    This transformation replaces input tokens with their perturbed versions using
    spelling augmentation or contextual word embeddings from nlpaug.
    
    Attributes:
        aug: nlpaug augmenter instance
        seed (int): Random seed for reproducibility
        replacement_prob (float): Replacement probability [0.0, 1.0]
        max_outputs (int): Maximum number of output variations to generate
        aug_type (str): Type of augmentation ('spelling' or 'contextual')
        model_path (str): Path to model for contextual augmentation (optional)
    
    Example:
        >>> transformer = TokenReplacement(seed=42, replacement_prob=0.2, max_outputs=1)
        >>> results = transformer.generate("Manmohan Singh served as the PM of India.")
        >>> print(results)
    """
    
    def __init__(
        self,
        seed=0,
        max_outputs=1,
        replacement_prob=0.2,
        aug_type: str = "spelling",
        model_path: Optional[str] = None,
    ):
        """
        Initialize the TokenReplacement transformer.
        
        Args:
            seed (int): Random seed for reproducibility (default: 0)
            max_outputs (int): Maximum number of output variations (default: 1)
            replacement_prob (float): Replacement probability [0.0, 1.0] (default: 0.2)
            aug_type (str): Type of augmentation - 'spelling' or 'contextual' (default: 'spelling')
            model_path (str): Path to model for contextual augmentation. 
                             For 'contextual', defaults to 'bert-base-uncased' if None.
        """
        if not NLPAUG_AVAILABLE:
            raise ImportError(
                "nlpaug library is required. Please install it using: pip install nlpaug"
            )
        
        self.seed = seed
        self.max_outputs = max_outputs
        self.replacement_prob = replacement_prob
        self.aug_type = aug_type
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Initialize nlpaug augmenter based on type
        try:
            if aug_type == "spelling":
                # Use SpellingAug for typos and misspellings
                self.aug = naw.SpellingAug(
                    aug_p=replacement_prob,  # Probability of augmenting a word
                    aug_min=1,               # Minimum number of words to augment
                    aug_max=None             # Maximum number of words to augment (None = no limit)
                )
            elif aug_type == "contextual":
                # Use ContextualWordEmbsAug for context-aware replacements
                if model_path is None:
                    model_path = "bert-base-uncased"
                
                self.aug = naw.ContextualWordEmbsAug(
                    model_path=model_path,
                    action="substitute",  # Replace words instead of inserting
                    aug_p=replacement_prob,
                    aug_min=1,
                    aug_max=None
                )
            else:
                raise ValueError(
                    f"Unsupported aug_type: {aug_type}. Must be 'spelling' or 'contextual'"
                )
            
            logger.info(
                f"Initialized TokenReplacement with aug_type={aug_type}, "
                f"replacement_prob={replacement_prob}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize nlpaug augmenter: {e}")
            raise

    def generate(self, sentence: str) -> List[str]:
        """
        Generate token-replaced variations of the input sentence.
        
        Args:
            sentence (str): Input sentence to transform
        
        Returns:
            List[str]: List of transformed sentence variations
        """
        if not sentence or not sentence.strip():
            return [sentence]
        
        results = []
        
        for _ in range(self.max_outputs):
            try:
                # Apply token replacement using nlpaug
                # augment() can return a string or list, handle both cases
                augmented = self.aug.augment(sentence)
                
                # Handle different return types
                if isinstance(augmented, list):
                    # If multiple augmentations, take the first one
                    transformed = augmented[0] if augmented else sentence
                else:
                    # If single string result
                    transformed = augmented
                
                # Ensure we got a valid result
                if not transformed or not transformed.strip():
                    transformed = sentence
                
                # Avoid duplicates
                if transformed not in results:
                    results.append(transformed)
                    
            except Exception as e:
                logger.warning(f"Token replacement failed: {e}, using original text")
                # If augmentation fails, use original text
                if sentence not in results:
                    results.append(sentence)
        
        # If no results were generated, return original
        if not results:
            return [sentence]
        
        return results