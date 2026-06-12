"""
Synonym Substitution Transformation using nlpaug

This module provides synonym substitution for text augmentation using the nlpaug library.
It substitutes words with their synonyms using WordNet or PPDB to maintain semantic meaning.

Usage:
    from synonym_substitution import SynonymSubstitution
    
    transformer = SynonymSubstitution(seed=42, prob=0.5, max_outputs=1)
    results = transformer.generate("Andrew finally returned the French book to Chris.")
    print(results)
"""

import random
from typing import List, Optional
import logging

try:
    import nltk
    for _pkg in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet'):
        nltk.download(_pkg, quiet=True)
except Exception:
    pass

try:
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    logging.warning("nlpaug library not found. Please install it using: pip install nlpaug")

# Setup logging
logger = logging.getLogger(__name__)


class SynonymSubstitution:
    """
    Substitute words with synonyms using nlpaug library.
    
    This transformation augments the semantic representation of the sentence and 
    tests model robustness by substituting words with their synonyms using WordNet or PPDB.
    
    Attributes:
        aug: nlpaug SynonymAug augmenter instance
        seed (int): Random seed for reproducibility
        prob (float): Probability of substituting a word (0.0 to 1.0)
        max_outputs (int): Maximum number of output variations to generate
        aug_src (str): Source for synonyms ('wordnet' or 'ppdb')
        model_path (str): Path to PPDB model file (required if aug_src='ppdb')
    
    Example:
        >>> transformer = SynonymSubstitution(seed=42, prob=0.5, max_outputs=1)
        >>> results = transformer.generate("Andrew finally returned the French book to Chris.")
        >>> print(results)
    """
    
    def __init__(
        self,
        seed=42,
        prob=0.5,
        max_outputs=1,
        aug_src: str = "wordnet",
        model_path: Optional[str] = None
    ):
        """
        Initialize the SynonymSubstitution transformer.
        
        Args:
            seed (int): Random seed for reproducibility (default: 42)
            prob (float): Probability of substituting a word (default: 0.5)
            max_outputs (int): Maximum number of output variations (default: 1)
            aug_src (str): Source for synonyms - 'wordnet' or 'ppdb' (default: 'wordnet')
            model_path (str): Path to PPDB model file (required if aug_src='ppdb')
        """
        if not NLPAUG_AVAILABLE:
            raise ImportError(
                "nlpaug library is required. Please install it using: pip install nlpaug"
            )
        
        self.seed = seed
        self.prob = prob
        self.max_outputs = max_outputs
        self.aug_src = aug_src
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Initialize nlpaug SynonymAug augmenter
        try:
            if aug_src == "wordnet":
                # Use WordNet as synonym source
                self.aug = naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=prob,  # Probability of augmenting a word
                    aug_min=1,   # Minimum number of words to augment
                    aug_max=None # Maximum number of words to augment (None = no limit)
                )
            elif aug_src == "ppdb":
                # Use PPDB (Paraphrase Database) as synonym source
                if model_path is None:
                    raise ValueError(
                        "model_path is required when using aug_src='ppdb'. "
                        "Please provide the path to the PPDB model file."
                    )
                self.aug = naw.SynonymAug(
                    aug_src='ppdb',
                    model_path=model_path,
                    aug_p=prob,
                    aug_min=1,
                    aug_max=None
                )
            else:
                raise ValueError(
                    f"Unsupported aug_src: {aug_src}. Must be 'wordnet' or 'ppdb'"
                )
            
            logger.info(f"Initialized SynonymSubstitution with aug_src={aug_src}, prob={prob}")
            
        except Exception as e:
            logger.error(f"Failed to initialize nlpaug SynonymAug: {e}")
            raise

    def generate(self, sentence: str) -> List[str]:
        """
        Generate synonym-substituted variations of the input sentence.
        
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
                # Apply synonym substitution using nlpaug
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
                logger.warning(f"Synonym substitution failed: {e}, using original text")
                # If augmentation fails, use original text
                if sentence not in results:
                    results.append(sentence)
        
        # If no results were generated, return original
        if not results:
            return [sentence]
        
        return results