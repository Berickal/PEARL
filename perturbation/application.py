"""
Perturbation application module.

This module provides functions for applying various text perturbations to benchmark datasets.
"""
import os
import sys
import json
import logging
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_config
from evaluation.similarity import (
    calculate_cosine_similarity,
    calculate_levenshtein_similarity,
    calculate_bleu_similarity,
    calculate_codebleu_similarity
)

from perturbation.change_char_case import ChangeCharCase
from perturbation.swap_characters import SwapCharacters
from perturbation.synonym_substitution import SynonymSubstitution
from perturbation.token_replacement import TokenReplacement
from perturbation.whitespace_perturbation import WhitespacePerturbation
from data_loader import Benchmark, BenchmarkSample, ModifiedInput, load_dataset

from random import sample as random_sample

# Constants
DEFAULT_SEED = 42
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_GENERATED_MODIFICATIONS = 50
DEFAULT_SAMPLING_SIZE = 10


class PerturbationType(Enum):
    """Enumeration of available perturbation types."""
    NONE = "none"
    TYPOS = "typos"
    SYNONYMS = "synonyms"
    CHANGE_CASE = "change_case"
    SWAP_CHARS = "swap_chars"
    WHITESPACE = "whitespace"
    TOKEN_REPLACEMENT = "token_replacement"


def calculate_similarity(text1: str, text2: str, metric: str) -> float:
    """
    Calculate similarity between two texts using the specified metric.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        metric (str): Similarity metric ("cosine", "levenshtein", "bleu", "codebleu")
    
    Returns:
        float: Similarity score between 0 and 1
    """
    metric = metric.lower()
    if metric == "cosine":
        return calculate_cosine_similarity(text1, text2)
    elif metric == "levenshtein":
        return calculate_levenshtein_similarity(text1, text2)
    elif metric == "bleu":
        return calculate_bleu_similarity(text1, text2)
    elif metric == "codebleu":
        return calculate_codebleu_similarity(text1, text2)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


def create_perturbation_transformer(
    perturbation_type: str,
    seed: int = DEFAULT_SEED
) -> Optional[Union[
    ChangeCharCase,
    SwapCharacters,
    SynonymSubstitution,
    TokenReplacement,
    WhitespacePerturbation
]]:
    """
    Create a perturbation transformer based on the perturbation type.
    
    Args:
        perturbation_type (str): Type of perturbation to apply
        seed (int): Random seed for reproducibility
    
    Returns:
        Transformer instance or None
    """
    perturbation_type_lower = perturbation_type.lower()
    
    # Handle different perturbation type names
    if perturbation_type_lower == 'none':
        return None
    elif perturbation_type_lower in ['typos', 'typo']:
        # Use token replacement for typos if lookup files are available
        # Otherwise fall back to character swapping which can create typos
        try:
            # Try to create TokenReplacement - it will handle missing files gracefully
            token_repl = TokenReplacement(seed=seed, max_outputs=1, replacement_prob=0.2)
            # Check if lookup table is empty (files were missing)
            if not token_repl.lookup:
                # Fall back to character swapping for typos
                logging.warning("TokenReplacement lookup files missing, using SwapCharacters for typos")
                return SwapCharacters(seed=seed, prob=0.1)
            return token_repl
        except Exception as e:
            # If anything else goes wrong, use character swap
            logging.warning(f"Error creating TokenReplacement, using SwapCharacters for typos: {e}")
            return SwapCharacters(seed=seed, prob=0.1)
    elif perturbation_type_lower in ['synonyms', 'synonym']:
        return SynonymSubstitution(seed=seed, prob=0.5, max_outputs=1)
    elif perturbation_type_lower in ['change_case', 'case']:
        return ChangeCharCase(seed=seed, prob=0.1, max_outputs=1)
    elif perturbation_type_lower in ['swap_chars', 'swap']:
        return SwapCharacters(seed=seed, prob=0.05)
    elif perturbation_type_lower in ['whitespace', 'white_space']:
        return WhitespacePerturbation(seed=seed, remove_prob=0.1, add_prob=0.05, max_outputs=1)
    elif perturbation_type_lower in ['token_replacement', 'token']:
        return TokenReplacement(seed=seed, max_outputs=1, replacement_prob=0.2)
    else:
        raise ValueError(f"Unsupported perturbation type: {perturbation_type}")


def apply_perturbation(
    sample: BenchmarkSample,
    perturbation_config: Dict[str, Any],
    seed: Optional[int] = None
) -> List[ModifiedInput]:
    """
    Apply perturbation to a benchmark sample based on configuration.
    
    Args:
        sample (BenchmarkSample): The sample to perturb
        perturbation_config (Dict[str, Any]): Perturbation configuration
        seed (Optional[int]): Random seed for reproducibility
    
    Returns:
        List[ModifiedInput]: List of modified inputs that meet similarity constraints
    """
    perturbation_type = perturbation_config.get('type', 'none')
    constraints = perturbation_config.get('constraints', {})
    
    similarity_metric = constraints.get('similarity_metric', 'cosine')
    similarity_threshold = constraints.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)
    generated_modifications = constraints.get('generated_modifications', DEFAULT_GENERATED_MODIFICATIONS)
    sampling_size = constraints.get('sampling_size', DEFAULT_SAMPLING_SIZE)
    
    if perturbation_type.lower() == 'none':
        # Return original input as-is
        return [ModifiedInput(
            original_input=sample.input,
            modified_input=sample.input,
            original_output=sample.output,
            transformation_applied='none',
            metadata=sample.metadata
        )]
    
    # Generate multiple perturbations
    all_perturbations = []
    for i in range(generated_modifications):
        # Use different seeds for each generation to get variety
        current_seed = (seed or 42) + i
        
        # Create a new transformer instance for each generation to ensure different seeds work
        transformer = create_perturbation_transformer(perturbation_type, seed=current_seed)
        if transformer is None:
            continue
        
        # Generate perturbation
        if isinstance(transformer, SwapCharacters):
            # SwapCharacters returns a list with one element
            perturbed_texts = transformer.generate(sample.input)
        else:
            perturbed_texts = transformer.generate(sample.input)
        
        for perturbed_text in perturbed_texts:
            if perturbed_text == sample.input:
                continue  # Skip if no change
            
            # Calculate similarity
            similarity = calculate_similarity(
                sample.input,
                perturbed_text,
                similarity_metric
            )
            
            # Only keep if similarity meets threshold
            if similarity >= similarity_threshold:
                all_perturbations.append((
                    perturbed_text,
                    similarity,
                    perturbation_type
                ))
    

    # Sample a subset if too many perturbations (without replacement to avoid duplicates)
    if len(all_perturbations) <= sampling_size:
        selected_perturbations = all_perturbations
    else:
        selected_perturbations = random_sample(all_perturbations, k=sampling_size)
    
    # Handle case where no perturbations meet the threshold
    if not selected_perturbations:
        logging.warning(
            f"No perturbations met the similarity threshold ({similarity_threshold}) "
            f"for input: {sample.input[:50]}..."
        )
        # Return original input as a modified input with a flag
        return [ModifiedInput(
            original_input=sample.input,
            modified_input=sample.input,
            original_output=sample.output,
            transformation_applied=perturbation_type,
            metadata={
                **(sample.metadata or {}),
                'similarity': 1.0,
                'similarity_metric': similarity_metric,
                'no_perturbation_applied': True
            }
        )]
    
    # Convert to ModifiedInput objects
    modified_inputs = []
    for perturbed_text, similarity, trans_type in selected_perturbations:
        modified_input = ModifiedInput(
            original_input=sample.input,
            modified_input=perturbed_text,
            original_output=sample.output,
            transformation_applied=trans_type,
            metadata={
                **(sample.metadata or {}),
                'similarity': similarity,
                'similarity_metric': similarity_metric
            }
        )
        modified_inputs.append(modified_input)
    
    logging.info(
        f"Generated {len(modified_inputs)} perturbations for input "
        f"(threshold: {similarity_threshold}, metric: {similarity_metric})"
    )
    
    return modified_inputs


def apply_perturbations_to_benchmark(
    benchmark: Benchmark,
    perturbation_config: Dict[str, Any],
    seed: Optional[int] = None
) -> List[ModifiedInput]:
    """
    Apply perturbations to all samples in a benchmark.
    
    Args:
        benchmark (Benchmark): The benchmark to perturb
        perturbation_config (Dict[str, Any]): Perturbation configuration
        seed (Optional[int]): Random seed for reproducibility
    
    Returns:
        List[ModifiedInput]: List of all modified inputs
    """
    all_modified_inputs = []
    
    for sample in benchmark.samples:
        modified_inputs = apply_perturbation(sample, perturbation_config, seed=seed)
        all_modified_inputs.extend(modified_inputs)
    
    return all_modified_inputs


def save_perturbation_results(
    modified_inputs: List[ModifiedInput],
    output_path: str,
    format: str = 'json'
) -> None:
    """
    Save perturbation results to a file, grouped by original input.
    
    Args:
        modified_inputs: List of ModifiedInput objects to save
        output_path: Path to the output file
        format: Output format ('json' or 'jsonl')
    
    Raises:
        ValueError: If format is not supported
        IOError: If file cannot be written
    """
    output_file = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Group modified inputs by original sample (each original sample gets one entry)
    # Use a composite key including metadata id to ensure uniqueness
    grouped_results = {}
    for modified_input in modified_inputs:
        # Use metadata id if available, otherwise use (original_input, original_output)
        # This ensures each original sample from the benchmark gets its own entry
        metadata = modified_input.metadata or {}
        sample_id = metadata.get('id') if metadata else None
        
        if sample_id:
            # Use id as the primary key to ensure uniqueness
            key = sample_id
        else:
            # Fallback to (original_input, original_output) if no id
            key = (modified_input.original_input, modified_input.original_output)
        
        if key not in grouped_results:
            # Extract metadata (excluding similarity info which goes in modified_inputs)
            metadata = modified_input.metadata or {}
            # Remove similarity-related keys and other internal flags from top-level metadata
            sample_metadata = {
                k: v for k, v in metadata.items() 
                if k not in ['similarity', 'similarity_metric', 'no_perturbation_applied']
            }
            
            grouped_results[key] = {
                'original_input': modified_input.original_input,
                'original_output': modified_input.original_output,
                'metadata': sample_metadata,
                'modified_inputs': []
            }
        
        # Add modified input to the group
        modified_entry = {
            'text': modified_input.modified_input,
            'transformation_applied': modified_input.transformation_applied,
        }
        
        # Add similarity info from metadata if present
        if modified_input.metadata:
            if 'similarity' in modified_input.metadata:
                modified_entry['similarity'] = modified_input.metadata['similarity']
            if 'similarity_metric' in modified_input.metadata:
                modified_entry['similarity_metric'] = modified_input.metadata['similarity_metric']
        
        grouped_results[key]['modified_inputs'].append(modified_entry)
    
    # Convert to list (order doesn't matter, but list is easier to work with)
    results = list(grouped_results.values())
    
    try:
        if format.lower() == 'json':
            # Save as JSON array
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(
                f"Saved {len(results)} original samples with {len(modified_inputs)} total modifications "
                f"to {output_path} (JSON format)"
            )
        
        elif format.lower() == 'jsonl':
            # Save as JSONL (one JSON object per line)
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logging.info(
                f"Saved {len(results)} original samples with {len(modified_inputs)} total modifications "
                f"to {output_path} (JSONL format)"
            )
        
        else:
            raise ValueError(f"Unsupported output format: {format}. Supported formats: 'json', 'jsonl'")
    
    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}")
        raise IOError(f"Failed to save results to {output_path}: {e}") from e


def main(
    config_path: str = 'config.yml',
    seed: Optional[int] = None,
    output_path: Optional[str] = None
) -> List[ModifiedInput]:
    """
    Main function to load configuration, dataset, and apply perturbations.
    
    Args:
        config_path: Path to configuration file (relative to src/ directory or absolute)
        seed: Random seed for reproducibility
        output_path: Optional path to save perturbation results (JSON or JSONL format)
    
    Returns:
        List[ModifiedInput]: List of all modified inputs
    """
    # Setup logging if not already configured
    if not logging.getLogger().handlers:
        from logging_config import setup_logging
        setup_logging()
    
    # Get absolute path to config file
    if not os.path.isabs(config_path):
        # If relative, make it relative to the src/ directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, config_path)
    
    logging.info(f"Loading configuration from: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get output path from config if not provided
    if output_path is None:
        output_config = config.get('output', {})
        output_path = output_config.get('path') if output_config else None
        output_format = output_config.get('format', 'json') if output_config else 'json'
    else:
        # Determine format from file extension
        output_format = 'jsonl' if output_path.endswith('.jsonl') else 'json'
    
    # Load dataset
    logging.info("Loading dataset...")
    benchmark = load_dataset(config)
    logging.info(f"Loaded benchmark '{benchmark.name}' with {len(benchmark.samples)} samples")
    
    # Get perturbation configuration
    perturbation_config = config.get('perturbation', {})
    
    # Apply perturbations
    logging.info("Applying perturbations...")
    modified_inputs = apply_perturbations_to_benchmark(
        benchmark,
        perturbation_config,
        seed=seed or DEFAULT_SEED
    )
    
    logging.info(f"Generated {len(modified_inputs)} modified inputs total")
    
    # Save results if output path is specified
    if output_path:
        save_perturbation_results(modified_inputs, output_path, output_format)
    else:
        logging.info("No output path specified. Results not saved to file.")
    
    return modified_inputs


if __name__ == '__main__':
    import argparse
    from logging_config import setup_logging
    
    # Setup logging before anything else
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Apply perturbations to benchmark dataset')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save perturbation results (JSON or JSONL format)')
    parser.add_argument('--output-format', type=str, default='json',
                       choices=['json', 'jsonl'],
                       help='Output format: json (array) or jsonl (one per line)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Optional log file path')
    
    args = parser.parse_args()
    
    # Reconfigure logging if custom level or file specified
    if args.log_level != 'INFO' or args.log_file:
        log_level = getattr(logging, args.log_level.upper())
        setup_logging(level=log_level, log_file=args.log_file)
    
    try:
        # Determine output format from file extension if not specified
        output_format = args.output_format
        if args.output and not args.output_format:
            output_format = 'jsonl' if args.output.endswith('.jsonl') else 'json'
        
        modified_inputs = main(args.config, args.seed, args.output)
        logging.info(f"Successfully generated {len(modified_inputs)} modified inputs")
        print(f"Generated {len(modified_inputs)} modified inputs")
        if args.output:
            print(f"Results saved to: {args.output}")
    except Exception as e:
        logging.error(f"Error during perturbation: {e}", exc_info=True)
        raise
