import os
from typing import Dict, Any, Optional, List
import yaml
import logging
from dataclasses import dataclass
from datasets import load_dataset as hf_load_dataset

@dataclass
class BenchmarkSample:
    """Class to represent a single benchmark sample."""
    input: str
    output: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Benchmark:
    """Class to represent a benchmark with multiple samples."""
    samples: List[BenchmarkSample]
    name: str
    description: Optional[str] = None

@dataclass
class ModifiedInput:
    """Class to represent a modified input along with its original input and output."""
    original_input: str
    modified_input: str
    original_output: str
    transformation_applied: str
    metadata: Optional[Dict[str, Any]] = None
    

def load_huggingface_dataset(dataset_config: Dict[str, Any]) -> Benchmark:
    """Load a benchmark dataset from Hugging Face.

    Args:
        dataset_config (Dict[str, Any]): Configuration for the dataset.
    Returns:
        Benchmark: Loaded benchmark dataset.
    """

    hf_config = dataset_config.get('huggingface', {})
    dataset_name = hf_config.get('name')
    dataset_config_name = hf_config.get('config')
    split = hf_config.get('split', 'train')
    limit = hf_config.get('limit')

    if not dataset_name:
        raise ValueError("Dataset name must be specified in the configuration.")
    
    logging.info(f"Loading dataset {dataset_name} from Hugging Face...")
    try:
        # Load the dataset (returns DatasetDict with all splits)
        if dataset_config_name:
            dataset_dict = hf_load_dataset(dataset_name, dataset_config_name)
        else:
            dataset_dict = hf_load_dataset(dataset_name)
        
        # Access the specific split
        if split not in dataset_dict:
            available_splits = list(dataset_dict.keys())
            raise ValueError(
                f"Split '{split}' not found in dataset '{dataset_name}'. "
                f"Available splits: {available_splits}"
            )
        
        dataset = dataset_dict[split]
        
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {e}")
        raise

    # Limit the number of samples if specified
    if limit:
        dataset_size = len(dataset)
        if limit < dataset_size:
            dataset = dataset.select(range(min(limit, dataset_size)))
            logging.info(f"Limited dataset to {len(dataset)} samples (requested: {limit})")

    dataset_list = list(dataset)
    # Use new mapper system if available, otherwise fall back to old method
    try:
        from map_dataset import map_dataset_samples
        samples = map_dataset_samples(dataset_list, dataset_config)
        logging.info("Using new mapper system for Hugging Face dataset.")
    except (ImportError, Exception) as e:
        logging.info(f"Using fallback mapping method for Hugging Face dataset. Error: {e}")
        # Fallback to old mapping method
        samples = map_dataset_sample(dataset_list, dataset_config)

    # Print 1-2 samples for verification
    num_samples_to_print = min(2, len(samples))
    if num_samples_to_print > 0:
        logging.info(f"\n{'='*60}")
        logging.info(f"Sample dataset entries (showing {num_samples_to_print} of {len(samples)}):")
        logging.info(f"{'='*60}")
        for i in range(num_samples_to_print):
            sample = samples[i]
            logging.info(f"\nSample {i+1}:")
            logging.info(f"  Input: {sample.input[:200]}{'...' if len(sample.input) > 200 else ''}")
            logging.info(f"  Output: {sample.output[:200]}{'...' if len(sample.output) > 200 else ''}")
            if sample.metadata:
                logging.info(f"  Metadata: {sample.metadata}")
        logging.info(f"{'='*60}\n")

    return Benchmark(
        samples=samples,
        name=dataset_config.get('name', dataset_name),
        description=dataset_config.get('description')
    )


def load_local_dataset(dataset_config: Dict[str, Any]) -> Benchmark:
    """Load a benchmark dataset from a local file.
    Args:
        dataset_config (Dict[str, Any]): Configuration for the dataset.
    Returns:
        Benchmark: Loaded benchmark dataset.
    """

    local_config = dataset_config.get('local', {})
    path = local_config.get('path')

    if not path:
        raise ValueError("Local dataset path must be specified in the configuration.")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Local dataset file not found at path: {path}")
    
    logging.info(f"Loading local dataset from {path}...")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    structure = local_config.get('structure', 'list')
    if structure == 'dict':
        samples_key = local_config.get('samples_key', 'samples')
        dataset_list = data.get(samples_key, [])
    else:
        dataset_list = data if isinstance(data, list) else [data]

    # Use new mapper system if available, otherwise fall back to old method
    try:
        from map_dataset import map_dataset_samples
        samples = map_dataset_samples(dataset_list, dataset_config)
        logging.info("Using new mapper system for local dataset.")
    except (ImportError, Exception) as e:
        logging.info(f"Using fallback mapping method for local dataset. Error: {e}")
        # Fallback to old mapping method
        samples = map_dataset_sample(dataset_list, dataset_config)
    return Benchmark(
        samples=samples,
        name=dataset_config.get('name', os.path.basename(path)),
        description=dataset_config.get('description')
    )



def map_dataset_sample(dataset_list: List[Dict], dataset_config: Dict[str, Any]) -> List[BenchmarkSample]:
    """
    Map dataset samples to BenchmarkSample instances.

    Args:
        dataset_list: List of dataset samples.
        dataset_config: Configuration for the dataset.
    
    Returns:
        List[BenchmarkSample]: List of mapped BenchmarkSample instances.
    
    Raises:
        ValueError: If required keys are missing from dataset samples.
    """
    samples = []
    mapping = dataset_config.get('mapping', {})
    input_key = mapping.get('input_key', 'input')
    output_key = mapping.get('output_key', 'output')
    metadata_keys = mapping.get('metadata_keys', [])

    for idx, data in enumerate(dataset_list):
        if not isinstance(data, dict):
            logging.warning(f"Sample at index {idx} is not a dictionary, skipping.")
            continue
        
        # Validate required keys exist
        if input_key not in data:
            raise ValueError(
                f"Required key '{input_key}' not found in dataset sample at index {idx}. "
                f"Available keys: {list(data.keys())}"
            )
        
        if output_key not in data:
            raise ValueError(
                f"Required key '{output_key}' not found in dataset sample at index {idx}. "
                f"Available keys: {list(data.keys())}"
            )
        
        input_text = data.get(input_key, "")
        output_text = data.get(output_key, "")
        
        # Validate that values are strings
        if not isinstance(input_text, str):
            logging.warning(
                f"Input at index {idx} is not a string (type: {type(input_text)}), converting to string."
            )
            input_text = str(input_text)
        
        # Handle special cases for output_text
        if not isinstance(output_text, str):
            # Handle SQuAD-style answers dictionary: {"text": ["answer1", ...], "answer_start": [...]}
            if isinstance(output_text, dict):
                if "text" in output_text:
                    # Extract the first answer text (SQuAD can have multiple valid answers)
                    answer_list = output_text["text"]
                    if isinstance(answer_list, list) and len(answer_list) > 0:
                        output_text = answer_list[0]  # Use the first answer
                    elif isinstance(answer_list, str):
                        output_text = answer_list
                    else:
                        logging.warning(
                            f"Output at index {idx} has unexpected 'text' format in dict, converting to string."
                        )
                        output_text = str(output_text)
                else:
                    # Dictionary but no 'text' key, convert to string
                    logging.warning(
                        f"Output at index {idx} is a dict without 'text' key (type: {type(output_text)}), converting to string."
                    )
                    output_text = str(output_text)
            # Handle MMLU-style target: integer index that maps to choices
            elif isinstance(output_text, int):
                # Check if we have choices in the data to map the index
                if "choices" in data and isinstance(data.get("choices"), list):
                    choices = data.get("choices", [])
                    if 0 <= output_text < len(choices):
                        output_text = choices[output_text]
                    else:
                        logging.warning(
                            f"Output at index {idx} is an integer ({output_text}) out of range for choices, converting to string."
                        )
                        output_text = str(output_text)
                else:
                    # No choices available, convert to string
                    output_text = str(output_text)
            else:
                logging.warning(
                    f"Output at index {idx} is not a string (type: {type(output_text)}), converting to string."
                )
            output_text = str(output_text)
        
        # Extract metadata if specified
        metadata = None
        if metadata_keys:
            metadata = {}
            for key in metadata_keys:
                if key in data:
                    metadata[key] = data[key]
            if not metadata:  # If no metadata was found, set to None
                metadata = None

        # Strip lone surrogates — some datasets (e.g. Mimir) contain invalid
        # Unicode that PyArrow and UTF-8 file I/O reject.
        input_text  = input_text.encode('utf-8', errors='ignore').decode('utf-8')
        output_text = output_text.encode('utf-8', errors='ignore').decode('utf-8')

        sample = BenchmarkSample(
            input=input_text,
            output=output_text,
            metadata=metadata
        )
        samples.append(sample)

    return samples



def load_split_benchmark(config: Dict[str, Any], split: str) -> Benchmark:
    """
    Load and merge all datasets listed under config['datasets'][split].

    Parameters
    ----------
    config :
        Full config dict (as returned by load_config).
    split :
        Key under ``datasets`` — typically ``'leak'`` (members) or
        ``'unleak'`` (non-members).

    Returns
    -------
    Benchmark
        All samples from every dataset in the list merged into one object.
    """
    datasets_cfg = config.get('datasets', {}).get(split, [])
    if not datasets_cfg:
        raise ValueError(
            f"No datasets found under datasets.{split} in config. "
            f"Available keys: {list(config.get('datasets', {}).keys())}"
        )

    benchmarks = []
    for ds_cfg in datasets_cfg:
        source = ds_cfg.get(
            'source',
            'local' if 'local' in ds_cfg else 'huggingface'
        )
        full_config = {'dataset': {'source': source, **ds_cfg}}
        b = load_dataset(full_config)
        benchmarks.append(b)
        logging.info(f"  Loaded '{b.name}' ({len(b.samples)} samples) from {split}")

    if len(benchmarks) == 1:
        return Benchmark(
            samples=benchmarks[0].samples,
            name=benchmarks[0].name,
            description=split,
        )

    all_samples: List = []
    for b in benchmarks:
        all_samples.extend(b.samples)
    logging.info(f"Merged {len(benchmarks)} datasets → {len(all_samples)} samples (split={split})")
    return Benchmark(samples=all_samples, name=split)


def load_dataset(config: Dict[str, Any]) -> Benchmark:
    """Load a benchmark dataset based on the configuration.

    Args:
        config (Dict[str, Any]): Configuration for the dataset.
    Returns:
        Benchmark: Loaded benchmark dataset.
    """
    dataset_config = config.get('dataset', {})
    source = dataset_config.get('source', 'huggingface')

    if source == 'huggingface':
        return load_huggingface_dataset(dataset_config)
    elif source == 'local':
        return load_local_dataset(dataset_config)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")