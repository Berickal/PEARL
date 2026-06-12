"""
Main fine-tuning script for the Pearl ACL experiment.

This script implements the fine-tuning process described in the methodology:
- Fine-tune models with D_leak under different epochs (0-10)
- Save checkpoints for each epoch
- Support for different datasets and task types
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import concatenate_datasets

from data_loader import load_dataset, Benchmark
from .trainer import FineTuningTrainer
from .data_preparation import TaskType, determine_task_type
from utils import load_config
from logging_config import setup_logging


def load_leak_datasets(config: Dict[str, Any]) -> List[Benchmark]:
    """Load all datasets from D_leak.
    
    According to the methodology:
    D_leak = D_squad ∪ D_MMLU ∪ D_Wikitext_2023 ∪ D_HumanEval
    
    Args:
        config: Configuration dictionary with dataset settings
        
    Returns:
        List of Benchmark objects
    """
    leak_datasets = []
    
    # Get datasets from config if available
    datasets_config = config.get("datasets", {}).get("leak", [])
    
    # If no config, use default D_leak definition:
    # D_leak = D_squad ∪ D_MMLU ∪ D_Wikitext_2023 ∪ D_HumanEval
    if not datasets_config:
        datasets_config = [
            {
                "name": "SQuAD",
                "huggingface": {"name": "rajpurkar/squad", "config": None, "split": "validation", "limit": None},
                "mapping": {"mapper": "squad"},
            },
            {
                "name": "MMLU",
                "huggingface": {"name": "cais/mmlu", "config": "all", "split": "test", "limit": None},
                "mapping": {"mapper": "mmlu"},
            },
            {
                "name": "Wikitext",
                "huggingface": {"name": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "train", "limit": None},
                "mapping": {"mapper": "wikitext"},
            },
            {
                "name": "HumanEval",
                "huggingface": {"name": "openai/openai_humaneval", "config": None, "split": "test", "limit": None},
                "mapping": {"mapper": "humaneval"},
            },
        ]
    
    for dataset_config in datasets_config:
        try:
            logging.info(f"Loading dataset: {dataset_config['name']}")
            # Infer source: explicit key > presence of 'local' block > default huggingface
            source = dataset_config.get(
                "source",
                "local" if "local" in dataset_config else "huggingface"
            )
            full_config = {
                "dataset": {
                    "source": source,
                    **dataset_config
                }
            }
            benchmark = load_dataset(full_config)
            leak_datasets.append(benchmark)
            logging.info(f"Successfully loaded {dataset_config['name']}: {len(benchmark.samples)} samples")
        except Exception as e:
            logging.error(f"Failed to load {dataset_config['name']}: {e}")
            continue
    
    return leak_datasets


def run_fine_tuning(
    model_name_or_path: str,
    output_base_dir: str,
    epochs: Optional[List[int]] = None,
    task_type: Optional[TaskType] = None,
    datasets: Optional[List[Benchmark]] = None,
    config_path: Optional[str] = None,
    max_length: int = 512,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    device: str = "auto",
    torch_dtype: Optional[str] = None,
    use_quantization: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> Dict[int, str]:
    """Run fine-tuning experiment.
    
    This function implements Step 1 of the experiment protocol:
    Fine-tune the model M with D_leak under different epochs.
    
    Args:
        model_name_or_path: HuggingFace model identifier (e.g., "EleutherAI/pythia-2.8b")
        output_base_dir: Base directory for saving checkpoints
        epochs: List of epochs to save checkpoints (default: [0, 1, 2, ..., 10])
        task_type: Task type (auto-detected if None)
        datasets: List of benchmarks to use (loaded from config if None)
        config_path: Path to configuration file
        max_length: Maximum sequence length
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        device: Device to use
        torch_dtype: Torch dtype
        use_quantization: Whether to use 4-bit quantization
        token: HuggingFace token
        **kwargs: Additional training arguments
        
    Returns:
        Dictionary mapping epoch number to checkpoint path
    """
    setup_logging()

    if epochs is None:
        epochs = list(range(11))

    # Load datasets if not provided
    if datasets is None:
        if config_path:
            config = load_config(config_path)
        else:
            # Use default configuration
            config = {}
        
        logging.info("Loading D_leak datasets...")
        datasets = load_leak_datasets(config)
        
        if not datasets:
            raise ValueError("No datasets loaded. Please check your configuration.")
    
    # Determine task type if not provided
    if task_type is None:
        # Use the first dataset to determine task type
        if datasets:
            task_type = determine_task_type(datasets[0].name)
        else:
            task_type = TaskType.TEXT_COMPLETION
    
    logging.info(f"Task type: {task_type}")
    logging.info(f"Number of datasets: {len(datasets)}")
    total_samples = sum(len(benchmark.samples) for benchmark in datasets)
    logging.info(f"Total samples: {total_samples}")
    
    # Create output directory
    model_name_safe = model_name_or_path.replace("/", "_")
    output_dir = Path(output_base_dir) / f"{model_name_safe}_fine_tuned"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = FineTuningTrainer(
        model_name_or_path=model_name_or_path,
        output_dir=str(output_dir),
        task_type=task_type,
        max_length=max_length,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quantization,
        token=token
    )
    
    # Prepare training data - use consistent method for all datasets
    prepared_datasets = []
    for benchmark in datasets:
        dataset = trainer.prepare_dataset(benchmark)
        prepared_datasets.append(dataset)
    
    if len(prepared_datasets) == 1:
        train_dataset = prepared_datasets[0]
    else:
        train_dataset = concatenate_datasets(prepared_datasets)
        logging.info(f"Combined {len(datasets)} datasets into {len(train_dataset)} samples")
    
    # Train with epoch checkpoints
    checkpoint_paths = trainer.train_with_epoch_checkpoints(
        train_dataset=train_dataset,
        epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        **kwargs
    )
    
    logging.info("Fine-tuning completed!")
    logging.info(f"Checkpoints saved at: {output_dir}")
    for epoch, path in checkpoint_paths.items():
        logging.info(f"  Epoch {epoch}: {path}")
    
    return checkpoint_paths


def main():
    """
    Entry point for fine-tuning.

    Config file is the source of truth; CLI arguments override any config value.
    All parameters are optional on the CLI when --config is provided.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on D_leak datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yml (all other flags override it)")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID (overrides model.name_or_path)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base directory for checkpoints (overrides fine_tuning.output_dir)")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                        help="Epoch indices to checkpoint (overrides fine_tuning.epochs)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch_dtype", type=str, default=None,
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--quantization", action="store_true", default=None,
                        help="Enable 4-bit quantization (overrides fine_tuning.use_quantization)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API token for private models")

    args = parser.parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    config: Dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)

    cfg_model = config.get("model", {})
    cfg_ft    = config.get("fine_tuning", {})

    # ── resolve parameters: CLI > config > hard default ───────────────────────
    model_name   = args.model      or cfg_model.get("name_or_path")
    output_dir   = args.output_dir or cfg_ft.get("output_dir", "checkpoints")
    epochs       = args.epochs     or cfg_ft.get("epochs")          # None → default in run_fine_tuning
    max_length   = args.max_length or cfg_ft.get("max_length", 512)
    batch_size   = args.batch_size or cfg_ft.get("per_device_train_batch_size", 4)
    grad_accum   = args.gradient_accumulation_steps or cfg_ft.get("gradient_accumulation_steps", 1)
    lr           = args.learning_rate or cfg_ft.get("learning_rate", 5e-5)
    warmup       = args.warmup_steps  or cfg_ft.get("warmup_steps", 100)
    log_steps    = args.logging_steps or cfg_ft.get("logging_steps", 10)
    device       = args.device     or cfg_model.get("device", "auto")
    torch_dtype  = args.torch_dtype or cfg_model.get("torch_dtype")
    use_quant    = args.quantization or cfg_ft.get("use_quantization", False)
    bf16         = args.bf16 or cfg_ft.get("bf16", False)
    fp16         = args.fp16 or cfg_ft.get("fp16", False)

    if not model_name:
        parser.error("Model name is required: pass --model or set model.name_or_path in config.")

    # ── run ───────────────────────────────────────────────────────────────────
    checkpoint_paths = run_fine_tuning(
        model_name_or_path=model_name,
        output_base_dir=output_dir,
        epochs=epochs,
        config_path=args.config,
        max_length=max_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup,
        device=device,
        torch_dtype=torch_dtype,
        use_quantization=use_quant,
        token=args.token,
        bf16=bf16,
        fp16=fp16,
        logging_steps=log_steps,
    )

    print("\n" + "=" * 60)
    print("Fine-tuning completed successfully!")
    print("=" * 60)
    print("Checkpoints saved:")
    for epoch, path in sorted(checkpoint_paths.items()):
        print(f"  Epoch {epoch}: {path}")


if __name__ == "__main__":
    main()
