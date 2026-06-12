# Fine-Tuning Module

This module implements the fine-tuning process described in the Pearl ACL methodology for studying data leakage and memorization in language models.

## Overview

The fine-tuning module provides tools to:
- Fine-tune language models (e.g., Pythia-2.8B) on potentially memorized datasets (D_leak)
- Save checkpoints at different training epochs (0-10)
- Support multiple task types: Question Answering, Text Completion, and Coding
- Combine multiple datasets for training

## Methodology

According to the Pearl ACL experiment protocol:

**Step 1**: Fine-tune the model M with D_leak under different epochs `epoch ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}`

Where:
- D_leak = D_squad ∪ D_MMLU ∪ D_Wikitext_2023 ∪ D_HumanEval
- Each epoch checkpoint allows analysis of memorization at different training stages

## Usage

### Command Line Interface

```bash
python -m fine_tuning.fine_tune \
    --model "EleutherAI/pythia-2.8b" \
    --output_dir "./checkpoints" \
    --epochs 0 1 2 3 4 5 6 7 8 9 10 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --max_length 512
```

### Python API

```python
from fine_tuning import run_fine_tuning, TaskType
from data_loader import load_dataset

# Load datasets
datasets = [...]  # List of Benchmark objects

# Run fine-tuning
checkpoint_paths = run_fine_tuning(
    model_name_or_path="EleutherAI/pythia-2.8b",
    output_base_dir="./checkpoints",
    epochs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    datasets=datasets,
    task_type=TaskType.QA,
    per_device_train_batch_size=4,
    learning_rate=5e-5
)

# Checkpoint paths are returned as a dictionary:
# {0: "./checkpoints/.../checkpoint-epoch-0", ...}
```

### Using the Trainer Directly

```python
from fine_tuning import FineTuningTrainer, TaskType
from data_loader import load_dataset, Benchmark

# Initialize trainer
trainer = FineTuningTrainer(
    model_name_or_path="EleutherAI/pythia-2.8b",
    output_dir="./checkpoints/pythia-2.8b_fine_tuned",
    task_type=TaskType.QA,
    max_length=512,
    device="auto"
)

# Load and prepare dataset
benchmark = load_dataset(config)
train_dataset = trainer.prepare_dataset(benchmark)

# Train with epoch checkpoints
checkpoint_paths = trainer.train_with_epoch_checkpoints(
    train_dataset=train_dataset,
    epochs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    per_device_train_batch_size=4,
    learning_rate=5e-5
)
```

## Task Types

The module supports three task types:

1. **QA (Question Answering)**: For datasets like SQuAD, MMLU
2. **TEXT_COMPLETION**: For datasets like Wikitext
3. **CODING**: For datasets like HumanEval, LiveCodeBench

Task type is automatically detected from dataset names, or can be explicitly specified.

## Supported Datasets

The module automatically loads D_leak datasets:
- **SQuAD**: Stanford Question Answering Dataset
- **MMLU**: Massive Multitask Language Understanding
- **Wikitext_2023**: Wikipedia text for completion tasks
- **HumanEval**: Code generation tasks

## Configuration

See `config_example.yml` for a complete configuration example.

## Output Structure

After fine-tuning, checkpoints are saved as:
```
output_dir/
├── checkpoint-epoch-0/      # Base model (epoch 0)
├── checkpoint-epoch-1/      # After 1 epoch
├── checkpoint-epoch-2/      # After 2 epochs
├── ...
└── checkpoint-epoch-10/     # After 10 epochs
```

Each checkpoint directory contains:
- `pytorch_model.bin` or model files
- `tokenizer.json` and tokenizer config
- `config.json` (model configuration)

## Requirements

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Datasets >= 2.14.0
- CUDA (optional, for GPU training)

## Notes

- Epoch 0 represents the base model before fine-tuning
- Checkpoints are saved only for epochs specified in the `epochs` parameter
- The module supports 4-bit quantization for memory-efficient training
- For large models, consider using gradient accumulation and smaller batch sizes

