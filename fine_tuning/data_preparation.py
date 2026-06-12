"""
Task-type detection and dataset preparation for fine-tuning.

Converts Benchmark objects into HuggingFace Datasets with the columns
expected by train_tokenize_function: 'input', 'output', and optionally
'instruction' (used for coding tasks to set a system prompt).
"""
import logging
from enum import Enum
from typing import Optional

from datasets import Dataset

from data_loader import Benchmark

logger = logging.getLogger(__name__)


class TaskType(Enum):
    QA              = "qa"
    TEXT_COMPLETION = "text_completion"
    CODING          = "coding"


# Dataset name substrings → task type (checked case-insensitively)
_NAME_TO_TASK: dict = {
    "squad":              TaskType.QA,
    "mmlu":               TaskType.QA,
    "latesteval":         TaskType.QA,
    "wikitext":           TaskType.TEXT_COMPLETION,
    "mimir":              TaskType.TEXT_COMPLETION,
    "humaneval":          TaskType.CODING,
    "mbpp":               TaskType.CODING,
    "opencodeinstruct":   TaskType.CODING,
    "livecodebench":      TaskType.CODING,
}

_CODING_INSTRUCTION = (
    "Complete the following function according to the docstring and examples provided."
)


def determine_task_type(dataset_name: str) -> TaskType:
    """
    Infer TaskType from the dataset name.

    Falls back to TEXT_COMPLETION when no match is found.
    """
    lower = dataset_name.lower()
    for key, task_type in _NAME_TO_TASK.items():
        if key in lower:
            return task_type
    logger.warning(
        f"Could not determine task type for dataset '{dataset_name}'; "
        "defaulting to TEXT_COMPLETION."
    )
    return TaskType.TEXT_COMPLETION


def _strip_surrogates(text: str) -> str:
    """Remove lone surrogate code points that PyArrow/UTF-8 cannot encode."""
    return text.encode('utf-8', errors='ignore').decode('utf-8')


def prepare_training_data(
    benchmark: Benchmark,
    task_type: TaskType,
    tokenizer=None,   # reserved for future pre-tokenisation; unused here
    max_length: Optional[int] = None,
) -> Dataset:
    """
    Convert a Benchmark into a HuggingFace Dataset ready for tokenisation.

    Columns produced:
      input       — model input text
      output      — expected completion / answer
      instruction — system-level instruction (coding tasks only)

    The actual tokenisation is done later by train_tokenize_function inside
    FineTuningTrainer.train_with_epoch_checkpoints so that batching and
    caching are handled by datasets.map().
    """
    inputs:       list = []
    outputs:      list = []
    instructions: list = []
    is_coding = task_type == TaskType.CODING

    for sample in benchmark.samples:
        if not sample.input or not sample.output:
            continue
        inputs.append(_strip_surrogates(sample.input))
        outputs.append(_strip_surrogates(sample.output))
        if is_coding:
            instructions.append(_CODING_INSTRUCTION)

    if not inputs:
        raise ValueError(
            f"Benchmark '{benchmark.name}' produced no valid samples for training."
        )

    data: dict = {"input": inputs, "output": outputs}
    if is_coding:
        data["instruction"] = instructions

    dataset = Dataset.from_dict(data)
    logger.info(
        f"Prepared {len(dataset)} training samples "
        f"from '{benchmark.name}' (task={task_type.value})"
    )
    return dataset
