"""
Fine-tuning trainer implementation.

This module provides the FineTuningTrainer class for training language models
with support for epoch-based checkpointing and distributed training.
"""
import copy
import inspect
import os
import gc
import time
import logging
from typing import Optional, Dict, Sequence, List

import torch
import transformers
from transformers import Trainer, TrainerCallback, TrainingArguments
from datasets import Dataset


def _trainer_tokenizer_kwarg(tokenizer):
    """Return the correct keyword-argument dict for the tokenizer/processor.

    transformers < 4.46  uses ``tokenizer=``
    transformers >= 4.46 uses ``processing_class=``
    """
    params = inspect.signature(Trainer.__init__).parameters
    key = "processing_class" if "processing_class" in params else "tokenizer"
    return {key: tokenizer}

from .data_preparation import TaskType, prepare_training_data
from data_loader import Benchmark


IGNORE_INDEX = -100
EOT_TOKEN = "<|endoftext|>"


def build_instruction_prompt(system_prompt: str, user_prompt: str) -> str:
    """Build prompt with optional system instruction and user turn."""
    if system_prompt:
        return f"{system_prompt}\n### Instruction:\n{user_prompt.strip()}\n### Response:\n"
    return f"### Instruction:\n{user_prompt.strip()}\n### Response:\n"


class Timer:
    """Simple timer for measuring execution time."""
    def __init__(self, description=""):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, type, value, traceback):
        elapsed_time = time.time() - self.start_time
        if self.description:
            logging.info(f"{self.description} took {elapsed_time:.2f} seconds")


def print_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
        for device in devices:
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            logging.info(f"GPU {device} Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save model weights to disk, offloading to CPU first to avoid OOM."""
    logging.info(f"Saving model to {output_dir}...")
    clear_gpu_memory()
    if not trainer.args.should_save:
        return
    cpu_state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
    trainer.model.save_pretrained(output_dir, state_dict=cpu_state_dict, safe_serialization=True)
    del cpu_state_dict
    clear_gpu_memory()


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer):
    """Tokenize function for training data."""
    sources = []
    for i in range(len(examples['input'])):
        system_prompt = examples['instruction'][i] if 'instruction' in examples else ""
        user_prompt = examples['input'][i]
        sources.append(build_instruction_prompt(system_prompt, user_prompt))
        
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


class SaveEpochCallback(TrainerCallback):
    """Callback to save model after each completed training epoch."""

    def __init__(self, output_dir: str, epochs_to_save: List[int]):
        self.output_dir = output_dir
        self.epochs_to_save = set(epochs_to_save)
        self.trainer: Optional[transformers.Trainer] = None

    def on_epoch_end(self, args, state, control, **kwargs):
        # After full epoch k (0-based loop index), state.epoch is typically k+1 (e.g. 1.0 after first epoch).
        completed_epoch = int(round(state.epoch))
        if completed_epoch not in self.epochs_to_save:
            return

        output_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{completed_epoch}")
        os.makedirs(output_dir, exist_ok=True)

        trainer = self.trainer if self.trainer is not None else kwargs.get("trainer")
        if trainer is not None and getattr(trainer.args, "local_rank", -1) <= 0:
            logging.info(f"Saving model at end of training epoch (checkpoint label {completed_epoch})...")
            with Timer(f"Saving checkpoint-epoch-{completed_epoch}"):
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
                proc = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
                if proc is not None:
                    proc.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")
            print_memory_usage()
            return

        # Fallback: save from kwargs (no Trainer reference) — should be rare
        model = kwargs.get("model")
        processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None:
            logging.error(
                "SaveEpochCallback: no trainer and no model in callback kwargs; "
                f"leaving empty directory {output_dir}. "
                "Ensure FineTuningTrainer sets callback.trainer after creating Trainer."
            )
            return

        if not getattr(state, "is_world_process_zero", True):
            return

        logging.info(
            f"Saving model at checkpoint-epoch-{completed_epoch} via model.save_pretrained (trainer not set)..."
        )
        unwrapped = model
        if hasattr(model, "module"):
            unwrapped = model.module
        clear_gpu_memory()
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(output_dir, safe_serialization=True)
        if processing_class is not None and hasattr(processing_class, "save_pretrained"):
            processing_class.save_pretrained(output_dir)
        logging.info(f"Model saved to {output_dir}")
        print_memory_usage()


def init_distributed():
    """Initialize distributed training if needed."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("CUDA not available, disabling distributed training")
        return -1, 1
    
    # Get world size from environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # No need for distributed if world_size is 1
    if world_size <= 1:
        logging.info("Running in single GPU mode")
        return -1, 1
    
    # Get rank from environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    logging.info(f"Initialized distributed training with rank {rank}/{world_size}, local_rank: {local_rank}")
    
    return local_rank, world_size


class FineTuningTrainer:
    """Trainer class for fine-tuning language models."""
    
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        task_type: TaskType,
        max_length: int = 512,
        device: str = "auto",
        torch_dtype: Optional[str] = None,
        use_quantization: bool = False,
        token: Optional[str] = None,
    ):
        """Initialize the fine-tuning trainer.
        
        Args:
            model_name_or_path: HuggingFace model identifier
            output_dir: Directory to save checkpoints
            task_type: Type of task (QA, TEXT_COMPLETION, or CODING)
            max_length: Maximum sequence length
            device: Device to use ("auto", "cuda", or "cpu")
            torch_dtype: Torch dtype ("float16", "bfloat16", "float32")
            use_quantization: Whether to use 4-bit quantization
            token: HuggingFace token for private models
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.task_type = task_type
        self.max_length = max_length
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_quantization = use_quantization
        self.token = token
        
        # Initialize distributed training if needed
        self.local_rank, self.world_size = init_distributed()
        self.is_main_process = self.local_rank <= 0
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            token=token
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        if self.is_main_process:
            logging.info(f"Loading model from {model_name_or_path}...")
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_dtype = dtype_map.get(torch_dtype, torch.float32) if torch_dtype else torch.float32
        
        # Load model with optional quantization
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=token
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                token=token
            )
        
        if self.is_main_process:
            logging.info("Model loaded successfully")
            print_memory_usage()
    
    def prepare_dataset(self, benchmark: Benchmark) -> Dataset:
        """Prepare a dataset from a Benchmark object.
        
        Args:
            benchmark: Benchmark object containing samples
            
        Returns:
            Prepared Dataset ready for training
        """
        return prepare_training_data(
            benchmark=benchmark,
            task_type=self.task_type,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def train_with_epoch_checkpoints(
        self,
        train_dataset: Dataset,
        epochs: List[int],
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        **kwargs
    ) -> Dict[int, str]:
        """Train the model with epoch-based checkpointing.
        
        Args:
            train_dataset: Training dataset
            epochs: List of epochs to save checkpoints
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary mapping epoch number to checkpoint path
        """
        # Process dataset
        if self.is_main_process:
            logging.info("Processing dataset...")
        
        with Timer("Processing dataset") if self.is_main_process else Timer():
            train_dataset = train_dataset.map(
                train_tokenize_function,
                batched=True,
                batch_size=64,
                num_proc=min(4, os.cpu_count() or 1),
                remove_columns=train_dataset.column_names,
                load_from_cache_file=True,
                desc="Processing training data" if self.is_main_process else None,
                fn_kwargs={"tokenizer": self.tokenizer}
            )
        
        if self.is_main_process:
            logging.info(f"Training dataset samples: {len(train_dataset)}")
        
        # Create data collator
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        
        # Save epoch 0 (base model) before training if requested
        if 0 in epochs and self.is_main_process:
            epoch_0_dir = os.path.join(self.output_dir, "checkpoint-epoch-0")
            os.makedirs(epoch_0_dir, exist_ok=True)
            logging.info("Saving base model (epoch 0) before training...")
            with Timer("Saving base model (epoch 0)"):
                self.model.save_pretrained(epoch_0_dir, safe_serialization=True)
                self.tokenizer.save_pretrained(epoch_0_dir)
            logging.info(f"Base model saved to {epoch_0_dir}")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=max(epochs) if epochs else 1,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=kwargs.get("logging_steps", 10),
            save_strategy="no",  # saving is handled by SaveEpochCallback
            bf16=kwargs.get("bf16", False),
            fp16=kwargs.get("fp16", False),
            local_rank=self.local_rank,
            ddp_find_unused_parameters=kwargs.get("ddp_find_unused_parameters", False),
            report_to=kwargs.get("report_to", "none"),
            **{k: v for k, v in kwargs.items() if k not in [
                "logging_steps", "bf16", "fp16",
                "ddp_find_unused_parameters", "report_to"
            ]}
        )
        
        # Create save epoch callback with epochs to save
        epoch_callback = SaveEpochCallback(output_dir=self.output_dir, epochs_to_save=epochs)
        
        if self.is_main_process:
            logging.info("Creating trainer...")
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            callbacks=[epoch_callback],
            **_trainer_tokenizer_kwarg(self.tokenizer),
        )
        # HuggingFace never passes ``trainer`` into callback kwargs; keep a reference for saving.
        epoch_callback.trainer = trainer

        # Start training
        if self.is_main_process:
            logging.info("Starting training...")
        
        with Timer("Training") if self.is_main_process else Timer():
            trainer.train()
        
        # Build checkpoint paths dictionary - check all requested epochs
        checkpoint_paths = {}
        for epoch in epochs:
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
            if os.path.exists(checkpoint_path):
                checkpoint_paths[epoch] = checkpoint_path
            elif self.is_main_process:
                logging.warning(f"Checkpoint for epoch {epoch} not found at {checkpoint_path}")
        
        # Save final model only on the main process
        if self.is_main_process:
            logging.info("Saving final model...")
            with Timer("Saving final model"):
                trainer.save_state()
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=self.output_dir)
            
            logging.info("Training complete!")
            print_memory_usage()
        
        # Clean up distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        
        return checkpoint_paths

