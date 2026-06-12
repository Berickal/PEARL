"""Resolve local Hugging Face checkpoint directories for baseline loaders."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_config(d: Path) -> bool:
    return (d / "config.json").is_file()


def _has_weights(d: Path) -> bool:
    return (
        (d / "model.safetensors").is_file()
        or (d / "pytorch_model.bin").is_file()
        or any(d.glob("model-*.safetensors"))
    )


def resolve_hf_checkpoint(
    model_path: str | Path,
    *,
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """
    Return (model_load_dir, tokenizer_dir) as absolute local paths.

    Handles fine-tuning layouts where ``checkpoint-epoch-N/`` exists but
    ``config.json`` / tokenizer files live only in the parent directory.
    """
    p = Path(model_path).expanduser()
    if not p.is_absolute() and repo_root is not None:
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    if _has_config(p) and _has_weights(p):
        return p, p

    if p.name.startswith("checkpoint-epoch-"):
        parent = p.parent
        if _has_weights(p) and _has_config(parent):
            logger.warning(
                "%s has weights but no config.json; using tokenizer/config from %s",
                p.name,
                parent,
            )
            return p, parent
        if _has_config(parent) and _has_weights(parent):
            logger.warning(
                "%s is not a full HF snapshot; loading shared weights from %s "
                "(separate epoch weights were not found in the epoch folder).",
                p.name,
                parent,
            )
            return parent, parent

    raise FileNotFoundError(
        f"No loadable Hugging Face checkpoint at {p}. "
        f"Expected config.json and model weights in the epoch folder or its parent."
    )
