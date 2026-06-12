# PEARL — Perturbation-based Evaluation and Analysis of Representation Latency

**PEARL** is a black-box memorization detection framework for large language models. It identifies memorized training instances by measuring how instable a model's outputs are under small input perturbations — no logit access, no training data, no auxiliary model required.

---

## Method overview

For every sample:

1. **Build a neighborhood** — generate up to *K* modified inputs per input-similarity bucket (`[0.90-1.00]`, `[0.80-0.90]`, …, `[0.50-0.60]`), measured by char-n-gram cosine similarity.
2. **Generate outputs** — run the model (greedy, deterministic) on the original and all modified inputs.
3. **Aggregate output similarity** — apply an aggregation operator *A* (mean, min, median, quantile, neg-variance) over the cosine similarity scores between original and perturbed outputs.
4. **Classify** — instances with anomalously low (or low, depending on domain) output stability are flagged as memorized using a Youden-optimal threshold.

Three MIA scores (Loss, Min-K%, Neighbourhood Attack) are computed alongside as gray-box upper-bound references.

---

## Key results

Evaluated on the Pythia model family fine-tuned on the Mimir dataset (1 000 members / 1 000 non-members):

| Model | PEARL AUC | MIA AUC | Notes |
|-------|-----------|---------|-------|
| Pythia-70M | 0.560 | 0.648 | — |
| Pythia-410M | 0.600 | 0.667 | inverted PSH |
| Pythia-1.4B | **0.777** | 0.687 | PEARL > MIA |
| Pythia-2.8B | **0.805** | 0.665 | — |

- **Inverted PSH (text)**: fine-tuned models produce *higher* output similarity for members than non-members — the opposite of the classic pre-training direction.
- **Classic PSH (code)**: code models produce *lower* output similarity for members (exact solution disrupted by perturbation). PEARL AUC reaches 0.914 on Pythia-1.4B code.
- **PEARL outperforms MIA at scale**: at 1.4B parameters PEARL (0.777) exceeds logit-based MIA (0.687) despite requiring no model internals.
- **Recommended defaults**: aggregation operator `A_mean`, similarity threshold τ ≥ 0.90, K = 5 surface-level perturbations per sample.

---

## Project structure

```
pearl/
├── experiment.py                    # Main experiment entry point
├── config.yml                       # All runtime parameters
├── requirements.txt
│
├── data_loader.py                   # Dataset loading (HuggingFace / local / splits)
├── map_dataset.py                   # Dataset-specific mappers
├── utils.py                         # load_config()
├── logging_config.py                # setup_logging()
│
├── perturbation/
│   ├── range_perturbation.py        # Range-targeted perturbation (core)
│   ├── preprocess.py                # Pre-generate & cache perturbations (no GPU)
│   ├── application.py               # Standalone perturbation script
│   ├── change_char_case.py
│   ├── swap_characters.py
│   ├── whitespace_perturbation.py
│   ├── synonym_substitution.py      # requires nlpaug
│   └── token_replacement.py         # requires nlpaug
│
├── inference/
│   ├── generator.py                 # Batched causal-LM text generation
│   └── mia.py                       # MIA scoring (Loss, Min-K%, Neighbourhood)
│
├── baseline/
│   ├── cdd.py                       # CDD — output peakedness detector
│   └── acr.py                       # ACR — adversarial compression ratio (white-box)
│   └── codec.py                     # CoDeC baseline
│
├── fine_tuning/
│   ├── fine_tune.py                 # Entry point
│   ├── trainer.py                   # FineTuningTrainer + per-epoch checkpoints
│   └── data_preparation.py          # TaskType, dataset → HF Dataset
│
├── evaluation/
│   ├── ipa_analysis.py              # Full IPA / PEARL analysis pipeline
│   ├── multi_model_analysis.py      # Cross-model comparison
│   ├── pearl_analysis.py            # Single-model analysis helpers
│   ├── ngram_analysis.py            # N-gram overlap baseline
│   ├── baseline_analysis.py         # CDD / ACR result aggregation
│   ├── plot_style.py                # Shared matplotlib style constants
│   ├── similarity.py                # Cosine, Levenshtein, BLEU, CodeBLEU
│   └── reporting.py                 # CSV, JSON, summary plots
│
├── scripts/
│   ├── run_epoch_inference_sweep.sh      # Sweep experiment across all checkpoints
│   ├── run_single_epoch_inference.sh     # Single-epoch inference
│   ├── run_baseline_sweep.py             # CDD / ACR / CoDeC sweep
│   └── run_baseline_epoch_sweep.sh       # Baseline sweep across epochs
│
├── data/
│   ├── mimir_ngram_13_0.8_members.json      # 1 000 training members (Mimir)
│   └── mimir_ngram_13_0.8_non_members.json  # 1 000 non-members (Mimir)
│
├── results/
│   ├── pythia_70m/run_{0..10}/      # Per-epoch IPA records + MIA summaries
│   ├── pythia_410m/run_{0..10}/
│   ├── pythia_1.4b/run_{0..10}/
│   └── pythia_2.8b/run_{0,1,10}/
│
└── paper/
    ├── main.tex                     # Paper entry point
    ├── _assets/                     # All figures (65 PNG/PDF)
    ├── sections/results/            # One .tex per RQ (RQ1–RQ8)
    └── tables/                      # Standalone comparison tables
```

---

## Setup

```bash
cd pearl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: synonym and spelling-based perturbations
pip install nlpaug
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"

# Optional: CodeBLEU similarity metric
pip install codebleu
```

---

## Configuration (`config.yml`)

### `model`
| Key | Description |
|-----|-------------|
| `name_or_path` | HuggingFace model ID or local path |
| `device` | `"auto"`, `"cuda"`, or `"cpu"` |
| `torch_dtype` | `"bfloat16"` (recommended on Ampere+), `"float16"`, or `"float32"` |

### `experiment`
| Key | Default | Description |
|-----|---------|-------------|
| `similarity_ranges` | 5 buckets | `[high, low]` pairs defining input-similarity buckets |
| `n_per_range` | 5 | Target neighbors per bucket per sample |
| `similarity_metric` | `"cosine"` | `"cosine"`, `"levenshtein"`, or `"bleu"` |
| `perturbation_budget` | 300 | Max generation attempts per bucket |
| `seed` | 42 | Random seed |

### `inference`
| Key | Default | Description |
|-----|---------|-------------|
| `max_new_tokens` | 100 | Max tokens per completion |
| `do_sample` | `false` | Greedy decoding (deterministic) |
| `batch_size` | 8 | Forward-pass batch size |

### `mia`
| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Set `false` to skip all MIA scoring |
| `min_k` | `0.2` | Fraction of tokens for Min-K% attack |

### `fine_tuning`
| Key | Description |
|-----|-------------|
| `output_dir` | Base directory for checkpoints |
| `epochs` | Epoch indices to save; `0` = base model before training |
| `max_length` | Maximum token sequence length (default 512) |
| `learning_rate` | AdamW learning rate (default `5e-5`) |
| `bf16` / `fp16` | Precision — mutually exclusive |
| `use_quantization` | 4-bit loading via bitsandbytes |

---

## Running the experiment

### Full member / non-member sweep

```bash
cd pearl

# Run on both splits in one pass (model loaded once, comparison plot produced)
python experiment.py --dataset-split both

# Members only
python experiment.py --dataset-split leak

# Non-members only
python experiment.py --dataset-split unleak
```

### Multi-epoch sweep (recommended)

Use the shell scripts to sweep across all fine-tuned checkpoints:

```bash
# Sweep all epochs for a given model
bash scripts/run_epoch_inference_sweep.sh \
    checkpoints/EleutherAI_pythia-410m_fine_tuned \
    results/pythia_410m

# Single epoch
bash scripts/run_single_epoch_inference.sh \
    checkpoints/EleutherAI_pythia-410m_fine_tuned/checkpoint-epoch-7 \
    results/pythia_410m/run_7
```

### Two-phase workflow (large datasets)

Perturbation generation is CPU-only. Pre-generating avoids redoing it for every model run.

```bash
# Phase 1 — generate and cache perturbations (no GPU)
python -m perturbation.preprocess \
    --config config.yml --dataset-split leak \
    --output data/perturbations_members.json

python -m perturbation.preprocess \
    --config config.yml --dataset-split unleak \
    --output data/perturbations_non_members.json

# Phase 2 — inference + MIA (GPU), reusing cached perturbations
python experiment.py \
    --config config.yml \
    --dataset-split both \
    --members-perturbations-file     data/perturbations_members.json \
    --non-members-perturbations-file data/perturbations_non_members.json
```

---

## Fine-tuning

```bash
# Config-driven (recommended)
python -m fine_tuning.fine_tune --config config.yml

# CLI overrides
python -m fine_tuning.fine_tune \
    --config config.yml \
    --model "EleutherAI/pythia-1.4b" \
    --output_dir "./checkpoints" \
    --epochs 0 1 2 3 4 5 6 7 8 9 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --bf16
```

Checkpoints are saved at `checkpoints/<model-slug>/checkpoint-epoch-{N}/`. Epoch 0 is the base model before any training.

---

## Baseline sweep (CDD / ACR)

```bash
# CDD sweep across all epochs
bash scripts/run_baseline_epoch_sweep.sh \
    checkpoints/EleutherAI_pythia-1.4b_fine_tuned \
    results/pythia_1.4b \
    --methods cdd

# ACR (slow — iterative GCG optimisation per sample)
python scripts/run_baseline_sweep.py \
    --model checkpoints/EleutherAI_pythia-1.4b_fine_tuned/checkpoint-epoch-7 \
    --epoch 7 --methods acr --max-samples 100
```

---

## Analysis and figures

After running experiments, generate all paper figures:

```bash
cd pearl

# Full IPA / PEARL analysis across all available models and epochs
python -m evaluation.ipa_analysis

# Single-model analysis
python -m evaluation.ipa_analysis --models pythia_1.4b --epoch 10

# Figure: AUC and |γ| trajectories across epochs
python evaluation/generate_rq5_auc_gamma_epochs.py --model pythia_1.4b --from-csv

# Figure: μ(S_E) and μ(S_G) across model sizes
python evaluation/generate_rq6_mu_scaling.py --epoch 10

# Figure: AUC per transformation type (bar chart)
python evaluation/generate_rq4_bar.py

# Figure: code-domain AUC + score distributions
python evaluation/generate_rq8_code_figures.py

# Figure: IPA score boxplot by task × aggregation operator
python evaluation/generate_boxplot_task_operator.py --epoch 10
```

All figures are written to `paper/_assets/`.

---

## Output files

### Per-epoch run (`results/<model>/run_<epoch>/`)

```
members/
├── records.json          # Per-perturbation records (input/output similarity, MIA scores)
├── summary.csv           # Mean output similarity per input-similarity bucket
├── mia_summary.csv       # Per-sample MIA scores (Loss, Min-K%, Neighbourhood)
├── similarity_curve.png  # Visual sensitivity curve
└── baseline_results.csv  # CDD / ACR scores (if baseline sweep was run)

non_members/              # Same structure
```

### `records.json` schema

```json
{
  "sample_idx": 0,
  "range_label": "[0.90-1.00]",
  "sim_low": 0.9,
  "sim_high": 1.0,
  "input_similarity": 0.977,
  "output_similarity": 0.821,
  "perturbation_type": "whitespace",
  "mia_loss": 2.51,
  "mia_min_k": -9.94,
  "mia_neighborhood": 1.18,
  "original_input": "...",
  "modified_input": "...",
  "original_output": "...",
  "modified_output": "..."
}
```

### Analysis outputs (`evaluation/reports/`)

| File | Contents |
|------|----------|
| `ipa_metrics.csv` | Per-model / epoch / operator: AUC, γ, precision, recall |
| `detection_at_youden.csv` | TP, FP, precision, recall at Youden-J threshold |
| `instance_scores.csv` | Per-sample IPA scores and flags (A_mean, one epoch) |
| `SUMMARY.md` | Human-readable analysis report |
| `plots/` | All diagnostic figures |

---

## IPA aggregation operators

| Operator | Formula | Best for |
|----------|---------|----------|
| `A_mean` | Mean output similarity | **Default** — best AUC on text |
| `A_median` | Median output similarity | Robust to outliers; highest \|γ\| |
| `A_min` | Minimum output similarity | High precision; best on code |
| `A_q10` / `A_q25` | 10th / 25th percentile | Best AUC on code |
| `A_neg_var` | Negative variance of output similarity | Captures consistency signal |


---

## MIA methods (baselines)

| Method | Access | Score direction | Key hyperparameter |
|--------|--------|----------------|--------------------|
| **Loss** | Gray-box (logits) | Lower raw loss = member | — |
| **Min-K% Prob** | Gray-box (logits) | Higher = member | `min_k = 0.2` |
| **Neighbourhood Attack** | Gray-box (logits) | Positive = member | Reuses PEARL's [0.90-1.00] neighbors |
| **CDD** | Black-box | Higher peakedness = member | `n_samples=20`, `α=0.05`, `ξ=0.01` |
| **ACR** | White-box (gradients) | Ratio < 1.0 = member | GCG, `P=10` tokens, `S=200` steps |

---

## Perturbation strategies

| Input similarity range | Primary transforms |
|------------------------|--------------------|
| `[0.90-1.00]` | whitespace insertions, minimal character swaps |
| `[0.80-0.90]` | character swaps (~3%), case flips (~5%) |
| `[0.70-0.80]` | whitespace (moderate), case flips (~7%) |
| `[0.60-0.70]` | whitespace (heavy), character swaps (~8%) |
| `[0.50-0.60]` | case flips (~20%), character swaps (~15%) |

With `nlpaug`: synonym substitution and spelling-based token replacement for lower ranges.

---

## References

- Shi et al. (2024). *Detecting Pretraining Data from Large Language Models*. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) — Min-K% Prob
- Mattern et al. (2023). *Membership Inference Attacks against Language Models via Neighbourhood Comparison*. [arXiv:2305.18462](https://arxiv.org/abs/2305.18462) — Neighbourhood Attack
- Dong et al. (2024). *Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models* — CDD
- Schwarzschild et al. (2024). *Rethinking LLM Memorization through the Lens of Adversarial Compression*. [arXiv:2404.15146](https://arxiv.org/abs/2404.15146) — ACR
- Carlini et al. (2021). *Extracting Training Data from Large Language Models* — Loss MIA, n-gram overlap
