# IPA / PEARL Results Analysis

Generated: 2026-06-12 07:01 UTC

## Methodology

Neighborhood aggregation score `S(x) = A({output_similarity | (x',y') ∈ N(x)})` with operators from `problem_formulation.tex`.

- **Memorization gap (paper):** γ = μ(S_G) − μ(S_E)
- **Z-score:** M_score(u) = (μ(S_G) − S(u)) / σ(S_G)
- **IPA rule (τ=1):** S(u) < μ(S_G) − τ·γ when members are more brittle; S(u) > μ(S_G) − τ·γ when members show higher output stability (empirical PSH)
- **AUC:** oriented so members are the positive class (auto-calibrated)
- **Counts in tables:** Youden-optimal threshold (max TPR − FPR)

## Models analysed

- **Pythia-70M** (`pythia_70m`): 11 epoch(s)
- **Pythia-410M** (`pythia_410m`): 11 epoch(s)
- **Pythia-1.4B** (`pythia_1.4b`): 11 epoch(s)
- **Pythia-2.8B** (`pythia_2.8b`): 2 epoch(s)

## IPA A_mean — latest epoch per model

| Model | Epoch | γ | |γ| | AUC | TP | FP | Recall | Precision |
|-------|-------|---|-----|-----|----|----|--------|-----------|
| Pythia-70M | 10 | -0.0510 | 0.0510 | 0.560 | 428 | 326 | 0.428 | 0.568 |
| Pythia-410M | 10 | -0.0706 | 0.0706 | 0.595 | 570 | 411 | 0.570 | 0.581 |
| Pythia-1.4B | 10 | -0.2133 | 0.2133 | 0.761 | 609 | 234 | 0.609 | 0.722 |
| Pythia-2.8B | 10 | -0.2544 | 0.2544 | 0.805 | 720 | 254 | 0.720 | 0.739 |

## MIA baselines (latest epoch)

| Model | Epoch | Method | AUC | TP | FP |
|-------|-------|--------|-----|----|----|
| Pythia-70M | 10 | MIA-loss | 0.645 | 456 | 235 |
| Pythia-70M | 10 | MIA-min-K | 0.648 | 547 | 315 |
| Pythia-70M | 10 | MIA-neighborhood | 0.659 | 509 | 252 |
| Pythia-410M | 10 | MIA-loss | 0.636 | 480 | 288 |
| Pythia-410M | 10 | MIA-min-K | 0.652 | 391 | 176 |
| Pythia-410M | 10 | MIA-neighborhood | 0.643 | 605 | 380 |
| Pythia-1.4B | 10 | MIA-loss | 0.663 | 514 | 257 |
| Pythia-1.4B | 10 | MIA-min-K | 0.668 | 540 | 270 |
| Pythia-1.4B | 10 | MIA-neighborhood | 0.664 | 553 | 294 |
| Pythia-2.8B | 10 | MIA-loss | 0.661 | 468 | 216 |
| Pythia-2.8B | 10 | MIA-min-K | 0.665 | 419 | 163 |
| Pythia-2.8B | 10 | MIA-neighborhood | 0.664 | 524 | 273 |

## Flagged instance counts (members, Youden J)

| Model | Epoch | Method | # flagged |
|-------|-------|--------|-----------|
| Pythia-70M | 10 | PEARL | 428 |
| Pythia-70M | 10 | MIA-loss | 456 |
| Pythia-70M | 10 | MIA-min-K | 547 |
| Pythia-70M | 10 | MIA-neighborhood | 509 |
| Pythia-410M | 10 | PEARL | 570 |
| Pythia-410M | 10 | MIA-loss | 480 |
| Pythia-410M | 10 | MIA-min-K | 391 |
| Pythia-410M | 10 | MIA-neighborhood | 605 |
| Pythia-1.4B | 10 | PEARL | 609 |
| Pythia-1.4B | 10 | MIA-loss | 514 |
| Pythia-1.4B | 10 | MIA-min-K | 540 |
| Pythia-1.4B | 10 | MIA-neighborhood | 553 |
| Pythia-2.8B | 10 | PEARL | 720 |
| Pythia-2.8B | 10 | MIA-loss | 468 |
| Pythia-2.8B | 10 | MIA-min-K | 419 |
| Pythia-2.8B | 10 | MIA-neighborhood | 524 |

## PEARL vs MIA overlap on members (Venn regions)

Columns: **PEARL** / **MIA** = total flagged; **Both** = intersection; **PEARL only** / **MIA only** = exclusive; **Neither** = not flagged by either.

| Model | Epoch | MIA | N | PEARL | MIA | Both | PEARL only | MIA only | Neither | Jaccard |
|-------|-------|-----|---|-------|-----|------|------------|----------|---------|---------|
| Pythia-70M | 0 | MIA-loss | 1000 | 402 | 401 | 198 | 204 | 203 | 395 | 0.327 |
| Pythia-70M | 0 | MIA-min-K | 1000 | 402 | 507 | 235 | 167 | 272 | 326 | 0.349 |
| Pythia-70M | 0 | MIA-neighborhood | 1000 | 402 | 472 | 215 | 187 | 257 | 341 | 0.326 |
| Pythia-70M | 1 | MIA-loss | 1000 | 599 | 448 | 298 | 301 | 150 | 251 | 0.398 |
| Pythia-70M | 1 | MIA-min-K | 1000 | 599 | 581 | 374 | 225 | 207 | 194 | 0.464 |
| Pythia-70M | 1 | MIA-neighborhood | 1000 | 599 | 510 | 338 | 261 | 172 | 229 | 0.438 |
| Pythia-70M | 2 | MIA-loss | 1000 | 471 | 463 | 268 | 203 | 195 | 334 | 0.402 |
| Pythia-70M | 2 | MIA-min-K | 1000 | 471 | 434 | 249 | 222 | 185 | 344 | 0.380 |
| Pythia-70M | 2 | MIA-neighborhood | 1000 | 471 | 490 | 279 | 192 | 211 | 318 | 0.409 |
| Pythia-70M | 3 | MIA-loss | 1000 | 335 | 388 | 166 | 169 | 222 | 443 | 0.298 |
| Pythia-70M | 3 | MIA-min-K | 1000 | 335 | 585 | 230 | 105 | 355 | 310 | 0.333 |
| Pythia-70M | 3 | MIA-neighborhood | 1000 | 335 | 491 | 195 | 140 | 296 | 369 | 0.309 |
| Pythia-70M | 4 | MIA-loss | 1000 | 401 | 390 | 211 | 190 | 179 | 420 | 0.364 |
| Pythia-70M | 4 | MIA-min-K | 1000 | 401 | 562 | 273 | 128 | 289 | 310 | 0.396 |
| Pythia-70M | 4 | MIA-neighborhood | 1000 | 401 | 501 | 244 | 157 | 257 | 342 | 0.371 |
| Pythia-70M | 5 | MIA-loss | 1000 | 425 | 450 | 229 | 196 | 221 | 354 | 0.354 |
| Pythia-70M | 5 | MIA-min-K | 1000 | 425 | 453 | 227 | 198 | 226 | 349 | 0.349 |
| Pythia-70M | 5 | MIA-neighborhood | 1000 | 425 | 491 | 253 | 172 | 238 | 337 | 0.382 |
| Pythia-70M | 6 | MIA-loss | 1000 | 501 | 456 | 271 | 230 | 185 | 314 | 0.395 |
| Pythia-70M | 6 | MIA-min-K | 1000 | 501 | 460 | 275 | 226 | 185 | 314 | 0.401 |
| Pythia-70M | 6 | MIA-neighborhood | 1000 | 501 | 475 | 273 | 228 | 202 | 297 | 0.388 |
| Pythia-70M | 7 | MIA-loss | 1000 | 473 | 461 | 262 | 211 | 199 | 328 | 0.390 |
| Pythia-70M | 7 | MIA-min-K | 1000 | 473 | 468 | 253 | 220 | 215 | 312 | 0.368 |
| Pythia-70M | 7 | MIA-neighborhood | 1000 | 473 | 501 | 274 | 199 | 227 | 300 | 0.391 |
| Pythia-70M | 8 | MIA-loss | 1000 | 457 | 448 | 254 | 203 | 194 | 349 | 0.390 |
| Pythia-70M | 8 | MIA-min-K | 1000 | 457 | 505 | 280 | 177 | 225 | 318 | 0.411 |
| Pythia-70M | 8 | MIA-neighborhood | 1000 | 457 | 538 | 287 | 170 | 251 | 292 | 0.405 |
| Pythia-70M | 9 | MIA-loss | 1000 | 404 | 463 | 232 | 172 | 231 | 365 | 0.365 |
| Pythia-70M | 9 | MIA-min-K | 1000 | 404 | 552 | 272 | 132 | 280 | 316 | 0.398 |
| Pythia-70M | 9 | MIA-neighborhood | 1000 | 404 | 496 | 245 | 159 | 251 | 345 | 0.374 |
| Pythia-70M | 10 | MIA-loss | 1000 | 428 | 456 | 243 | 185 | 213 | 359 | 0.379 |
| Pythia-70M | 10 | MIA-min-K | 1000 | 428 | 547 | 272 | 156 | 275 | 297 | 0.387 |
| Pythia-70M | 10 | MIA-neighborhood | 1000 | 428 | 509 | 264 | 164 | 245 | 327 | 0.392 |
| Pythia-410M | 0 | MIA-loss | 1000 | 734 | 504 | 390 | 344 | 114 | 152 | 0.460 |
| Pythia-410M | 0 | MIA-min-K | 1000 | 734 | 496 | 379 | 355 | 117 | 149 | 0.445 |
| Pythia-410M | 0 | MIA-neighborhood | 1000 | 734 | 508 | 388 | 346 | 120 | 146 | 0.454 |
| Pythia-410M | 1 | MIA-loss | 1000 | 191 | 595 | 145 | 46 | 450 | 359 | 0.226 |
| Pythia-410M | 1 | MIA-min-K | 1000 | 191 | 535 | 142 | 49 | 393 | 416 | 0.243 |
| Pythia-410M | 1 | MIA-neighborhood | 1000 | 191 | 629 | 159 | 32 | 470 | 339 | 0.240 |
| Pythia-410M | 2 | MIA-loss | 1000 | 501 | 512 | 281 | 220 | 231 | 268 | 0.384 |
| Pythia-410M | 2 | MIA-min-K | 1000 | 501 | 552 | 303 | 198 | 249 | 250 | 0.404 |
| Pythia-410M | 2 | MIA-neighborhood | 1000 | 501 | 619 | 332 | 169 | 287 | 212 | 0.421 |
| Pythia-410M | 3 | MIA-loss | 1000 | 460 | 455 | 264 | 196 | 191 | 349 | 0.406 |
| Pythia-410M | 3 | MIA-min-K | 1000 | 460 | 549 | 302 | 158 | 247 | 293 | 0.427 |
| Pythia-410M | 3 | MIA-neighborhood | 1000 | 460 | 650 | 333 | 127 | 317 | 223 | 0.429 |
| Pythia-410M | 4 | MIA-loss | 1000 | 323 | 384 | 164 | 159 | 220 | 457 | 0.302 |
| Pythia-410M | 4 | MIA-min-K | 1000 | 323 | 420 | 171 | 152 | 249 | 428 | 0.299 |
| Pythia-410M | 4 | MIA-neighborhood | 1000 | 323 | 650 | 231 | 92 | 419 | 258 | 0.311 |
| Pythia-410M | 5 | MIA-loss | 1000 | 527 | 459 | 283 | 244 | 176 | 297 | 0.403 |
| Pythia-410M | 5 | MIA-min-K | 1000 | 527 | 428 | 269 | 258 | 159 | 314 | 0.392 |
| Pythia-410M | 5 | MIA-neighborhood | 1000 | 527 | 628 | 361 | 166 | 267 | 206 | 0.455 |
| Pythia-410M | 6 | MIA-loss | 1000 | 586 | 393 | 279 | 307 | 114 | 300 | 0.399 |
| Pythia-410M | 6 | MIA-min-K | 1000 | 586 | 538 | 369 | 217 | 169 | 245 | 0.489 |
| Pythia-410M | 6 | MIA-neighborhood | 1000 | 586 | 579 | 374 | 212 | 205 | 209 | 0.473 |
| Pythia-410M | 7 | MIA-loss | 1000 | 401 | 465 | 249 | 152 | 216 | 383 | 0.404 |
| Pythia-410M | 7 | MIA-min-K | 1000 | 401 | 422 | 231 | 170 | 191 | 408 | 0.390 |
| Pythia-410M | 7 | MIA-neighborhood | 1000 | 401 | 603 | 279 | 122 | 324 | 275 | 0.385 |
| Pythia-410M | 8 | MIA-loss | 1000 | 379 | 483 | 227 | 152 | 256 | 365 | 0.357 |
| Pythia-410M | 8 | MIA-min-K | 1000 | 379 | 426 | 207 | 172 | 219 | 402 | 0.346 |
| Pythia-410M | 8 | MIA-neighborhood | 1000 | 379 | 616 | 264 | 115 | 352 | 269 | 0.361 |
| Pythia-410M | 9 | MIA-loss | 1000 | 429 | 482 | 261 | 168 | 221 | 350 | 0.402 |
| Pythia-410M | 9 | MIA-min-K | 1000 | 429 | 448 | 247 | 182 | 201 | 370 | 0.392 |
| Pythia-410M | 9 | MIA-neighborhood | 1000 | 429 | 631 | 303 | 126 | 328 | 243 | 0.400 |
| Pythia-410M | 10 | MIA-loss | 1000 | 570 | 480 | 322 | 248 | 158 | 272 | 0.442 |
| Pythia-410M | 10 | MIA-min-K | 1000 | 570 | 391 | 267 | 303 | 124 | 306 | 0.385 |
| Pythia-410M | 10 | MIA-neighborhood | 1000 | 570 | 605 | 380 | 190 | 225 | 205 | 0.478 |
| Pythia-1.4B | 0 | MIA-loss | 1000 | 124 | 516 | 83 | 41 | 433 | 443 | 0.149 |
| Pythia-1.4B | 0 | MIA-min-K | 1000 | 124 | 525 | 83 | 41 | 442 | 434 | 0.147 |
| Pythia-1.4B | 0 | MIA-neighborhood | 1000 | 124 | 585 | 85 | 39 | 500 | 376 | 0.136 |
| Pythia-1.4B | 1 | MIA-loss | 1000 | 319 | 504 | 182 | 137 | 322 | 359 | 0.284 |
| Pythia-1.4B | 1 | MIA-min-K | 1000 | 319 | 538 | 191 | 128 | 347 | 334 | 0.287 |
| Pythia-1.4B | 1 | MIA-neighborhood | 1000 | 319 | 650 | 218 | 101 | 432 | 249 | 0.290 |
| Pythia-1.4B | 2 | MIA-loss | 100 | 49 | 29 | 16 | 33 | 13 | 38 | 0.258 |
| Pythia-1.4B | 2 | MIA-min-K | 100 | 49 | 28 | 16 | 33 | 12 | 39 | 0.262 |
| Pythia-1.4B | 2 | MIA-neighborhood | 100 | 49 | 53 | 29 | 20 | 24 | 27 | 0.397 |
| Pythia-1.4B | 3 | MIA-loss | 100 | 54 | 33 | 19 | 35 | 14 | 32 | 0.279 |
| Pythia-1.4B | 3 | MIA-min-K | 100 | 54 | 31 | 18 | 36 | 13 | 33 | 0.269 |
| Pythia-1.4B | 3 | MIA-neighborhood | 100 | 54 | 42 | 28 | 26 | 14 | 32 | 0.412 |
| Pythia-1.4B | 4 | MIA-loss | 100 | 45 | 32 | 17 | 28 | 15 | 40 | 0.283 |
| Pythia-1.4B | 4 | MIA-min-K | 100 | 45 | 50 | 26 | 19 | 24 | 31 | 0.377 |
| Pythia-1.4B | 4 | MIA-neighborhood | 100 | 45 | 47 | 24 | 21 | 23 | 32 | 0.353 |
| Pythia-1.4B | 5 | MIA-loss | 100 | 86 | 55 | 48 | 38 | 7 | 7 | 0.516 |
| Pythia-1.4B | 5 | MIA-min-K | 100 | 86 | 30 | 27 | 59 | 3 | 11 | 0.303 |
| Pythia-1.4B | 5 | MIA-neighborhood | 100 | 86 | 39 | 36 | 50 | 3 | 11 | 0.405 |
| Pythia-1.4B | 6 | MIA-loss | 100 | 54 | 34 | 19 | 35 | 15 | 31 | 0.275 |
| Pythia-1.4B | 6 | MIA-min-K | 100 | 54 | 51 | 31 | 23 | 20 | 26 | 0.419 |
| Pythia-1.4B | 6 | MIA-neighborhood | 100 | 54 | 38 | 23 | 31 | 15 | 31 | 0.333 |
| Pythia-1.4B | 7 | MIA-loss | 100 | 59 | 30 | 20 | 39 | 10 | 31 | 0.290 |
| Pythia-1.4B | 7 | MIA-min-K | 100 | 59 | 46 | 31 | 28 | 15 | 26 | 0.419 |
| Pythia-1.4B | 7 | MIA-neighborhood | 100 | 59 | 38 | 27 | 32 | 11 | 30 | 0.386 |
| Pythia-1.4B | 8 | MIA-loss | 100 | 55 | 30 | 18 | 37 | 12 | 33 | 0.269 |
| Pythia-1.4B | 8 | MIA-min-K | 100 | 55 | 45 | 28 | 27 | 17 | 28 | 0.389 |
| Pythia-1.4B | 8 | MIA-neighborhood | 100 | 55 | 61 | 37 | 18 | 24 | 21 | 0.468 |
| Pythia-1.4B | 9 | MIA-loss | 100 | 57 | 30 | 18 | 39 | 12 | 31 | 0.261 |
| Pythia-1.4B | 9 | MIA-min-K | 100 | 57 | 46 | 27 | 30 | 19 | 24 | 0.355 |
| Pythia-1.4B | 9 | MIA-neighborhood | 100 | 57 | 61 | 37 | 20 | 24 | 19 | 0.457 |
| Pythia-1.4B | 10 | MIA-loss | 1000 | 609 | 514 | 353 | 256 | 161 | 230 | 0.458 |
| Pythia-1.4B | 10 | MIA-min-K | 1000 | 609 | 540 | 372 | 237 | 168 | 223 | 0.479 |
| Pythia-1.4B | 10 | MIA-neighborhood | 1000 | 609 | 553 | 376 | 233 | 177 | 214 | 0.478 |
| Pythia-2.8B | 1 | MIA-loss | 1000 | 354 | 454 | 207 | 147 | 247 | 399 | 0.344 |
| Pythia-2.8B | 1 | MIA-min-K | 1000 | 354 | 478 | 212 | 142 | 266 | 380 | 0.342 |
| Pythia-2.8B | 1 | MIA-neighborhood | 1000 | 354 | 675 | 260 | 94 | 415 | 231 | 0.338 |
| Pythia-2.8B | 10 | MIA-loss | 1000 | 720 | 468 | 374 | 346 | 94 | 186 | 0.460 |
| Pythia-2.8B | 10 | MIA-min-K | 1000 | 720 | 419 | 338 | 382 | 81 | 199 | 0.422 |
| Pythia-2.8B | 10 | MIA-neighborhood | 1000 | 720 | 524 | 415 | 305 | 109 | 171 | 0.501 |

Venn diagrams: `plots/<model>_epoch<E>_pearl_mia_venn.png` (large pairwise), `plots/<model>_epoch<E>_pearl_all_mias_venn.png` (unified 4-set), and `plots/cross_model_venn_<mia>_epoch<E>.png`.

## Output files

- `ipa_metrics.csv` — full metrics per model/epoch/operator
- `detection_at_youden.csv` — detection counts (subset of ipa_metrics)
- `mia_metrics.csv`, `cdd_metrics.csv` — baselines when available
- `mia_overlap.csv` — PEARL vs MIA overlap on members (Venn counts)
- `model_epoch_comparison.csv` — cross-model AUC and flagged counts
- `plots/cross_model_auc_epoch*.png` — PEARL + MIA AUC (no CDD)
- `plots/cross_model_flagged_epoch*.png` — flagged member counts
- `plots/*_pearl_mia_venn.png` — large pairwise PEARL vs each MIA
- `plots/*_pearl_all_mias_venn.png` — unified PEARL + all MIAs Venn
- `plots/cross_model_venn_*_epoch*.png` — Venns by MIA method
- `pythia_410m_score_distributions_epoch*.csv` — per-sample scores
- `plots/pythia_410m_epoch*_ipa_scores_boxplot.png` — IPA operator boxplots
- `plots/pythia_410m_epoch*_detection_scores_boxplot.png` — PEARL / MIA / CDD
- `model_size_auc_gamma_epochs_0_1_10.csv` — AUC & γ by model size
- `plots/model_size_auc_epochs_0_1_10.png` — AUC vs size (epochs 0, 1, 10)
- `plots/model_size_gamma_epochs_0_1_10.png` — |γ| vs size (epochs 0, 1, 10)
