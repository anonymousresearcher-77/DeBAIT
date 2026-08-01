# DeBAIT — De-Biasing via Aligned Instruction Tuning

> Anonymous code submission for double-blind peer review. Paper title,
> author names, and any other identifying information have been withheld
> and will be added after the review process concludes.

This repository contains the training, evaluation, and analysis pipeline used
to fine-tune instruction-tuned LLMs against several bias/toxicity training
signals (BiasDPO, CivilComments, BBQ-derived preference pairs) using a family
of DPO/IPO-style losses, and to evaluate the resulting models on BBQ,
ImplicitBBQ, BOLD, ToxiGen, and FairMTBench.

## Contents

- [Datasets](#datasets)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Citation](#citation)


## Datasets

None of the raw data is bundled in this repository. Download each dataset from their original source, as we have done, and point them into the paths below (or point the corresponding `--*-path` / `--*-dir` CLI
flag at wherever you keep them).

| Dataset | Source | Used for | Expected local path (default) |
|---|---|---|---|
| **BBQ** | [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ/tree/main/data) | BBQ evaluation (`score_bbq`); also converted into DPO preference pairs for training (`bbq_dpo_loader.py`, the `BBQ-DPO` component of `AllBias`) | `data/bbq/combined.jsonl` (all BBQ category files concatenated into one JSONL) |
| **ImplicitBBQ** | [ssrivastava22/ImplicitBBQ](https://github.com/ssrivastava22/ImplicitBBQ) | ImplicitBBQ evaluation, tests whether debiasing transfers past BBQ's explicit identity cues | `data/implicitBBQ/` (per-category `.jsonl`; auto-downloaded from the repo's raw GitHub URLs if not found locally) |
| **BOLD** | [amazon-science/bold](https://github.com/amazon-science/bold) | Open-ended generation regard evaluation, scored with the `sasha/regardv3` classifier | `data/bold/` (per-domain `*_prompt.json` files, e.g. `race_prompt.json`) |
| **BiasDPO / CivilComments** | [SaketR1/bias-grpo-data](https://huggingface.co/datasets/SaketR1/bias-grpo-data) | Training data - `BiasDPO` preference pairs and toxicity-thresholded `CivilComments` pairs; both feed into the combined `AllBias` training mix | `data/bias-grpo-data/bias-dpo/` (parquet) and `data/bias-grpo-data/civil-comments/train-00000-of-00001.parquet` |
| **ToxiGen** | [toxigen/toxigen-data](https://huggingface.co/datasets/toxigen/toxigen-data) | Toxicity classification evaluation (accuracy, macro-F1, per-group FNR) | `data/toxigen/annotated_train.csv` |
| **FairMTBench** | [FanZT6/FairMT-bench](https://github.com/FanZT6/FairMT-bench) | Multi-turn dialogue bias evaluation, judged with a Llama-Guard-style model | `data/fairmt/FairMT_1K/` (per-category `.json`, e.g. `Jailbreak_Tips.json`) |

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | CLI entrypoint. Parses arguments, orchestrates baseline eval → fine-tuning → post-FT eval → results logging, with model/eval caching. |
| `config.py` | `Config` object built from CLI args; central place for hyperparameters and dataset/category mappings. |
| `training.py` | Training loops for preference-pair losses (DPO/IPO family) and single-sequence SFT-style losses. |
| `losses.py` | Loss implementations: cross-entropy, label smoothing, DPO, IPO, hinge, and KL-regularised variants. |
| `data.py` | Dataset loading/tokenization glue: `SFTDataset`, `BiasDpoDataset`, collate functions, and loaders for BOLD / ToxiGen / FairMTBench. |
| `bbq_utils.py` | Low-level BBQ JSONL loading and formatting into HF `Dataset` objects. |
| `bbq_dpo_loader.py` | Converts BBQ context/question rows into DPO-style (chosen, rejected) preference pairs. |
| `bias_dataset_loaders.py` | Loaders for BiasDPO, CivilComments, and UnQover parquet files, plus `load_all_bias_datasets` for the combined `AllBias` training mix. |
| `prompts.py` / `prompt_utils.py` | Registries of training instructions and evaluation prompts, and the composition logic that assembles them into final prompts. |
| `generation.py` | Model-generation helpers shared across evaluators (BBQ, ToxiGen, BOLD, FairMTBench multi-turn + judge). |
| `evaluation.py` | Scoring logic for each eval dataset (accuracy, s_DIS/s_AMB bias scores, regard, F1, bias rate) and the top-level `evaluate` / `evaluate_lightweight` entrypoints. |
| `implicit_bbq_eval.py` | Standalone loader + generation-based scorer for ImplicitBBQ. |
| `contamination.py` | Min-K%++ and TS-Guessing pretraining-data contamination checks, run once against the base checkpoint. |
| `generate_corpus.py` | Builds the post-cutoff news corpus used to calibrate the Min-K%++ contamination check. |
| `results_xlsx.py` | Four-sheet Excel results writer (per-run results, per-epoch curves, config index, vertical results). |
| `utils.py` | Model/eval caching (keyed by run config), result-row construction, and CSV/XLSX result appenders. |
| `run_hashed.sh` | Experiment runner: loops over models × loss functions × instruction/prompt combinations for the main paper sweep. |


The contamination check (`contamination.py`) additionally uses a small post-cutoff news corpus for Min-K%++ calibration, built on the fly by `generate_corpus.py` from public RSS feeds (BBC, NYT, CNN, NPR, etc.) rather than a fixed downloadable dataset.

## Installation

```bash
pip install torch transformers datasets pandas openpyxl tqdm
```

A CUDA-capable GPU is strongly recommended for both fine-tuning and generation-based evaluation; `config.py` falls back to CPU automatically if
none is available.

## Usage

Run a single experiment:

```bash
python main.py \
    --model-id meta-llama/Llama-3.2-3B-Instruct \
    --train-dataset AllBias \
    --loss-fn ce_ipo \
    --ipo-tau 0.1 \
    --eval-datasets BBQ BOLD ToxiGen FairMTBench ImplicitBBQ \
    --output-dir outputs/my_run \
    --results-csv outputs/results.csv
```

To reproduce the full paper sweep (multiple models × loss functions × instruction/prompt combinations), see `run_hashed.sh`, which wraps `main.py` with the shared arguments used across all runs. All CLI flags are documented in `main.py::parse_args`.

## Outputs

Each run writes to `--output-dir`:
- `config.json` — full resolved configuration for the run
- `training_log.json` — per-epoch training/eval metrics
- `eval_<dataset>_{baseline,postft}.json` and per-sample `.jsonl` dumps
- `contamination_report.json` (if `--run-contamination-check` is set)

Across runs, `utils.py` and `results_xlsx.py` append to:
- `--results-csv` — one row per run, all baseline/post-FT/delta metrics
- the equivalent `.xlsx` — a four-sheet workbook (per-run results, per-epoch
  curves, config index, and a vertical baseline/post-FT/Δ comparison sheet)

Fine-tuned models and baseline evaluations are cached under `--cache-dir`, keyed by a hash of the relevant config fields, so repeat runs with identical settings skip retraining/re-evaluation (`--retrain` / `--reeval-baseline` to force a refresh).

## Citation

Citation details are withheld during the double-blind review process and will be added following acceptance/publication.
