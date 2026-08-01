from __future__ import annotations
import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from bias_dataset_loaders import BIASDPO_ALL_CATEGORIES
from config import Config
from contamination import run_contamination_analysis, contamination_cache_key, load_contamination_result
from data import load_train_data, _load_bbq_eval_records_for_contamination, _gather_other_contamination_texts, FAIRMT10K_DIMENSIONS
from evaluation import evaluate
from prompts import INSTRUCTION_REGISTRY, EVAL_PROMPT_REGISTRY
from training import fine_tune
from utils import (
    is_model_cached, is_baseline_eval_cached, load_model_from_cache,
    save_model_to_cache, load_baseline_eval_from_cache, save_baseline_eval_to_cache,
    build_result_row, append_results_csv, append_results_xlsx,
    model_cache_path, baseline_eval_cache_path, _model_cache_key, _baseline_eval_cache_key,
)

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bias-aware aligned instruction tuning pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-dataset", choices=["BiasDPO", "CivilComments", "AllBias"], default="BiasDPO")
    p.add_argument("--biasdpo-dir", default="data/bias-dpo", metavar="DIR")
    p.add_argument("--biasdpo-categories", nargs="*", choices=BIASDPO_ALL_CATEGORIES, default=None, metavar="CAT")
    p.add_argument("--civil-comments-path", default="data/civil-comments/train-00000-of-00001.parquet", metavar="PATH")
    p.add_argument("--civil-toxicity-threshold", type=float, default=0.5, metavar="T")
    p.add_argument("--bold-n-samples", type=int, default=None, metavar="N")
    p.add_argument("--toxigen-n-samples", type=int, default=None, metavar="N")
    p.add_argument("--fairmtbench-n-samples", type=int, default=None, metavar="N")
    p.add_argument("--bbq-single-pass", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-cot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hinge-lambda", type=float, default=0.3, metavar="λ_h")
    p.add_argument("--eval-datasets", nargs="+", choices=["BBQ", "BOLD", "ToxiGen", "FairMTBench", "ImplicitBBQ"], default=["BBQ", "BOLD", "ToxiGen", "FairMTBench", "ImplicitBBQ"], metavar="DS")
    p.add_argument("--eval-dataset", choices=["BBQ", "BOLD", "ToxiGen", "FairMTBench", "ImplicitBBQ"], default=None, metavar="DS", help="(DEPRECATED)")
    p.add_argument("--implicit-bbq-categories", nargs="+", choices=["Gender_identity", "Race_ethnicity", "Race_x_SES", "Religion", "SES", "Sexual_orientation"], default=["Gender_identity", "Race_ethnicity", "Race_x_SES", "Religion", "SES", "Sexual_orientation"], metavar="CAT")
    p.add_argument("--implicit-bbq-local-dir", default="data/implicitBBQ", metavar="DIR")
    p.add_argument("--fairmtbench-path", default="data/fairmt/FairMT_1K", metavar="PATH")
    p.add_argument("--fairmtbench-categories", nargs="+", default=["Anaphora_Ellipsis", "Fixed_Format", "Interference_Misinformation", "Jailbreak_Tips", "Negative_Feedback", "Scattered_Questions"], metavar="CAT")
    p.add_argument("--fairmtbench-dimensions", nargs="+", choices=FAIRMT10K_DIMENSIONS, default=None, metavar="DIM")
    p.add_argument("--fairmtbench-groups", nargs="+", default=None, metavar="GRP")
    p.add_argument("--fairmtbench-judge-model", default="meta-llama/Llama-Guard-3-8B", metavar="HF_ID")
    p.add_argument("--fairmtbench-max-new-tokens", type=int, default=256, metavar="N")
    p.add_argument("--fairmtbench-judge-max-new-tokens", type=int, default=20, metavar="N")
    p.add_argument("--fairmtbench-offload-policy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-bbq-dpo", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bbq-dpo-categories", nargs="*", default="Age", metavar="CAT")
    p.add_argument("--bbq-categories", nargs="+", default=["Race_ethnicity"], metavar="CAT")
    p.add_argument("--bbq-eval-categories", nargs="+", default=None, metavar="CAT")
    p.add_argument("--n-samples", type=int, default=None, metavar="N")
    p.add_argument("--bold-path", default="data/bold", metavar="PATH")
    p.add_argument("--toxigen-path", default="data/toxigen/annotated_train.csv", metavar="PATH")
    p.add_argument("--toxigen-threshold", type=float, default=2.5, metavar="T")
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct", metavar="HF_ID")
    p.add_argument("--loss-fn", choices=["ce", "label_smoothing", "focal", "ce_hinge", "ce_unlikelihood", "ce_dpo", "ce_kl", "ce_hinge_kl", "ce_dpo_kl", "ce_dpo_kl_entropy", "ce_dpo_kl_conditioned", "ce_dpo_kl_counterfactual", "ce_dpo_kl_all", "ce_ipo", "ce_ipo_kl"], default="ce_dpo_kl")
    p.add_argument("--ipo-tau", type=float, default=0.1, metavar="τ")
    p.add_argument("--bias-lambda", type=float, default=0.5, metavar="λ")
    p.add_argument("--bias-margin", type=float, default=0.15, metavar="M")
    p.add_argument("--dpo-beta", type=float, default=0.1, metavar="β")
    p.add_argument("--kl-lambda", type=float, default=0.1, metavar="λ")
    p.add_argument("--label-smooth", type=float, default=0.1, metavar="ε")
    p.add_argument("--focal-gamma", type=float, default=2.0, metavar="γ")
    p.add_argument("--entropy-lambda", type=float, default=0.05, metavar="λ")
    p.add_argument("--condition-weights", nargs=4, type=float, default=[2.0, 1.5, 1.0, 0.5], metavar=("W_AN", "W_ANN", "W_DN", "W_DNN"))
    p.add_argument("--cf-lambda", type=float, default=0.1, metavar="λ")
    p.add_argument("--cf-margin", type=float, default=0.5, metavar="γ")
    p.add_argument("--epochs", type=int, default=3, metavar="N")
    p.add_argument("--lr", type=float, default=2e-5, metavar="LR")
    p.add_argument("--batch-size", type=int, default=8, metavar="B")
    p.add_argument("--kl_mode", choices=["label_only", "full_sequence"], default="full_sequence")
    p.add_argument("--dpo-ref", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stereo-from-metadata", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-instruction", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--per-epoch-eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train-instructions", nargs="*", choices=list(INSTRUCTION_REGISTRY.keys()), default=[], metavar="INSTR")
    p.add_argument("--eval-prompts", nargs="*", choices=list(EVAL_PROMPT_REGISTRY.keys()), default=[], metavar="PROMPT")
    p.add_argument("--output-dir", default="outputs", metavar="DIR")
    p.add_argument("--results-csv", default="outputs/results.csv", metavar="PATH")
    p.add_argument("--run-id", default=None, metavar="ID")
    p.add_argument("--cache-dir", default="cache", metavar="DIR")
    p.add_argument("--retrain", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--reeval-baseline", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--baseline-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--run-contamination-check", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--contamination-calibration-path", default=None, metavar="PATH")
    p.add_argument("--contamination-calibration-n", type=int, default=300, metavar="N")
    p.add_argument("--contamination-min-k-percent", type=float, default=20.0, metavar="K")
    p.add_argument("--contamination-max-items", type=int, default=300, metavar="N")
    p.add_argument("--contamination-ts-guessing-samples", type=int, default=200, metavar="N")
    p.add_argument("--contamination-ts-guessing-max-new-tokens", type=int, default=20, metavar="N")
    p.add_argument("--reasoning-max-new-tokens", type=int, default=1024, metavar="N")
    p.add_argument("--reasoning-open-tag", type=str, default="<think>", metavar="TAG")
    p.add_argument("--reasoning-close-tag", type=str, default="</think>", metavar="TAG")

    args = p.parse_args()
    if args.eval_dataset is not None:
        warnings.warn("--eval-dataset is deprecated. Use --eval-datasets.", DeprecationWarning, 2)
        args.eval_datasets = [args.eval_dataset]
    return args

def run_pipeline(cfg: Config) -> None:
    t_start = time.time()
    print("\n" + "=" * 60)
    print(f"Run ID             : {cfg.run_id}\nModel              : {cfg.model_id}\nTrain dataset      : {cfg.train_dataset}")
    if cfg.train_dataset in ("BiasDPO", "AllBias"):
        cats_str = ", ".join(cfg.biasdpo_categories) if cfg.biasdpo_categories else "all"
        print(f"BiasDPO dir        : {cfg.biasdpo_dir}  (categories: {cats_str})")
    if cfg.train_dataset in ("CivilComments", "AllBias"):
        print(f"CivilComments path : {cfg.civil_comments_path}  (threshold={cfg.civil_toxicity_threshold})")
    print(f"Eval  datasets     : {cfg.eval_datasets}\nBBQ eval  cats     : {cfg.bbq_eval_categories}")
    if "FairMTBench" in cfg.eval_datasets:
        print(f"FairMTBench path   : {cfg.fairmtbench_path}\nFairMTBench cats   : {cfg.fairmtbench_categories}\nFairMTBench dims   : {cfg.fairmtbench_dimensions or 'all'}\nFairMTBench groups : {cfg.fairmtbench_groups or 'all'}")
    print(f"Loss fn            : {cfg.loss_fn}")
    if cfg.loss_fn in ("ce_ipo", "ce_ipo_kl"): print(f"IPO τ              : {cfg.ipo_tau}")
    print(f"Train instructions : {', '.join(cfg.train_instruction_names) or '(none)'}\nEval prompts       : {', '.join(cfg.eval_prompt_names) or '(none)'}")
    print(f"Reasoning max toks : {cfg.reasoning_max_new_tokens}  (open={cfg.reasoning_open_tag!r} close={cfg.reasoning_close_tag!r})")
    print(f"Epochs / LR / BS   : {cfg.epochs} / {cfg.lr} / {cfg.batch_size}\nCache dir          : {cfg.cache_dir}\nRetrain            : {cfg.retrain}\nBaseline only      : {cfg.baseline_only}\nRe-eval baseline   : {cfg.reeval_baseline}\nDevice             : {cfg.device}")
    if cfg.device == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(0)}\nVRAM               : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)
    
    cfg.save()
    model_cached, baseline_ready = is_model_cached(cfg), is_baseline_eval_cached(cfg)
    print(f"\n[Cache status]\n  Fine-tuned model : {'HIT  → ' + str(model_cache_path(cfg)) if model_cached else 'MISS'}\n  Baseline eval    : {'HIT  → ' + str(baseline_eval_cache_path(cfg)) if baseline_ready else 'MISS'}")
    if cfg.retrain and model_cached: print("  --retrain set: will ignore model cache and retrain from scratch.")
    if cfg.reeval_baseline and baseline_ready: print("  --reeval-baseline set: will ignore baseline eval cache and re-run.")
    
    print("\nLoading base tokenizer…")
    base_tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if base_tokenizer.pad_token is None: base_tokenizer.pad_token = base_tokenizer.eos_token

    if cfg.run_contamination_check:
        print("\n[Step 1.5] Contamination analysis (Min-K%++ / TS-Guessing)…")
        bbq_eval_records = _load_bbq_eval_records_for_contamination(cfg)
        other_texts = _gather_other_contamination_texts(cfg)
        _n_bbq = (min(len(bbq_eval_records), cfg.contamination_max_items or len(bbq_eval_records)) if bbq_eval_records else 0)
        needed_keys = []
        if bbq_eval_records:
            needed_keys.append(contamination_cache_key(cfg.model_id, "BBQ", "min_k_plus_plus", k_percent=cfg.contamination_min_k_percent, n=_n_bbq))
            needed_keys.append(contamination_cache_key(cfg.model_id, "BBQ", "ts_guessing", n=cfg.contamination_ts_guessing_samples))
        for ds_name, texts in other_texts.items():
            _n = min(len(texts), cfg.contamination_max_items or len(texts))
            needed_keys.append(contamination_cache_key(cfg.model_id, ds_name, "min_k_plus_plus", k_percent=cfg.contamination_min_k_percent, n=_n))
            
        all_cached = bool(needed_keys) and all(load_contamination_result(cfg.cache_dir, k) is not None for k in needed_keys)
        if all_cached:
            print("  [contamination] All requested checks already cached, skipping model load.")
        elif not bbq_eval_records and not other_texts:
            print("  [contamination] Nothing to score -- skipping.")
        else:
            contam_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16).to(cfg.device)
            contamination_report = run_contamination_analysis(contam_model, base_tokenizer, cfg, bbq_records=bbq_eval_records or None, other_texts_by_dataset=other_texts or None)
            contam_out = cfg.output_dir / "contamination_report.json"
            with open(contam_out, "w") as f: json.dump(contamination_report, f, indent=2)
            print(f"  Contamination report saved -> {contam_out}")
            del contam_model
            if cfg.device == "cuda": torch.cuda.empty_cache()

    use_cached_baseline = baseline_ready and not cfg.reeval_baseline
    if use_cached_baseline:
        print("\n[Step 2] Baseline evaluation — CACHE HIT, skipping inference…")
        baseline_metrics = load_baseline_eval_from_cache(cfg)
    else:
        print("\n[Step 2] Baseline evaluation — running inference…")
        base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16).to(cfg.device)
        baseline_metrics = evaluate(base_model, base_tokenizer, cfg, tag="baseline")
        save_baseline_eval_to_cache(baseline_metrics, cfg)
        del base_model
        if cfg.device == "cuda": torch.cuda.empty_cache()

    if cfg.baseline_only:
        wall = time.time() - t_start
        print("\n[--baseline-only] Skipping fine-tuning and post-FT evaluation.")
        result_row: Dict = build_result_row(cfg=cfg, baseline_metrics=baseline_metrics, postft_metrics={}, wall_seconds=wall, n_train_samples=0, n_biased=0)
        append_results_csv(csv_path=cfg.results_csv, result_row=result_row)
        print(f"\n── Cache summary ─────────────────────────────────────\n  Eval  cache key  : {_baseline_eval_cache_key(cfg)}\n  Eval  cache path : {baseline_eval_cache_path(cfg)}\n  Baseline from    : {'disk cache' if use_cached_baseline else 'live inference'}\n\nDone. Wall time: {wall / 60:.1f} min")
        return

    use_cached_model = model_cached and not cfg.retrain
    if use_cached_model:
        print("\n[Step 3] Fine-tuning — CACHE HIT, loading model…")
        ft_model, ft_tokenizer = load_model_from_cache(cfg)
        n_train_samples = 0
    else:
        if cfg.retrain and model_cached: print("\n[Step 3] Fine-tuning — RETRAIN flag set, ignoring cache…")
        else: print("\n[Step 3] Fine-tuning — CACHE MISS, training from scratch…")
        print("  Loading base model for fine-tuning…")
        ft_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16).to(cfg.device)
        ft_tokenizer = base_tokenizer
        
        print(f"  Loading training data [{cfg.train_dataset}]…")
        train_data = load_train_data(cfg)
        n_train_samples = len(train_data)
        print(f"  {n_train_samples} samples loaded")
        ft_model = fine_tune(ft_model, ft_tokenizer, train_data, cfg)
        save_model_to_cache(ft_model, ft_tokenizer, cfg)

    print("\n[Step 4] Post-fine-tuning evaluation…")
    postft_metrics = evaluate(ft_model, ft_tokenizer, cfg, tag="postft")
    
    print("\n── Delta (post-FT minus baseline) ──────────────────")
    for k in sorted(set(baseline_metrics) & set(postft_metrics)):
        try:
            b_val, p_val = float(baseline_metrics[k]), float(postft_metrics[k])
            delta = p_val - b_val
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
            ds, metric = k.split("/", 1)
            print(f"  {ds:<15} {metric:<30}: {b_val:.4f} → {p_val:.4f}  ({arrow}{abs(delta):.4f})")
        except (TypeError, ValueError, AttributeError): pass
        
    wall = time.time() - t_start
    result_row = build_result_row(cfg=cfg, baseline_metrics=baseline_metrics, postft_metrics=postft_metrics, wall_seconds=wall, n_train_samples=n_train_samples, n_biased=n_train_samples)
    append_results_csv(csv_path=cfg.results_csv, result_row=result_row)
    
    epoch_logs: List[Dict] = []
    log_path = cfg.output_dir / "training_log.json"
    if log_path.exists():
        with open(log_path) as f: epoch_logs = json.load(f)
    append_results_xlsx(xlsx_path=cfg.results_csv.with_suffix(".xlsx"), result_row=result_row, epoch_logs=epoch_logs, run_id=cfg.run_id, timestamp_utc=result_row.get("timestamp_utc", ""), cfg_dict=cfg.to_dict())
    
    print(f"\n── Cache summary ─────────────────────────────────────\n  Model cache key  : {_model_cache_key(cfg)}\n  Model cache path : {model_cache_path(cfg)}\n  Eval  cache key  : {_baseline_eval_cache_key(cfg)}\n  Eval  cache path : {baseline_eval_cache_path(cfg)}\n  Baseline from    : {'disk cache' if use_cached_baseline else 'live inference'}\n  FT model from    : {'disk cache' if use_cached_model else 'fresh training'}\n\nDone. Wall time: {wall / 60:.1f} min")

if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args)
    run_pipeline(cfg)