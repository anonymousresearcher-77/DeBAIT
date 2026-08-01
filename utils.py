from __future__ import annotations
from typing import Dict, List, Optional
import csv
import hashlib
import socket
import json
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import Config

def _md5(payload: str) -> str:
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

def _model_cache_key(cfg: "Config") -> str:
    cats = ("|".join(sorted(cfg.biasdpo_categories)) if cfg.biasdpo_categories else "all")
    instrs = "|".join(cfg.train_instruction_names)
    canonical = f"model={cfg.model_id}__ds={cfg.train_dataset}__cats={cats}__loss={cfg.loss_fn}__ep={cfg.epochs}__lr={cfg.lr}__bs={cfg.batch_size}__bl={cfg.bias_lambda}__beta={cfg.dpo_beta}__kl={cfg.kl_lambda}__tau={cfg.ipo_tau}__instr={instrs}__use_instr={cfg.use_instruction}__cot={cfg.use_cot}"
    model_slug = cfg.model_id.replace("/", "--")
    return f"{model_slug}__{cfg.train_dataset}__{cfg.loss_fn}__{_md5(canonical)}"

def _baseline_eval_cache_key(cfg: "Config") -> str:
    cats = "|".join(sorted(cfg.bbq_eval_categories))
    datasets = "|".join(sorted(cfg.eval_datasets))
    prompts = "|".join(cfg.eval_prompt_names)
    fairmt_dims = "|".join(sorted(cfg.fairmtbench_dimensions)) if cfg.fairmtbench_dimensions else "all"
    fairmt_grps = "|".join(sorted(cfg.fairmtbench_groups)) if cfg.fairmtbench_groups else "all"
    canonical = f"model={cfg.model_id}__ds={datasets}__bbq_cats={cats}__prompts={prompts}__reasoning_max_new_tokens={cfg.reasoning_max_new_tokens}__reasoning_open_tag={cfg.reasoning_open_tag}__reasoning_close_tag={cfg.reasoning_close_tag}__bbq_sp={cfg.bbq_single_pass}__tox_thresh={cfg.toxigen_threshold}__use_instr={cfg.use_instruction}__fairmt_path={cfg.fairmtbench_path}__fairmt_dims={fairmt_dims}__fairmt_grps={fairmt_grps}"
    model_slug = cfg.model_id.replace("/", "--")
    datasets_slug = datasets.replace("|", "-")
    return f"{model_slug}__{datasets_slug}__baseline__{_md5(canonical)}"

def model_cache_path(cfg: "Config") -> Path:
    return cfg.cache_dir / "models" / _model_cache_key(cfg)

def save_model_to_cache(model, tokenizer, cfg: "Config") -> Path:
    dest = model_cache_path(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    tokenizer.save_pretrained(str(dest))
    meta = {"cache_key": _model_cache_key(cfg), "cached_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg.to_dict()}
    with open(dest / "cache_meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"  [cache] Fine-tuned model saved → {dest}")
    return dest

def load_model_from_cache(cfg: "Config"):
    dest = model_cache_path(cfg)
    if not (dest / "config.json").exists(): return None, None
    print(f"  [cache] Fine-tuned model found → loading from {dest}")
    tokenizer = AutoTokenizer.from_pretrained(str(dest))
    model = AutoModelForCausalLM.from_pretrained(str(dest), torch_dtype=torch.bfloat16).to(cfg.device)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def is_model_cached(cfg: "Config") -> bool:
    return (model_cache_path(cfg) / "config.json").exists()

def baseline_eval_cache_path(cfg: "Config") -> Path:
    return cfg.cache_dir / "evals" / f"{_baseline_eval_cache_key(cfg)}.json"

def save_baseline_eval_to_cache(metrics: Dict, cfg: "Config") -> Path:
    dest = baseline_eval_cache_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cache_key": _baseline_eval_cache_key(cfg), "cached_at_utc": datetime.now(timezone.utc).isoformat(), "config_snapshot": {"model_id": cfg.model_id, "eval_datasets": cfg.eval_datasets, "eval_prompt_names": cfg.eval_prompt_names, "bbq_eval_categories": cfg.bbq_eval_categories}, "metrics": metrics}
    with open(dest, "w") as f: json.dump(payload, f, indent=2)
    print(f"  [cache] Baseline eval saved → {dest}")
    return dest

def load_baseline_eval_from_cache(cfg: "Config") -> Optional[Dict]:
    dest = baseline_eval_cache_path(cfg)
    if not dest.exists(): return None
    with open(dest) as f: payload = json.load(f)
    print(f"  [cache] Baseline eval loaded from cache → {dest}\n          Cached at: {payload.get('cached_at_utc', 'unknown')}")
    return payload.get("metrics", {})

def is_baseline_eval_cached(cfg: "Config") -> bool:
    return baseline_eval_cache_path(cfg).exists()

def build_result_row(cfg, baseline_metrics, postft_metrics, wall_seconds, n_train_samples, n_biased) -> Dict:
    row: Dict = {**cfg.to_dict(), "run_id": cfg.run_id, "n_train_samples": n_train_samples, "n_biased_samples": n_biased, "wall_time_s": round(wall_seconds, 1), "timestamp_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname()}
    all_keys = sorted(set(baseline_metrics) | set(postft_metrics))
    for k in all_keys:
        row[f"baseline_{k}"] = baseline_metrics.get(k, "")
        row[f"postft_{k}"] = postft_metrics.get(k, "")
        try: row[f"delta_{k}"] = round(float(postft_metrics.get(k, 0)) - float(baseline_metrics.get(k, 0)), 4)
        except (TypeError, ValueError): row[f"delta_{k}"] = ""
    return row

def append_results_csv(csv_path: Path, result_row: Dict) -> None:
    """Append one run's result row to the shared results CSV.

    Different loss functions surface different metric columns (e.g. IPO runs have no 'dpo' component, per-category BBQ columns 
    vary with --bbq-eval-categories), so the column set can grow between runs. Ratherthan fail with a fixed fieldnames list, 
    the header is widened as needed and the file is rewritten with the union of columns seen so far.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: List[Dict] = []
    fieldnames: List[str] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        if existing_rows:
            fieldnames = list(existing_rows[0].keys())

    for key in result_row:
        if key not in fieldnames:
            fieldnames.append(key)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        writer.writerow(result_row)
    print(f"\nResults appended -> {csv_path}")

def append_results_xlsx(xlsx_path: Path, result_row: Dict, epoch_logs: List[Dict], run_id: str, timestamp_utc: str, cfg_dict: Dict) -> None:
    from results_xlsx import append_results_xlsx as _append
    _append(xlsx_path=xlsx_path, result_row=result_row, epoch_logs=epoch_logs, run_id=run_id, timestamp_utc=timestamp_utc, cfg_dict=cfg_dict)