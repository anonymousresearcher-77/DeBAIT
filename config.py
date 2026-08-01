from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

LABEL_TOKEN = {0: "A", 1: "B", 2: "C"}
CTX_COND_MAP: Dict[str, int] = {"ambig": 0, "disambig": 1}
POL_MAP: Dict[str, int] = {"neg": 0, "nonneg": 1}
_UNKNOWN_TAGS = frozenset({"unknown", "undetermined", "can't determine", "cannot determine"})

def find_unknown_label(answer_info: Dict) -> Optional[int]:
    for idx, key in enumerate(["ans0", "ans1", "ans2"]):
        entry = answer_info.get(key)
        if isinstance(entry, list) and len(entry) > 1:
            tag = str(entry[1]).strip().lower()
            if tag in _UNKNOWN_TAGS:
                return idx
    return None

class Config:
    CATEGORY_MAP: Dict[str, str] = {
        "Age": "age", "Disability_status": "disability",
        "Gender_identity": "gender", "Nationality": "nationality",
        "Physical_appearance": "race", "Race_ethnicity": "race",
        "Religion": "religion", "SES": "socioeconomic",
        "Sexual_orientation": "sexual-orientation",
        "Race_x_SES": "race and socioeconomic status",
    }

    def __init__(self, args: argparse.Namespace):
        self.train_dataset = args.train_dataset
        self.eval_datasets = args.eval_datasets
        self.bbq_categories = args.bbq_categories
        self.n_samples = args.n_samples
        self.bold_path = args.bold_path
        self.toxigen_path = args.toxigen_path
        self.toxigen_threshold = args.toxigen_threshold
        self.model_id = args.model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = args.loss_fn
        self.bias_lambda = args.bias_lambda
        self.bias_margin = args.bias_margin
        self.bold_n_samples = getattr(args, "bold_n_samples", None)
        self.toxigen_n_samples = getattr(args, "toxigen_n_samples", None)
        self.fairmtbench_n_samples = getattr(args, "fairmtbench_n_samples", None)
        self.implicit_bbq_categories = args.implicit_bbq_categories
        self.implicit_bbq_local_dir = args.implicit_bbq_local_dir
        self.biasdpo_dir = getattr(args, "biasdpo_dir", "data/bias-dpo")
        self.biasdpo_categories = getattr(args, "biasdpo_categories", None)
        self.civil_comments_path = getattr(args, "civil_comments_path", "data/civil-comments/train-00000-of-00001.parquet")
        self.civil_toxicity_threshold = getattr(args, "civil_toxicity_threshold", 0.5)
        self.use_cot = args.use_cot
        self.hinge_lambda = args.hinge_lambda
        self.dpo_beta = args.dpo_beta
        self.dpo_ref = args.dpo_ref
        self.kl_lambda = args.kl_lambda
        self.kl_mode = args.kl_mode
        self.label_smooth = args.label_smooth
        self.focal_gamma = args.focal_gamma
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.use_instruction = args.use_instruction
        self.per_epoch_eval = args.per_epoch_eval
        self.stereo_from_metadata = args.stereo_from_metadata
        self.output_dir = Path(args.output_dir)
        self.results_csv = Path(args.results_csv)
        self.run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bbq_single_pass = args.bbq_single_pass
        
        self.run_contamination_check = args.run_contamination_check
        self.contamination_calibration_path = args.contamination_calibration_path
        self.contamination_calibration_n = args.contamination_calibration_n
        self.contamination_min_k_percent = args.contamination_min_k_percent
        self.contamination_max_items = args.contamination_max_items
        self.contamination_ts_guessing_samples = args.contamination_ts_guessing_samples
        self.contamination_ts_guessing_max_new_tokens = args.contamination_ts_guessing_max_new_tokens
        
        self.include_bbq_dpo = getattr(args, "include_bbq_dpo", False)
        self.bbq_dpo_categories = getattr(args, "bbq_dpo_categories", None)
        self.fairmtbench_dimensions = getattr(args, "fairmtbench_dimensions", None)
        self.fairmtbench_groups = getattr(args, "fairmtbench_groups", None)
        
        self.cache_dir = Path(args.cache_dir)
        self.retrain = args.retrain
        self.reeval_baseline = args.reeval_baseline
        self.baseline_only = args.baseline_only
        
        self.bbq_eval_categories = args.bbq_eval_categories if args.bbq_eval_categories is not None else args.bbq_categories
        self.fairmtbench_path = args.fairmtbench_path
        self.fairmtbench_categories = args.fairmtbench_categories
        self.fairmtbench_judge_model = args.fairmtbench_judge_model
        self.fairmtbench_max_new_tokens = args.fairmtbench_max_new_tokens
        self.fairmtbench_judge_max_new_tokens = args.fairmtbench_judge_max_new_tokens
        self.fairmtbench_offload_policy = args.fairmtbench_offload_policy
        self.entropy_lambda = args.entropy_lambda
        
        w = args.condition_weights
        self.condition_weights: Dict[Tuple[str, str], float] = {
            ("ambig", "neg"): w[0], ("ambig", "nonneg"): w[1],
            ("disambig", "neg"): w[2], ("disambig", "nonneg"): w[3],
        }
        self.cf_lambda = args.cf_lambda
        self.cf_margin = args.cf_margin
        self.ipo_tau = args.ipo_tau
        self.train_instruction_names = args.train_instructions
        self.eval_prompt_names = args.eval_prompts
        self.reasoning_max_new_tokens = getattr(args, "reasoning_max_new_tokens", 1024)
        self.reasoning_open_tag = getattr(args, "reasoning_open_tag", "<think>")
        self.reasoning_close_tag = getattr(args, "reasoning_close_tag", "</think>")

        _kl_losses = {"ce_kl", "ce_hinge_kl", "ce_dpo_kl", "ce_dpo_kl_entropy", "ce_dpo_kl_conditioned", "ce_dpo_kl_counterfactual", "ce_dpo_kl_all", "ce_ipo_kl"}
        self.needs_frozen_model = (self.loss_fn in _kl_losses or (self.dpo_ref and self.loss_fn in {"ce_dpo"}) or self.loss_fn in {"ce_ipo", "ce_ipo_kl"})
        _cf_losses = {"ce_dpo_kl_counterfactual", "ce_dpo_kl_all"}
        self.needs_pairs = self.loss_fn in _cf_losses

    def to_dict(self) -> Dict:
        cw = self.condition_weights
        return {
            "model_id": self.model_id, "train_dataset": self.train_dataset,
            "eval_datasets": "|".join(self.eval_datasets), "bbq_categories": "|".join(self.bbq_categories),
            "bbq_eval_categories": "|".join(self.bbq_eval_categories), "n_samples": self.n_samples,
            "loss_fn": self.loss_fn, "bias_lambda": self.bias_lambda, "bias_margin": self.bias_margin,
            "dpo_beta": self.dpo_beta, "dpo_ref": self.dpo_ref, "kl_lambda": self.kl_lambda,
            "kl_mode": self.kl_mode, "label_smooth": self.label_smooth, "focal_gamma": self.focal_gamma,
            "entropy_lambda": self.entropy_lambda, "cf_lambda": self.cf_lambda, "cf_margin": self.cf_margin,
            "ipo_tau": self.ipo_tau, "cond_w_ambig_neg": cw.get(("ambig", "neg"), 2.0),
            "cond_w_ambig_nonneg": cw.get(("ambig", "nonneg"), 1.5), "cond_w_disambig_neg": cw.get(("disambig", "neg"), 1.0),
            "cond_w_disambig_nonneg": cw.get(("disambig", "nonneg"), 0.5), "epochs": self.epochs,
            "lr": self.lr, "batch_size": self.batch_size, "use_instruction": self.use_instruction,
            "per_epoch_eval": self.per_epoch_eval, "stereo_from_metadata": self.stereo_from_metadata,
            "bold_path": str(self.bold_path), "toxigen_path": str(self.toxigen_path),
            "toxigen_threshold": self.toxigen_threshold, "device": self.device,
            "train_instructions": "|".join(self.train_instruction_names),
            "eval_prompts": "|".join(self.eval_prompt_names),
            "reasoning_max_new_tokens": self.reasoning_max_new_tokens,
            "reasoning_open_tag": self.reasoning_open_tag, "reasoning_close_tag": self.reasoning_close_tag,
            "fairmtbench_path": str(self.fairmtbench_path),
            "fairmtbench_categories": "|".join(self.fairmtbench_categories),
            "fairmtbench_judge_model": self.fairmtbench_judge_model,
            "fairmtbench_max_new_tokens": self.fairmtbench_max_new_tokens,
            "fairmtbench_dimensions": "|".join(self.fairmtbench_dimensions) if self.fairmtbench_dimensions else "all",
            "fairmtbench_groups": "|".join(self.fairmtbench_groups) if self.fairmtbench_groups else "all",
            "biasdpo_dir": str(self.biasdpo_dir),
            "biasdpo_categories": "|".join(self.biasdpo_categories) if self.biasdpo_categories else "all",
            "civil_comments_path": str(self.civil_comments_path),
            "civil_toxicity_threshold": self.civil_toxicity_threshold,
            "use_cot": self.use_cot, "hinge_lambda": self.hinge_lambda,
            "bbq_single_pass": self.bbq_single_pass, "cache_dir": str(self.cache_dir),
            "retrain": self.retrain, "reeval_baseline": self.reeval_baseline,
            "baseline_only": self.baseline_only, "include_bbq_dpo": self.include_bbq_dpo,
            "bbq_dpo_categories": "|".join(self.bbq_dpo_categories) if self.bbq_dpo_categories else "all",
            "implicit_bbq_categories": "|".join(self.implicit_bbq_categories),
            "bold_n_samples": self.bold_n_samples, "toxigen_n_samples": self.toxigen_n_samples,
            "fairmtbench_n_samples": self.fairmtbench_n_samples,
        }

    def save(self) -> None:
        path = self.output_dir / "config.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Config saved -> {path}")