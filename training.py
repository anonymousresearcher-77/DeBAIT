from __future__ import annotations
from typing import Dict, List
import json
from collections import defaultdict
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM

from config import Config
from data import BiasDpoDataset, SFTDataset, biasdpo_collate, pad_collate
from losses import compute_biasdpo_loss, compute_loss
from evaluation import evaluate_lightweight

def fine_tune(model, tokenizer, records, cfg) -> torch.nn.Module:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    _PREFERENCE_DATASETS = {"BiasDPO", "CivilComments", "AllBias"}
    is_preference = cfg.train_dataset in _PREFERENCE_DATASETS
    print(f"  Loss fn             : {cfg.loss_fn}\n  Train instructions  : {', '.join(cfg.train_instruction_names) or '(none)'}\n  Eval prompts        : {', '.join(cfg.eval_prompt_names) or '(none)'}")
    
    if is_preference:
        dataset = BiasDpoDataset(records, tokenizer, cfg)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: biasdpo_collate(b, pad_id))
        return _fine_tune_preference(model, tokenizer, loader, cfg)
    
    dataset = SFTDataset(records, tokenizer, cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: pad_collate(b, pad_id))
    return _fine_tune_sft(model, tokenizer, loader, cfg)

def _fine_tune_preference(model, tokenizer, loader, cfg) -> torch.nn.Module:
    _kl_losses = {"ce_kl", "ce_hinge_kl", "ce_dpo_kl", "ce_dpo_kl_entropy", "ce_dpo_kl_conditioned", "ce_dpo_kl_counterfactual", "ce_dpo_kl_all", "ce_ipo_kl", "ce_dpo_hinge_kl"}
    _ipo_losses = {"ce_ipo", "ce_ipo_kl"}
    needs_frozen = cfg.needs_frozen_model or cfg.loss_fn in _kl_losses | _ipo_losses
    frozen_model = None
    if needs_frozen:
        print("  Loading frozen reference model…")
        frozen_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16).to(cfg.device)
        frozen_model.eval()
        for p in frozen_model.parameters(): p.requires_grad_(False)
        
    print(f"  KL mode             : {cfg.kl_mode}\n  DPO reference       : {cfg.dpo_ref}")
    if cfg.loss_fn in ("ce_ipo", "ce_ipo_kl"): print(f"  IPO τ               : {cfg.ipo_tau}\n  Note: IPO already regularises; KL penalty is ablation-only.")
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    epoch_logs: List[Dict] = []
    model.train()
    
    for epoch in range(cfg.epochs):
        accum: Dict[str, float] = defaultdict(float)
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            tti_c = torch.zeros_like(batch["chosen_input_ids"])
            tti_r = torch.zeros_like(batch["rejected_input_ids"])
            chosen_out = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"], token_type_ids=tti_c)
            rejected_out = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"], token_type_ids=tti_r)
            
            frozen_chosen_logits = frozen_rejected_logits = None
            if frozen_model is not None:
                with torch.no_grad():
                    frozen_chosen_logits = frozen_model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"], token_type_ids=tti_c).logits.to(cfg.device)
                    frozen_rejected_logits = frozen_model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"], token_type_ids=tti_r).logits.to(cfg.device)
                    
            total_loss, components = compute_biasdpo_loss(chosen_logits=chosen_out.logits, rejected_logits=rejected_out.logits, chosen_labels=batch["chosen_labels"], rejected_labels=batch["rejected_labels"], cfg=cfg, frozen_chosen_logits=frozen_chosen_logits, frozen_rejected_logits=frozen_rejected_logits)
            
            if not torch.isfinite(total_loss):
                print("    [WARNING] Non-finite loss — skipping batch")
                optimizer.zero_grad()
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            for k, v in components.items(): accum[k] += v.item()
            n_batches += 1
            
        row = {k: round(v / n_batches, 6) for k, v in accum.items()} if n_batches else {}
        row["epoch"] = epoch + 1
        if cfg.per_epoch_eval: row.update(evaluate_lightweight(model, tokenizer, cfg))
        epoch_logs.append(row)
        train_parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(row.items()) if k != "epoch" and not k.startswith("eval_"))
        print(f"    [{epoch+1}/{cfg.epochs}]  TRAIN: {train_parts}")
        
    log_path = cfg.output_dir / "training_log.json"
    with open(log_path, "w") as f: json.dump(epoch_logs, f, indent=2)
    print(f"  Training log saved -> {log_path}")
    return model

def _fine_tune_sft(model, tokenizer, loader, cfg) -> torch.nn.Module:
    frozen_model = None
    if cfg.needs_frozen_model:
        print("  Loading frozen reference model…")
        frozen_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16).to(cfg.device)
        frozen_model.eval()
        for p in frozen_model.parameters(): p.requires_grad_(False)
        
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    epoch_logs: List[Dict] = []
    model.train()
    
    for epoch in range(cfg.epochs):
        accum: Dict[str, float] = defaultdict(float)
        n_batches = 0
        for batch in loader:
            correct_token_ids = batch.pop("correct_token_id").to(cfg.device)
            stereo_token_ids = batch.pop("stereo_token_id").to(cfg.device)
            context_condition_ids = batch.pop("context_condition_id").to(cfg.device)
            question_polarity_ids = batch.pop("question_polarity_id").to(cfg.device)
            paired_input_ids = batch.pop("paired_input_ids").to(cfg.device)
            paired_attention_mask = batch.pop("paired_attention_mask").to(cfg.device)
            has_pair = batch.pop("has_pair").to(cfg.device)
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            
            frozen_logits = None
            if frozen_model is not None:
                with torch.no_grad(): frozen_logits = frozen_model(**batch).logits
                
            outputs = model(**batch)
            paired_logits = None
            if cfg.needs_pairs and has_pair.any():
                paired_out = model(input_ids=paired_input_ids, attention_mask=paired_attention_mask)
                paired_logits = paired_out.logits
                
            total_loss, components = compute_loss(logits=outputs.logits, labels=batch["labels"], correct_token_ids=correct_token_ids, stereo_token_ids=stereo_token_ids, cfg=cfg, frozen_logits=frozen_logits, attention_mask=batch.get("attention_mask"), context_condition_ids=context_condition_ids, question_polarity_ids=question_polarity_ids, paired_logits=paired_logits, has_pair=has_pair)
            
            if not torch.isfinite(total_loss):
                print("    [WARNING] Non-finite loss — skipping batch")
                optimizer.zero_grad()
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            for k, v in components.items(): accum[k] += v.item()
            n_batches += 1
            
        row = {k: round(v / n_batches, 6) for k, v in accum.items()} if n_batches else {}
        row["epoch"] = epoch + 1
        if cfg.per_epoch_eval: row.update(evaluate_lightweight(model, tokenizer, cfg))
        epoch_logs.append(row)
        train_parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(row.items()) if k != "epoch" and not k.startswith("eval_"))
        print(f"    [{epoch+1}/{cfg.epochs}]  TRAIN: {train_parts}")
        
    log_path = cfg.output_dir / "training_log.json"
    with open(log_path, "w") as f: json.dump(epoch_logs, f, indent=2)
    print(f"  Training log saved -> {log_path}")
    return model