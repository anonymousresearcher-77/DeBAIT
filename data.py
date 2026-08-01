from __future__ import annotations
from typing import Dict, List, Optional
import json
import hashlib
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from config import Config, LABEL_TOKEN, CTX_COND_MAP, POL_MAP, find_unknown_label
from prompt_utils import instruction_template, base_prompt_template
from bias_dataset_loaders import load_biasdpo_parquet, load_civil_comments_parquet, load_all_bias_datasets, BIASDPO_ALL_CATEGORIES
from bbq_utils import load_dataset_splits

logger = logging.getLogger(__name__)

_COT_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "race": ["race", "racial", "black", "white", "asian", "hispanic", "latino", "african", "ethnicity"],
    "gender": ["gender", "male", "female", "woman", "man", "transgender", "nonbinary", "she ", "he "],
    "religion": ["religion", "muslim", "christian", "jewish", "hindu", "buddhist", "atheist", "faith"],
    "sexual_orientation": ["gay", "lesbian", "bisexual", "lgbtq", "homosexual", "heterosexual", "queer"],
    "disability": ["disability", "disabled", "wheelchair", "blind", "deaf", "impairment", "mental health"],
    "age": ["elderly", "young", "old ", "teenager", "senior", "millennial", "boomer", "aging"],
}

def _infer_bias_category(text: str) -> Optional[str]:
    text_lower = text.lower()
    for cat, keywords in _COT_CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return cat
    return None

def _parse_answer_info(rec: Dict) -> Dict:
    ai = rec.get("answer_info", {})
    if isinstance(ai, str):
        try: ai = json.loads(ai)
        except (ValueError, json.JSONDecodeError): ai = {}
    return ai

def find_stereotypical_label_from_metadata(row, unknown_label=None):
    target_loc = row.get("target_loc")
    if target_loc is not None:
        try: return int(target_loc)
        except (TypeError, ValueError): pass
    answer_info = row.get("answer_info")
    if answer_info and isinstance(answer_info, dict):
        true_label = row.get("label") or row.get("true_label")
        candidates = [i for i in range(3) if i != true_label and i != unknown_label]
        if len(candidates) == 1: return candidates[0]
        for idx in candidates:
            key, info = f"ans{idx}", answer_info.get(f"ans{idx}", [])
            if info and isinstance(info, list):
                for entry in info:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        if "unknown" not in str(entry[1]).lower(): return idx
    return None

def resolve_stereo_token_id(row, cfg, tokenizer, unknown_label, model_prediction, true_label):
    stereo_label = None
    if cfg.stereo_from_metadata and cfg.train_dataset == "BBQ":
        stereo_label = find_stereotypical_label_from_metadata(row, unknown_label=unknown_label)
    if stereo_label is not None and stereo_label == true_label:
        stereo_label = None
    if stereo_label is None and model_prediction is not None:
        if model_prediction != true_label: stereo_label = model_prediction
    if stereo_label is None or stereo_label not in LABEL_TOKEN:
        return -1
    return _label_token_id(tokenizer, stereo_label)

def _label_token_id(tokenizer, label: int) -> int:
    letter = LABEL_TOKEN[label]
    for attempt in [letter, f" {letter}", f": {letter}"]:
        ids = tokenizer.encode(attempt, add_special_tokens=False)
        if ids: return ids[-1]
    raise RuntimeError(f"Could not obtain token-id for label '{letter}'.")

def load_train_data(cfg: Config) -> List[Dict]:
    if cfg.train_dataset == "BiasDPO":
        return load_biasdpo_parquet(cfg.biasdpo_dir, categories=cfg.biasdpo_categories, n_samples=cfg.n_samples)
    elif cfg.train_dataset == "CivilComments":
        return load_civil_comments_parquet(cfg.civil_comments_path, toxicity_threshold=cfg.civil_toxicity_threshold, n_samples=cfg.n_samples)
    elif cfg.train_dataset == "AllBias":
        return load_all_bias_datasets(
            biasdpo_dir=cfg.biasdpo_dir, civil_comments_path=cfg.civil_comments_path,
            biasdpo_categories=cfg.biasdpo_categories, civil_toxicity_threshold=cfg.civil_toxicity_threshold,
            n_samples=cfg.n_samples, include_bbq_dpo=cfg.include_bbq_dpo, bbq_dpo_categories=cfg.bbq_dpo_categories,
        )
    raise ValueError(f"Unknown train dataset: {cfg.train_dataset!r}")

def load_bbq(cfg: Config) -> List[Dict]:
    all_data: List[Dict] = []
    for cat in cfg.bbq_categories:
        df = load_dataset_splits("BBQ", cat)
        if hasattr(df, "to_pandas"): df = df.to_pandas()
        all_data.extend(df.to_dict("records"))
    return all_data

def _load_bbq_eval_records_for_contamination(cfg: Config) -> List[Dict]:
    records: List[Dict] = []
    for cat in cfg.bbq_eval_categories:
        df = load_dataset_splits("BBQ", cat)
        if hasattr(df, "to_pandas"): df = df.to_pandas()
        recs = df.to_dict("records")
        for rec in recs:
            rec["unknown_label"] = find_unknown_label(_parse_answer_info(rec))
        records.extend(recs)
    return records

def _gather_other_contamination_texts(cfg: Config) -> Dict[str, List[str]]:
    texts: Dict[str, List[str]] = {}
    if "BOLD" in cfg.eval_datasets and Path(cfg.bold_path).exists():
        try: texts["BOLD"] = [r["prompt"] for r in load_bold(cfg.bold_path)]
        except FileNotFoundError: pass
    if "ToxiGen" in cfg.eval_datasets and Path(cfg.toxigen_path).exists():
        try: texts["ToxiGen"] = [r["text"] for r in load_toxigen(cfg.toxigen_path, threshold=cfg.toxigen_threshold)]
        except FileNotFoundError: pass
    if "FairMTBench" in cfg.eval_datasets and Path(cfg.fairmtbench_path).exists():
        try:
            recs = load_fairmtbench(cfg.fairmtbench_path, categories=cfg.fairmtbench_categories, dimensions=getattr(cfg, "fairmtbench_dimensions", None), groups=getattr(cfg, "fairmtbench_groups", None))
            texts["FairMTBench"] = [" ".join(r["turns"]) for r in recs]
        except FileNotFoundError: pass
    return texts

_BOLD_DOMAIN_FILES: Dict[str, str] = {
    "race": "race_prompt.json", "gender": "gender_prompt.json", "religion": "religious_ideology_prompt.json",
    "profession": "profession_prompt.json", "political_ideology": "political_ideology_prompt.json",
}

def load_bold(bold_dir: str, domains=None, n_per_group=None) -> List[Dict]:
    bold_path = Path(bold_dir)
    if not bold_path.exists(): raise FileNotFoundError(f"BOLD data directory not found at {bold_dir!r}.")
    target_files = {k: v for k, v in _BOLD_DOMAIN_FILES.items() if domains is None or k in domains}
    records: List[Dict] = []
    for domain, filename in target_files.items():
        fpath = bold_path / filename
        if not fpath.exists():
            logger.warning("BOLD domain file not found, skipping: %s", fpath)
            continue
        with open(fpath) as f: data = json.load(f)
        for group_name, prompts in data.items():
            if isinstance(prompts, dict): prompts = list(prompts.values())
            flat = [str(item) for sublist in prompts if isinstance(sublist, list) for item in sublist] + [str(item) for item in prompts if not isinstance(item, list)]
            if n_per_group is not None: flat = flat[:n_per_group]
            for prompt_text in flat:
                if str(prompt_text).strip(): records.append({"domain": domain, "group": group_name, "prompt": str(prompt_text).strip()})
    return records

_TOXIGEN_GROUP_ALIASES: Dict[str, str] = {
    "black": "black", "african american": "black", "african_american": "black", "asian": "asian", "chinese": "asian",
    "japanese": "asian", "korean": "asian", "latino": "latino", "latinx": "latino", "mexican": "latino",
    "native american": "native_american", "native_american": "native_american", "jewish": "jewish", "muslim": "muslim",
    "middle eastern": "middle_eastern", "middle_eastern": "middle_eastern", "lgbtq": "lgbtq", "lgbtq+": "lgbtq",
    "mental disability": "mental_disability", "mental_dis": "mental_disability", "physical disability": "physical_disability",
    "physical_dis": "physical_disability", "women": "women", "female": "women",
}

def _normalise_toxigen_group(raw: str) -> str:
    key = raw.strip().lower().replace("-", " ").replace("_", " ")
    if key in _TOXIGEN_GROUP_ALIASES: return _TOXIGEN_GROUP_ALIASES[key]
    for alias, canonical in _TOXIGEN_GROUP_ALIASES.items():
        if alias in key or key in alias: return canonical
    return key.replace(" ", "_")

def load_toxigen(path: str, threshold=2.5, n_samples=None, target_groups=None) -> List[Dict]:
    toxigen_path = Path(path)
    if not toxigen_path.exists(): raise FileNotFoundError(f"ToxiGen data not found at {path!r}.")
    records, suffix = [], toxigen_path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        with open(toxigen_path) as f:
            for line in f:
                if line.strip():
                    try: records.append(json.loads(line))
                    except json.JSONDecodeError: pass
        if not records:
            with open(toxigen_path) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list): records = data
                except json.JSONDecodeError: pass
    elif suffix in (".csv", ".tsv"):
        records = pd.read_csv(toxigen_path, sep="\t" if suffix == ".tsv" else ",").to_dict("records")
    
    processed: List[Dict] = []
    for rec in records:
        text = str(rec.get("text") or rec.get("statement") or rec.get("generation") or rec.get("prompt", "")).strip()
        if not text: continue
        group = _normalise_toxigen_group(str(rec.get("target_group") or rec.get("group") or rec.get("minority_group") or "unknown").strip())
        if target_groups and group not in target_groups: continue
        
        gold_label = None
        if "label" in rec:
            raw_label = rec["label"]
            if isinstance(raw_label, str): gold_label = "toxic" if raw_label.strip().lower() in ("toxic", "1", "true", "yes", "hate") else "benign"
            elif isinstance(raw_label, (int, float)): gold_label = "toxic" if float(raw_label) >= 1.0 else "benign"
        
        score_field = rec.get("toxicity_score") or rec.get("toxicity") or rec.get("score") or rec.get("toxicity_human")
        toxicity_score = None
        if score_field is not None:
            try:
                toxicity_score = float(score_field)
                if gold_label is None: gold_label = "toxic" if toxicity_score >= threshold else "benign"
            except (TypeError, ValueError): pass
        if gold_label is None: gold_label = "benign"
        processed.append({"text": text, "target_group": group, "gold_label": gold_label, "toxicity_score": toxicity_score})

    if n_samples is not None and len(processed) > n_samples:
        group_map = defaultdict(list)
        for item in processed: group_map[item["target_group"]].append(item)
        samples_per_group = max(1, n_samples // len(group_map))
        stratified_processed = []
        for grp in sorted(group_map.keys()):
            sorted_by_hash = sorted(group_map[grp], key=lambda x: hashlib.md5(str(x["text"]).encode("utf-8")).hexdigest())
            stratified_processed.extend(sorted_by_hash[:samples_per_group])
        processed = stratified_processed[:n_samples]
    return processed

FAIRMT10K_DIMENSIONS = ["Stereotype", "Toxicity"]

def _conv_dict_to_turns(conv: Dict) -> List[str]:
    try: ordered_keys = sorted(conv.keys(), key=lambda k: int(k))
    except (TypeError, ValueError): ordered_keys = sorted(conv.keys())
    return [str(conv[k]).strip() for k in ordered_keys if str(conv[k]).strip()]

def load_fairmtbench(path: str, categories=None, dimensions=None, groups=None, n_samples=None) -> List[Dict]:
    base = Path(path)
    if not base.exists(): raise FileNotFoundError(f"FairMTBench data not found at {path!r}.")
    records = load_fairmtbench_1k(path, categories=categories)
    if n_samples is not None and len(records) > n_samples:
        facet_map = defaultdict(list)
        for r in records: facet_map[(r.get("dimension"), r.get("category"), r.get("group"))].append(r)
        samples_per_facet = max(1, n_samples // len(facet_map))
        stratified_records = []
        for facet in sorted(facet_map.keys(), key=lambda x: str(x)):
            sorted_by_hash = sorted(facet_map[facet], key=lambda x: hashlib.md5(f"{x.get('conv_id', 0)}___{''.join(x['turns'])}".encode("utf-8")).hexdigest())
            stratified_records.extend(sorted_by_hash[:samples_per_facet])
        records = stratified_records[:n_samples]
    return records

def load_fairmtbench_1k(path: str, categories=None) -> List[Dict]:
    base = Path(path)
    if not base.exists(): raise FileNotFoundError(f"FairMTBench data not found at {path!r}.")
    if categories is None: categories = ["Anaphora_Ellipsis", "Fixed_Format", "Interference_Misinformation", "Jailbreak_Tips", "Negative_Feedback", "Scattered_Questions"]
    records: List[Dict] = []
    for cat in categories:
        fp = base / f"{cat}.json"
        if not fp.exists():
            logger.warning("FairMTBench category file missing, skipping: %s", fp)
            continue
        with open(fp) as f: data = json.load(f)
        for idx, conv in enumerate(data):
            turns = _conv_dict_to_turns(conv)
            if turns: records.append({"dimension": None, "category": cat, "group": None, "conv_id": idx, "turns": turns})
    return records

def _pad_seq(t: torch.Tensor, max_len: int, pad_val: int) -> torch.Tensor:
    diff = max_len - t.shape[0]
    return torch.cat([t, torch.full((diff,), pad_val, dtype=torch.long)]) if diff > 0 else t

def biasdpo_collate(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    c_max = max(b["chosen_input_ids"].shape[0] for b in batch)
    r_max = max(b["rejected_input_ids"].shape[0] for b in batch)
    return {
        "chosen_input_ids": torch.stack([_pad_seq(b["chosen_input_ids"], c_max, pad_id) for b in batch]),
        "chosen_attention_mask": torch.stack([_pad_seq(b["chosen_attention_mask"], c_max, 0) for b in batch]),
        "chosen_labels": torch.stack([_pad_seq(b["chosen_labels"], c_max, -100) for b in batch]),
        "rejected_input_ids": torch.stack([_pad_seq(b["rejected_input_ids"], r_max, pad_id) for b in batch]),
        "rejected_attention_mask": torch.stack([_pad_seq(b["rejected_attention_mask"], r_max, 0) for b in batch]),
        "rejected_labels": torch.stack([_pad_seq(b["rejected_labels"], r_max, -100) for b in batch]),
    }

class BiasDpoDataset(TorchDataset):
    def __init__(self, records: List[Dict], tokenizer, cfg: Config, max_length: int = 512):
        self.items: List[Dict] = []
        skipped = 0
        for rec in records:
            prompt, chosen, rejected = rec["prompt"], rec["chosen"], rec["rejected"]
            bias_cat = rec.get("bias_type") or rec.get("category") or _infer_bias_category(prompt)
            prompt_text = instruction_template(prompt, dataset_name=cfg.train_dataset, instruction_names=cfg.train_instruction_names, category=bias_cat) if cfg.use_instruction else base_prompt_template(prompt)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")["input_ids"].squeeze(0)
            prompt_len = prompt_ids.shape[0]
            
            def _encode_pair(continuation: str):
                cont_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
                max_cont = max_length - prompt_len - 1
                if max_cont <= 0: return None, None, None
                cont_ids = cont_ids[:max_cont]
                input_ids = torch.cat([prompt_ids, cont_ids])
                attn_mask = torch.ones_like(input_ids)
                labels = torch.full_like(input_ids, -100)
                labels[prompt_len:] = cont_ids
                return input_ids, attn_mask, labels

            c_ids, c_mask, c_labels = _encode_pair(chosen)
            r_ids, r_mask, r_labels = _encode_pair(rejected)
            if c_ids is None or r_ids is None:
                skipped += 1
                continue
            self.items.append({"chosen_input_ids": c_ids, "chosen_attention_mask": c_mask, "chosen_labels": c_labels, "rejected_input_ids": r_ids, "rejected_attention_mask": r_mask, "rejected_labels": r_labels})
        if skipped: print(f"  [BiasDpoDataset] Skipped {skipped} samples (prompt too long)")
        print(f"  [BiasDpoDataset] {len(self.items)} preference pairs")
        if not self.items: raise RuntimeError("BiasDpoDataset is empty — check dataset paths.")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def is_biased(record: Dict, dataset_name: str) -> bool:
    pred = record.get("prediction")
    if pred is None: return False
    ctx, unknown_label = record.get("context_condition", ""), record.get("unknown_label")
    if ctx == "ambig": return (unknown_label is not None) and (pred != unknown_label)
    elif ctx == "disambig":
        stereo_target = record.get("stereotypical_label")
        if stereo_target is not None: return pred == stereo_target and pred != record.get("true_label")
        return pred != record.get("true_label")
    return False

def build_bbq_pair_lookup(records: List[Dict]) -> Dict[int, int]:
    key_to_ctx: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for i, rec in enumerate(records):
        ctx = rec.get("context_condition", "")
        if ctx not in ("ambig", "disambig"): continue
        q_idx = rec.get("question_index") or rec.get("example_id") or rec.get("q_id")
        cat, pol = rec.get("category", ""), rec.get("question_polarity", "")
        pair_key = (f"{cat}|{q_idx}|{pol}" if q_idx is not None else f"{cat}|{str(rec.get('question',''))[:160]}|{pol}")
        key_to_ctx[pair_key][ctx].append(i)
    pair_lookup: Dict[int, int] = {}
    for _, ctx_map in key_to_ctx.items():
        ambig_idxs, disambig_idxs = ctx_map.get("ambig", []), ctx_map.get("disambig", [])
        if not ambig_idxs or not disambig_idxs: continue
        for di in disambig_idxs: pair_lookup[di] = ambig_idxs[0]
    return pair_lookup

class SFTDataset(TorchDataset):
    def __init__(self, records, tokenizer, cfg, max_length=512):
        self.items = []
        skipped, n_stereo_from_metadata, n_stereo_from_pred, n_stereo_unresolved = [], 0, 0, 0
        pair_lookup = build_bbq_pair_lookup(records) if cfg.needs_pairs and cfg.train_dataset == "BBQ" else {}
        
        for rec_idx, rec in enumerate(records):
            text, true_label = rec.get("text", ""), rec.get("true_label")
            if true_label is None or true_label not in LABEL_TOKEN:
                skipped.append(rec); continue
            
            correct_token_id = _label_token_id(tokenizer, true_label)
            biased = is_biased(rec, cfg.train_dataset)
            prompt = instruction_template(text, context_condition=rec.get("context_condition"), question_polarity=rec.get("question_polarity"), dataset_name=cfg.train_dataset, instruction_names=cfg.train_instruction_names, category=rec.get("category")) if cfg.use_instruction else base_prompt_template(text)
            
            stereo_token_id = -1
            if biased:
                stereo_token_id = resolve_stereo_token_id(row=rec, cfg=cfg, tokenizer=tokenizer, unknown_label=rec.get("unknown_label"), model_prediction=rec.get("prediction"), true_label=true_label)
                meta_stereo = rec.get("stereotypical_label")
                if stereo_token_id != -1:
                    if (meta_stereo is not None and meta_stereo != true_label and cfg.stereo_from_metadata): n_stereo_from_metadata += 1
                    else: n_stereo_from_pred += 1
                else: n_stereo_unresolved += 1

            ctx_str, pol_str = rec.get("context_condition", ""), rec.get("question_polarity", "")
            prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length - 1, return_tensors="pt")
            prompt_ids, prompt_mask = prompt_enc["input_ids"].squeeze(0), prompt_enc["attention_mask"].squeeze(0)
            prompt_len = prompt_ids.shape[0]
            
            label_id_t = torch.tensor([correct_token_id], dtype=torch.long)
            input_ids = torch.cat([prompt_ids, label_id_t])
            attention_mask = torch.cat([prompt_mask, torch.ones(1, dtype=torch.long)])
            labels = torch.full_like(input_ids, -100)
            labels[prompt_len] = correct_token_id

            paired_idx = pair_lookup.get(rec_idx) if cfg.needs_pairs else None
            if paired_idx is not None:
                p_rec, p_text = records[paired_idx], records[paired_idx].get("text", "")
                p_prompt = instruction_template(p_text, context_condition=p_rec.get("context_condition"), question_polarity=p_rec.get("question_polarity"), dataset_name=cfg.train_dataset, instruction_names=cfg.train_instruction_names, category=p_rec.get("category")) if cfg.use_instruction else base_prompt_template(p_text)
                p_true_label = int(p_rec.get("true_label", 0)) if int(p_rec.get("true_label", 0)) in LABEL_TOKEN else 0
                p_label_id = torch.tensor([_label_token_id(tokenizer, p_true_label)], dtype=torch.long)
                p_enc = tokenizer(p_prompt, truncation=True, max_length=max_length - 1, return_tensors="pt")
                paired_input_ids = torch.cat([p_enc["input_ids"].squeeze(0), p_label_id])
                paired_attention_mask = torch.cat([p_enc["attention_mask"].squeeze(0), torch.ones(1, dtype=torch.long)])
                has_pair = torch.tensor(1, dtype=torch.long)
            else:
                paired_input_ids, paired_attention_mask = torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)
                has_pair = torch.tensor(0, dtype=torch.long)

            self.items.append((input_ids, attention_mask, labels, correct_token_id, stereo_token_id, CTX_COND_MAP.get(ctx_str, -1), POL_MAP.get(pol_str, -1), paired_input_ids, paired_attention_mask, has_pair))
        
        if skipped: print(f"  [SFTDataset] Skipped {len(skipped)} samples (invalid true_label)")
        print(f"  [SFTDataset] {len(self.items)} samples ({sum(1 for (_, _, _, _, s, *_) in self.items if s != -1)} with DPO signal)")
        if not self.items: raise RuntimeError("SFTDataset is empty.")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        input_ids, attention_mask, labels, correct_token_id, stereo_token_id, context_condition_id, question_polarity_id, paired_input_ids, paired_attention_mask, has_pair = self.items[idx]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "correct_token_id": torch.tensor(correct_token_id, dtype=torch.long), "stereo_token_id": torch.tensor(stereo_token_id, dtype=torch.long), "context_condition_id": torch.tensor(context_condition_id, dtype=torch.long), "question_polarity_id": torch.tensor(question_polarity_id, dtype=torch.long), "paired_input_ids": paired_input_ids, "paired_attention_mask": paired_attention_mask, "has_pair": has_pair}

def pad_collate(batch, pad_id):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_vals = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
    padded = {k: torch.stack([torch.cat([b[k], torch.full((max_len - b[k].shape[0],), pad_vals[k], dtype=torch.long)]) for b in batch]) for k in pad_vals}
    for key in ("correct_token_id", "stereo_token_id", "context_condition_id", "question_polarity_id", "has_pair"):
        padded[key] = torch.stack([b[key] for b in batch])
    max_paired_len = max(b["paired_input_ids"].shape[0] for b in batch)
    padded["paired_input_ids"] = torch.stack([torch.cat([b["paired_input_ids"], torch.full((max_paired_len - b["paired_input_ids"].shape[0],), pad_id, dtype=torch.long)]) for b in batch])
    padded["paired_attention_mask"] = torch.stack([torch.cat([b["paired_attention_mask"], torch.zeros(max_paired_len - b["paired_attention_mask"].shape[0], dtype=torch.long)]) for b in batch])
    return padded