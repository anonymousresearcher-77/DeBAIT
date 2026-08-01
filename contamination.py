"""
Contamination analysis utilities: Min-K%++ pretraining-data detection and TS-Guessing verbatim-completion probing.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ===========================================================================
# Min-K%++ pretraining-data detection
# ===========================================================================

@torch.no_grad()
def min_k_percent_plus_plus_score(
    model,
    tokenizer,
    text: str,
    device: str,
    k_percent: float = 20.0,
    max_length: int = 1024,
    min_tokens: int = 8,
) -> Optional[float]:
    """
    Compute the Min-K%++ contamination score for a single text under `model`.
    """
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return None

    out = model(**enc)
    logits = out.logits[:, :-1, :].float() 
    targets = input_ids[:, 1:]    

    log_probs = F.log_softmax(logits, dim=-1)       
    probs = log_probs.exp()               
    token_log_probs = log_probs.gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1).squeeze(0)                

    mu = (probs * log_probs).sum(dim=-1).squeeze(0)
    e_log_p_sq = (probs * log_probs.pow(2)).sum(dim=-1).squeeze(0)
    sigma = (e_log_p_sq - mu.pow(2)).clamp(min=1e-6).sqrt()

    z = (token_log_probs - mu) / sigma

    n = z.shape[0]
    if n < min_tokens:
        return None

    k = max(1, int(round(n * k_percent / 100.0)))
    lowest_k, _ = torch.topk(z, k, largest=False)
    return lowest_k.mean().item()


def score_corpus_min_k_plus_plus(
    model,
    tokenizer,
    texts: Sequence[str],
    device: str,
    k_percent: float = 20.0,
    max_length: int = 1024,
    desc: str = "Min-K%++ scoring",
) -> List[Optional[float]]:
    """Score a list of texts; returns one score (or None) per text, same order."""
    model.eval()
    scores: List[Optional[float]] = []
    for text in tqdm(texts, desc=desc):
        scores.append(
            min_k_percent_plus_plus_score(
                model, tokenizer, text, device,
                k_percent=k_percent, max_length=max_length,
            )
        )
    return scores


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def compare_min_k_plus_plus(
    target_scores: List[Optional[float]],
    calibration_scores: List[Optional[float]],
) -> Dict:
    """
    Compare a benchmark's Min-K%++ score distribution against a calibration distribution using a Welch's-t-like z-stat.
    """
    t = [s for s in target_scores if s is not None]
    c = [s for s in calibration_scores if s is not None]
    if not t or not c:
        return {
            "n_target": len(t), "n_calibration": len(c),
            "error": "insufficient scorable items in target and/or calibration set",
        }

    mean_t, mean_c = _mean(t), _mean(c)
    std_t, std_c = _std(t), _std(c)
    shift = mean_t - mean_c

    se = ((std_t ** 2 / max(len(t), 1)) + (std_c ** 2 / max(len(c), 1))) ** 0.5
    z_stat = shift / se if se and se > 0 else float("nan")

    return {
        "n_target": len(t),
        "n_calibration": len(c),
        "mean_target": round(mean_t, 4),
        "mean_calibration": round(mean_c, 4),
        "std_target": round(std_t, 4) if std_t == std_t else None,
        "std_calibration": round(std_c, 4) if std_c == std_c else None,
        "shift": round(shift, 4),
        "z_stat": round(z_stat, 4) if z_stat == z_stat else None,
        "flagged_likely_contaminated": bool(z_stat == z_stat and z_stat > 2.0),
    }


def load_calibration_corpus(
    path: Optional[str],
    n_samples: Optional[int] = None,
    min_chars: int = 200,
    seed: int = 0,
) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration corpus not found at {path!r}.")

    texts: List[str] = []
    if p.suffix.lower() in (".jsonl", ".json"):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(obj.get("text", "")).strip()
                if len(text) >= min_chars:
                    texts.append(text)
    else:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if len(line) >= min_chars:
                    texts.append(line)

    if n_samples is not None and len(texts) > n_samples:
        rng = random.Random(seed)
        texts = rng.sample(texts, n_samples)

    return texts


_WORD_RE = re.compile(r"\w+")
_OPTION_LINE_RE = re.compile(
    r"(?:^|\n)\s*([ABC])[\.\)]\s*(.+?)(?=(?:\n\s*[ABC][\.\)])|\Z)",
    re.DOTALL,
)


def _normalise_for_match(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))


def _token_overlap_ratio(a: str, b: str) -> float:
    """Symmetric token-overlap ratio in [0, 1]; cheap proxy for near-exact match."""
    ta = set(_normalise_for_match(a).split())
    tb = set(_normalise_for_match(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def extract_bbq_options_from_text(text: str) -> Optional[Tuple[str, List[str]]]:
    matches = list(_OPTION_LINE_RE.finditer(text))
    if len(matches) < 3:
        return None
    by_letter = {m.group(1): m.group(2).strip() for m in matches}
    if not all(l in by_letter for l in ("A", "B", "C")):
        return None
    options = [by_letter["A"], by_letter["B"], by_letter["C"]]
    context_and_question = text[: matches[0].start()].strip()
    return context_and_question, options


@dataclass
class TSGuessingItem:
    """One TS-Guessing probe instance for a multiple-choice benchmark item."""
    context: str
    question: str
    options: List[str] 
    masked_index: int
    true_label: Optional[int] = None
    unknown_label: Optional[int] = None
    category: Optional[str] = None
    context_condition: Optional[str] = None
    example_id: Optional[str] = None


def build_ts_guessing_item_from_bbq(rec: Dict, seed: int = 0) -> Optional[TSGuessingItem]:
    true_label = rec.get("label") if rec.get("label") is not None else rec.get("true_label")
    if true_label is None:
        return None
    true_label = int(true_label)

    context, question, options = "", "", None

    raw_opts = [rec.get(f"ans{i}") for i in range(3)]
    if all(o is not None for o in raw_opts):
        options = [str(o) for o in raw_opts]
        context = str(rec.get("context", ""))
        question = str(rec.get("question", ""))
    else:
        parsed = extract_bbq_options_from_text(str(rec.get("text", "")))
        if parsed is None:
            return None
        context, options = parsed[0], parsed[1]
        question = ""

    unknown_label = rec.get("unknown_label")
    candidates = [i for i in range(3) if i != true_label and i != unknown_label]
    if not candidates:
        candidates = [i for i in range(3) if i != true_label]
    if not candidates:
        return None

    seed_material = f"{rec.get('example_id', '')}|{seed}"
    rng = random.Random(int(hashlib.md5(seed_material.encode()).hexdigest(), 16) % (2**31))
    masked_index = rng.choice(candidates)

    return TSGuessingItem(
        context=context,
        question=question,
        options=options,
        masked_index=masked_index,
        true_label=true_label,
        unknown_label=unknown_label,
        category=rec.get("category"),
        context_condition=rec.get("context_condition"),
        example_id=rec.get("example_id"),
    )


def render_ts_guessing_prompt(item: TSGuessingItem) -> str:
    letters = ["A", "B", "C"]
    lines = [item.context]
    if item.question:
        lines += ["", item.question]
    lines.append("")
    for i, opt in enumerate(item.options):
        lines.append(f"{letters[i]}. [MASKED]" if i == item.masked_index else f"{letters[i]}. {opt}")
    lines.append("")
    lines.append(
        f"The text of option {letters[item.masked_index]} has been masked. "
        f"Based on your training data, reproduce the EXACT original wording "
        f"of option {letters[item.masked_index]} as it appears in this "
        f"specific dataset item. Reply with the option text only."
    )
    return "\n".join(lines)


@torch.no_grad()
def run_ts_guessing_probe(
    model,
    tokenizer,
    item: TSGuessingItem,
    device: str,
    max_new_tokens: int = 20,
) -> Dict:
    prompt = render_ts_guessing_prompt(item)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, temperature=1.0,
    )
    guess = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
    true_text = item.options[item.masked_index]

    exact_match = _normalise_for_match(guess) == _normalise_for_match(true_text)
    overlap = _token_overlap_ratio(guess, true_text)

    return {
        "example_id": item.example_id,
        "category": item.category,
        "context_condition": item.context_condition,
        "masked_index": item.masked_index,
        "true_text": true_text,
        "guess": guess,
        "exact_match": exact_match,
        "token_overlap": round(overlap, 4),
        "near_match": bool(overlap >= 0.8),
    }


def run_ts_guessing_suite(
    model,
    tokenizer,
    records: List[Dict],
    device: str,
    n_samples: Optional[int] = 200,
    seed: int = 0,
    max_new_tokens: int = 20,
    desc: str = "TS-Guessing",
) -> Dict:
    model.eval()
    rng = random.Random(seed)

    items: List[TSGuessingItem] = []
    n_skipped_schema = 0
    for rec in records:
        item = build_ts_guessing_item_from_bbq(rec, seed=seed)
        if item is not None:
            items.append(item)
        else:
            n_skipped_schema += 1

    if n_skipped_schema:
        print(
            f"  [TS-Guessing] Skipped {n_skipped_schema}/{len(records)} records "
            f"(no usable answer-option schema found)."
        )

    if n_samples is not None and len(items) > n_samples:
        items = rng.sample(items, n_samples)

    results = [
        run_ts_guessing_probe(model, tokenizer, item, device, max_new_tokens=max_new_tokens)
        for item in tqdm(items, desc=desc)
    ]

    n = len(results)
    n_exact = sum(1 for r in results if r["exact_match"])
    n_near = sum(1 for r in results if r["near_match"])
    exact_rate = 100 * n_exact / n if n else 0.0
    near_rate = 100 * n_near / n if n else 0.0

    per_cat: Dict[str, List[Dict]] = {}
    for r in results:
        per_cat.setdefault(r["category"] or "unknown", []).append(r)
    per_cat_rates = {
        cat: {
            "exact_rate": round(100 * sum(1 for x in rs if x["exact_match"]) / len(rs), 2),
            "near_rate": round(100 * sum(1 for x in rs if x["near_match"]) / len(rs), 2),
            "n": len(rs),
        }
        for cat, rs in per_cat.items()
    }

    return {
        "n_probed": n,
        "n_skipped_schema": n_skipped_schema,
        "exact_match_rate": round(exact_rate, 2),
        "near_match_rate": round(near_rate, 2),
        "per_category": per_cat_rates,
        "per_item": results,
        "note": (
            "Distractor options are frequently short/generic ('the woman', "
            "'not enough info', a name), so a nonzero exact-match rate is "
            "expected by chance. Establish a chance baseline by running this "
            "same probe against a benchmark released AFTER the model's "
            "training cutoff before flagging contamination from these "
            "numbers alone."
        ),
    }


# ===========================================================================
# Caching
# ===========================================================================

def _md5(payload: str) -> str:
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def contamination_cache_key(model_id: str, dataset_name: str, method: str, **params) -> str:
    canonical = f"model={model_id}__ds={dataset_name}__method={method}__" + "__".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    model_slug = model_id.replace("/", "--")
    return f"{model_slug}__{dataset_name}__{method}__{_md5(canonical)}"


def contamination_cache_path(cache_dir, key: str) -> Path:
    return Path(cache_dir) / "contamination" / f"{key}.json"


def save_contamination_result(cache_dir, key: str, result: Dict) -> Path:
    dest = contamination_cache_path(cache_dir, key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(result, f, indent=2)
    return dest


def load_contamination_result(cache_dir, key: str) -> Optional[Dict]:
    dest = contamination_cache_path(cache_dir, key)
    if not dest.exists():
        return None
    with open(dest) as f:
        return json.load(f)


def bbq_records_to_texts(records: List[Dict]) -> List[str]:
    """Flatten BBQ-style records to raw text for Min-K%++ scoring."""
    texts = []
    for r in records:
        if r.get("context") is not None or r.get("question") is not None:
            texts.append(f"{r.get('context', '')} {r.get('question', '')}".strip())
        else:
            texts.append(str(r.get("text", "")))
    return texts


def run_contamination_analysis(
    model,
    tokenizer,
    cfg,
    bbq_records: Optional[List[Dict]] = None,
    other_texts_by_dataset: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    report: Dict[str, Dict] = {}

    calibration_texts = load_calibration_corpus(
        getattr(cfg, "contamination_calibration_path", None),
        n_samples=getattr(cfg, "contamination_calibration_n", 300),
    )
    if not calibration_texts:
        print(
            "  [contamination] WARNING: no --contamination-calibration-path "
            "given (or file empty/missing). Min-K%++ scores will be reported "
            "WITHOUT a calibration baseline -- raw scores only, no z-stat / "
            "flagged_likely_contaminated verdict. Pass a corpus of text "
            "dated after the model's training cutoff for a real comparison."
        )

    k_percent = getattr(cfg, "contamination_min_k_percent", 20.0)
    max_items = getattr(cfg, "contamination_max_items", 300)

    datasets_to_score: Dict[str, List[str]] = {}
    if bbq_records:
        datasets_to_score["BBQ"] = bbq_records_to_texts(bbq_records)
    if other_texts_by_dataset:
        datasets_to_score.update(other_texts_by_dataset)

    calib_scores = None
    if calibration_texts:
        calib_scores = score_corpus_min_k_plus_plus(
            model, tokenizer, calibration_texts, cfg.device,
            k_percent=k_percent, desc="Min-K%++ [calibration]",
        )

    for ds_name, texts in datasets_to_score.items():
        if max_items is not None and len(texts) > max_items:
            rng = random.Random(0)
            texts = rng.sample(texts, max_items)

        key = contamination_cache_key(
            cfg.model_id, ds_name, "min_k_plus_plus", k_percent=k_percent, n=len(texts),
        )
        cached = load_contamination_result(cfg.cache_dir, key)
        if cached is not None:
            print(f"  [contamination] Min-K%++ [{ds_name}] cache HIT")
            report[f"min_k_plus_plus/{ds_name}"] = cached
            continue

        target_scores = score_corpus_min_k_plus_plus(
            model, tokenizer, texts, cfg.device,
            k_percent=k_percent, desc=f"Min-K%++ [{ds_name}]",
        )
        if calib_scores is not None:
            result = compare_min_k_plus_plus(target_scores, calib_scores)
        else:
            valid = [s for s in target_scores if s is not None]
            result = {
                "n_target": len(valid),
                "mean_target": round(_mean(valid), 4) if valid else None,
                "std_target": round(_std(valid), 4) if valid else None,
                "note": "no calibration corpus provided -- see warning above",
            }
        save_contamination_result(cfg.cache_dir, key, result)
        report[f"min_k_plus_plus/{ds_name}"] = result
        print(f"  [contamination] Min-K%++ [{ds_name}]: {result}")

    if bbq_records:
        n_ts = getattr(cfg, "contamination_ts_guessing_samples", 200)
        ts_key = contamination_cache_key(cfg.model_id, "BBQ", "ts_guessing", n=n_ts)
        cached_ts = load_contamination_result(cfg.cache_dir, ts_key)
        if cached_ts is not None:
            print("  [contamination] TS-Guessing [BBQ] cache HIT")
            report["ts_guessing/BBQ"] = cached_ts
        else:
            ts_result = run_ts_guessing_suite(
                model, tokenizer, bbq_records, cfg.device,
                n_samples=n_ts,
                max_new_tokens=getattr(cfg, "contamination_ts_guessing_max_new_tokens", 20),
                desc="TS-Guessing [BBQ]",
            )
            save_contamination_result(cfg.cache_dir, ts_key, ts_result)
            report["ts_guessing/BBQ"] = ts_result
            print(
                f"  [contamination] TS-Guessing [BBQ]: "
                f"exact={ts_result['exact_match_rate']}%  "
                f"near={ts_result['near_match_rate']}%  "
                f"(n={ts_result['n_probed']})"
            )

    return report
