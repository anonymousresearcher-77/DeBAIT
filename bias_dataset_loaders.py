from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from bbq_dpo_loader import load_bbq_as_dpo

logger = logging.getLogger(__name__)


BIASDPO_ALL_CATEGORIES: List[str] = [
    "age",
    "disability",
    "gender",
    "nationality",
    "physical_appearance",
    "race",
    "religion",
    "sexual_orientation",
    "socioeconomic",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _read_parquet_files(directory: str, glob: str = "*.parquet") -> pd.DataFrame:
    base = Path(directory)
    files = sorted(base.glob(glob))
    
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        # Normalize schema before concat — rename chosen/rejected if present
        df = df.rename(columns={
            "chosen": "favorable_completion",
            "rejected": "unfavorable_completion"
        })
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True)
    
    dupes = [c for c in combined.columns if list(combined.columns).count(c) > 1]
    if dupes:
        raise ValueError(
            f"Duplicate columns after concat: {dupes}\n"
            f"Check that all parquet files in {directory!r} have consistent schemas."
        )
    
    return combined


def _sample(records: List[Dict], n_samples: Optional[int]) -> List[Dict]:
    if n_samples is not None and n_samples < len(records):
        return random.sample(records, n_samples)
    return records


def _require_columns(df: pd.DataFrame, required: List[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

def load_biasdpo_parquet(
    directory: str,
    categories: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load BiasDPO preference pairs from parquet files.
    """
    df = _read_parquet_files(directory)

    if "category" not in df.columns:
        df["category"] = "original biasdpo"
    else:
        df["category"] = df["category"].fillna("original biasdpo")
        
    _require_columns(
        df,
        ["prompt", "category", "favorable_completion", "unfavorable_completion"],
        source="BiasDPO",
    )

    if categories:
        before = len(df)
        df = df[df["category"].isin(categories)].reset_index(drop=True)
        logger.info(
            "BiasDPO: filtered %d → %d rows by categories %s",
            before, len(df), categories,
        )

    records: List[Dict] = []
    for _, row in df.iterrows():
        # 4. Parse using the standardized column names
        prompt    = str(row["prompt"]).strip()
        chosen    = str(row["favorable_completion"]).strip()
        rejected  = str(row["unfavorable_completion"]).strip()
        category  = str(row["category"]).strip() 

        if not prompt or not chosen or not rejected:
            continue

        records.append(
            {
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
                "category": category,
            }
        )

    records = _sample(records, n_samples)

    print(
        f"  [BiasDPO] {len(records)} preference pairs loaded"
        + (f" (categories: {', '.join(categories)})" if categories else " (all categories)")
    )
    return records

def load_civil_comments_parquet(
    path: str,
    toxicity_threshold: float = 0.5,
    n_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load CivilComments DPO pairs from a single parquet file.
    """
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"CivilComments parquet not found: {path!r}")

    df = pd.read_parquet(fp)

    _require_columns(
        df,
        ["toxicity", "prompt", "favorable_completion", "unfavorable_completion"],
        source="CivilComments",
    )

    records: List[Dict] = []
    for _, row in df.iterrows():
        prompt   = str(row["prompt"]).strip()
        chosen   = str(row["favorable_completion"]).strip()
        rejected = str(row["unfavorable_completion"]).strip()

        if not prompt or not chosen or not rejected:
            continue

        records.append(
            {
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
                "category": None,   # CivilComments has no category column
            }
        )

    records = _sample(records, n_samples)

    print(f"  [CivilComments] {len(records)} preference pairs loaded from {path}")
    return records


def load_bbq_dpo(
    categories: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load BBQ disambiguated examples as DPO preference pairs.
    """

    if isinstance(categories, str):
        categories = [categories]
 
    # Map from BiasDPO / AllBias category names back to BBQ folder names so callers can use either convention.
    _BIASDPO_TO_BBQ: Dict[str, str] = {
        "age":               "Age",
        "disability":        "Disability_status",
        "gender":            "Gender_identity",
        "nationality":       "Nationality",
        "physical_appearance": "Physical_appearance",
        "race":              "Race_ethnicity",
        "religion":          "Religion",
        "socioeconomic":     "SES",
        "sexual_orientation": "Sexual_orientation",
    }
 
    # Default: all BBQ categories
    _ALL_BBQ_CATS = list(_BIASDPO_TO_BBQ.values())
 
    if categories is None:
        bbq_cats = _ALL_BBQ_CATS
    else:
        # Accept either BBQ names ("Race_ethnicity") or BiasDPO names ("race")
        bbq_cats = []
        for c in categories:
            if c in _BIASDPO_TO_BBQ:          # BiasDPO name → convert
                bbq_cats.append(_BIASDPO_TO_BBQ[c])
            elif c in _ALL_BBQ_CATS:           # already a BBQ name
                bbq_cats.append(c)
            else:
                logger.warning("load_bbq_dpo: unknown category %r — skipping", c)
 
    records = load_bbq_as_dpo(categories=bbq_cats, n_samples=n_samples)
    return records

def load_all_bias_datasets(
    biasdpo_dir: str,
    civil_comments_path: str,
    biasdpo_categories: Optional[List[str]] = None,
    civil_toxicity_threshold: float = 0.5,
    n_samples: Optional[int] = None,
    bbq_dpo_categories: Optional[List[str]] = None,
    include_bbq_dpo: bool = True,
) -> List[Dict]:
    """
    Load and shuffle all bias datasets combined (BiasDPO + CivilComments + BBQ-DPO).
    """
    all_records: List[Dict] = []

    try:
        biasdpo_records = load_biasdpo_parquet(biasdpo_dir, categories=biasdpo_categories)
        for r in biasdpo_records:
            r["source"] = "BiasDPO"
        all_records.extend(biasdpo_records)
    except FileNotFoundError as e:
        logger.warning("Skipping BiasDPO (not found): %s", e)

    try:
        civil_records = load_civil_comments_parquet(civil_comments_path, toxicity_threshold=civil_toxicity_threshold)
        for r in civil_records:
            r["source"] = "CivilComments"
        all_records.extend(civil_records)
    except FileNotFoundError as e:
        logger.warning("Skipping CivilComments (not found): %s", e)

    if include_bbq_dpo:
        try:
            if isinstance(bbq_dpo_categories, str):
                bbq_dpo_categories = [c.strip() for c in bbq_dpo_categories.split("|") if c.strip()]
            bbq_records = load_bbq_dpo(categories=bbq_dpo_categories)  # source is set to "BBQ" internally
            all_records.extend(bbq_records)
        except Exception as e:
            logger.warning("Skipping BBQ-DPO: %s", e)

    if not all_records:
        raise RuntimeError(
            "AllBias: no records loaded from any dataset. "
            "Check that at least one dataset path is valid."
        )
 
    random.shuffle(all_records)
    all_records = _sample(all_records, n_samples)
 
    source_counts: Dict[str, int] = {}
    for r in all_records:
        src = r.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    counts_str = "  ".join(f"{s}={n}" for s, n in sorted(source_counts.items()))
    print(f"  [AllBias] {len(all_records)} total preference pairs  ({counts_str})")
 
    return all_records