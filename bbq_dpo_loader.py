"""
BBQ as training data loader for DeBAIT.
Converts BBQ context_condition rows into preference pairs that match the
schema expected by BiasDpoDataset / load_all_bias_datasets.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Dict, List, Optional

from bbq_utils import load_dataset_splits

logger = logging.getLogger(__name__)


_BBQ_TO_BIASDPO_CAT: Dict[str, str] = {
    "Age":                "age",
    "Disability_status":  "disability",
    "Gender_identity":    "gender",
    "Nationality":        "nationality",
    "Physical_appearance":"physical_appearance",
    "Race_ethnicity":     "race",
    "Religion":           "religion",
    "SES":                "socioeconomic",
    "Sexual_orientation":  "sexual_orientation",
}

_UNKNOWN_TAG = "unknown"
LABEL_TOKEN = {0: "A", 1: "B", 2: "C"}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# Each template takes (phrase, clue) for chosen, or (phrase,) for rejected.
# Selection is a deterministic function of example_id (see _pick_template),
# so re-running produces identical output while still varying phrasing
# across rows rather than using one fixed connector.

_CHOSEN_TEMPLATES = [
    lambda phrase, clue: f"{phrase}, since the passage says: \"{clue}\"",
    lambda phrase, clue: f"{phrase}. The context makes this clear: \"{clue}\"",
    lambda phrase, clue: f"{phrase} — this follows directly from \"{clue}\"",
    lambda phrase, clue: f"{phrase}, as indicated by the passage: \"{clue}\"",
    lambda phrase, clue: f"{phrase}. We can tell because \"{clue}\"",
]

_REJECTED_TEMPLATES = [
    lambda phrase: f"{phrase} — there isn't enough context to say for certain.",
    lambda phrase: f"{phrase}, since the passage doesn't provide a clear answer.",
    lambda phrase: f"{phrase}. The information given isn't sufficient to determine this.",
    lambda phrase: f"{phrase}, as the context leaves this ambiguous.",
    lambda phrase: f"{phrase} — the passage doesn't specify enough to know.",
]

_AMBIG_CHOSEN_TEMPLATES = [
    lambda phrase: f"{phrase} — the passage doesn't say enough to tell.",
    lambda phrase: f"{phrase}, since nothing in the context settles this.",
    lambda phrase: f"{phrase}. There's no detail here that points either way.",
    lambda phrase: f"{phrase}, as the passage gives no basis for choosing.",
    lambda phrase: f"{phrase} — the context simply doesn't specify.",
]

_AMBIG_REJECTED_TEMPLATES = [
    lambda phrase: f"{phrase}.",
    lambda phrase: f"{phrase} — that's the obvious answer.",
    lambda phrase: f"{phrase}, clearly.",
    lambda phrase: f"{phrase}, obviously.",
    lambda phrase: f"{phrase} — it has to be.",
]


def _pick_template(pool: list, example_id) -> int:
    """Deterministic, reproducible template selection keyed on example_id."""
    try:
        seed = int(example_id)
    except (TypeError, ValueError):
        seed = abs(hash(str(example_id)))
    return seed % len(pool)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_answer_info(answer_info) -> dict:
    if isinstance(answer_info, dict):
        return answer_info
    if isinstance(answer_info, str):
        try:
            return json.loads(answer_info)
        except (ValueError, TypeError):
            return {}
    return {}


def _find_unknown_label(answer_info: dict) -> Optional[int]:
    for idx in range(3):
        entry = answer_info.get(f"ans{idx}")
        if isinstance(entry, list) and len(entry) > 1:
            if str(entry[1]).strip().lower() == _UNKNOWN_TAG:
                return idx
    return None


def _parse_additional_metadata(additional_metadata) -> dict:
    """additional_metadata is stored as a JSON string in the dataframe,
    same convention as answer_info (see bbq_df_to_hf)."""
    if isinstance(additional_metadata, dict):
        return additional_metadata
    if isinstance(additional_metadata, str):
        try:
            return json.loads(additional_metadata)
        except (ValueError, TypeError):
            return {}
    return {}


def _find_stereotyped_label(
    answer_info: dict,
    additional_metadata: dict,
    unknown_label: int,
    question_polarity: str,
) -> Optional[int]:
    """
    Identify which of the two non-unknown answer options is the
    bias-confirming ("stereotyped") one for THIS specific question.
    """
    stereotyped_groups = additional_metadata.get("stereotyped_groups", [])
    if not stereotyped_groups:
        return None
    stereotyped_groups_lower = {str(g).strip().lower() for g in stereotyped_groups}

    stereotyped_group_idx = None
    for idx in range(3):
        if idx == unknown_label:
            continue
        entry = answer_info.get(f"ans{idx}")
        if isinstance(entry, list) and len(entry) > 1:
            tag = str(entry[1]).strip()
            tag_normalized = re.sub(r'(?<!^)(?=[A-Z])', ' ', tag).lower().strip()
            if tag_normalized in stereotyped_groups_lower or tag.lower() in stereotyped_groups_lower:
                stereotyped_group_idx = idx
                break

    if stereotyped_group_idx is None:
        return None

    polarity = str(question_polarity).strip().lower()
    non_unknown_indices = [i for i in range(3) if i != unknown_label]
    non_stereotyped_idx = next(
        (i for i in non_unknown_indices if i != stereotyped_group_idx), None
    )

    if polarity == "neg":
        return stereotyped_group_idx
    elif polarity == "nonneg":
        return non_stereotyped_idx
    else:
        return None


def _last_sentence(context: str) -> str:
    context = context.strip()
    if not context:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(context) if s.strip()]
    if not sentences:
        return context
    return sentences[-1]


def _format_bbq_prompt(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:
    return (
        f"The following is a multiple-choice question. Select the best option (A, B, or C).\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        f"(A) {ans0}\n"
        f"(B) {ans1}\n"
        f"(C) {ans2}"
    )

def load_bbq_as_dpo(
    categories: List[str],
    n_samples: Optional[int] = None,
    include_ambig: bool = True,
) -> List[Dict]:
    """
    Load BBQ examples as DPO preference pairs, covering both context conditions.
    """
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split("|") if c.strip()]

    records: List[Dict] = []
    skipped_no_unknown = 0
    skipped_true_is_unknown = 0
    skipped_bad_label = 0
    skipped_missing_fields = 0
    skipped_no_stereotype_match = 0   # ambig-only

    for cat in categories:
        df = load_dataset_splits("BBQ", cat)
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        for _, row in df.iterrows():
            condition = str(row.get("context_condition", "")).lower()
            if condition not in ("disambig", "ambig"):
                continue
            if condition == "ambig" and not include_ambig:
                continue

            context  = str(row.get("context", "")).strip()
            question = str(row.get("question", "")).strip()
            ans0     = str(row.get("ans0", "")).strip()
            ans1     = str(row.get("ans1", "")).strip()
            ans2     = str(row.get("ans2", "")).strip()
            if not (context and question and ans0 and ans1 and ans2):
                skipped_missing_fields += 1
                continue

            answer_info = _parse_answer_info(row.get("answer_info", {}))
            unknown_label = _find_unknown_label(answer_info)
            if unknown_label is None:
                skipped_no_unknown += 1
                continue

            ans_by_idx = {0: ans0, 1: ans1, 2: ans2}
            example_id = row.get("example_id")
            norm_cat = _BBQ_TO_BIASDPO_CAT.get(cat, cat.lower())
            prompt = _format_bbq_prompt(context, question, ans0, ans1, ans2)

            # DISAMBIG: chosen=correct, rejected=unknown
            if condition == "disambig":
                try:
                    true_label = int(row.get("label", -1))
                except (TypeError, ValueError):
                    skipped_bad_label += 1
                    continue
                if true_label not in (0, 1, 2):
                    skipped_bad_label += 1
                    continue
                if true_label == unknown_label:
                    skipped_true_is_unknown += 1
                    continue

                chosen_phrase   = ans_by_idx[true_label]
                rejected_phrase = ans_by_idx[unknown_label]
                chosen_letter   = LABEL_TOKEN[true_label]
                rejected_letter = LABEL_TOKEN[unknown_label]

                clue = _last_sentence(context)
                chosen_tpl_idx = _pick_template(_CHOSEN_TEMPLATES, example_id)
                rejected_tpl_idx = _pick_template(_REJECTED_TEMPLATES, example_id)

                if clue:
                    chosen_body = _CHOSEN_TEMPLATES[chosen_tpl_idx](chosen_phrase, clue)
                else:
                    chosen_body = chosen_phrase
                rejected_body = _REJECTED_TEMPLATES[rejected_tpl_idx](rejected_phrase)

            # AMBIG: chosen=unknown, rejected=stereotyped guess
            else:
                additional_metadata = _parse_additional_metadata(
                    row.get("additional_metadata", {})
                )
                question_polarity = row.get("question_polarity", "")
                stereo_idx = _find_stereotyped_label(
                    answer_info, additional_metadata, unknown_label, question_polarity,
                )
                if stereo_idx is None:
                    skipped_no_stereotype_match += 1
                    continue

                chosen_phrase   = ans_by_idx[unknown_label]
                rejected_phrase = ans_by_idx[stereo_idx]
                chosen_letter   = LABEL_TOKEN[unknown_label]
                rejected_letter = LABEL_TOKEN[stereo_idx]

                chosen_tpl_idx = _pick_template(_AMBIG_CHOSEN_TEMPLATES, example_id)
                rejected_tpl_idx = _pick_template(_AMBIG_REJECTED_TEMPLATES, example_id)

                chosen_body = _AMBIG_CHOSEN_TEMPLATES[chosen_tpl_idx](chosen_phrase)
                rejected_body = _AMBIG_REJECTED_TEMPLATES[rejected_tpl_idx](rejected_phrase)

            chosen_text   = f"{chosen_letter}) {chosen_body}"
            rejected_text = f"{rejected_letter}) {rejected_body}"

            records.append({
                "prompt":   prompt,
                "chosen":   chosen_text,
                "rejected": rejected_text,
                "category": norm_cat,
                "source":   "BBQ",
                "_condition": condition,
            })

    n_disambig = sum(1 for r in records if r.get("_condition") == "disambig")
    n_ambig    = sum(1 for r in records if r.get("_condition") == "ambig")

    logger.info(
        "BBQ DPO loader: %d pairs (%d disambig, %d ambig) from %d categories | "
        "skipped: no_unknown=%d, true_is_unknown=%d, bad_label=%d, "
        "missing_fields=%d, no_stereotype_match=%d",
        len(records), n_disambig, n_ambig, len(categories),
        skipped_no_unknown, skipped_true_is_unknown, skipped_bad_label,
        skipped_missing_fields, skipped_no_stereotype_match,
    )
    print(
        f"  [BBQ-DPO] {len(records)} preference pairs "
        f"({n_disambig} disambig, {n_ambig} ambig) "
        f"from categories: {', '.join(categories)}"
        + (f"  (skipped {skipped_no_unknown} no-unknown, "
           f"{skipped_true_is_unknown} true-is-unknown, "
           f"{skipped_bad_label} bad-label, "
           f"{skipped_missing_fields} missing-fields, "
           f"{skipped_no_stereotype_match} no-stereotype-match)"
           if any([skipped_no_unknown, skipped_true_is_unknown,
                   skipped_bad_label, skipped_missing_fields,
                   skipped_no_stereotype_match])
           else "")
    )
    if n_disambig != n_ambig:
        print(
            f"  [BBQ-DPO] NOTE: disambig ({n_disambig}) and ambig ({n_ambig}) "
            f"counts are NOT balanced — likely due to no-stereotype-match "
            f"skips on the ambig side (no disambig-side equivalent exists). "
            f"If you need exact balance, check skipped_no_stereotype_match "
            f"above before assuming the raw BBQ split was uneven."
        )

    for r in records:
        r.pop("_condition", None)

    if n_samples is not None and n_samples < len(records):
        records = random.sample(records, n_samples)

    return records


ANSWER_SUFFIX = "\nAnswer:"


def _debug_compose_prompt(
    prompt: str,
    use_instruction: bool = True,
    instruction_prefix: str = "",
    instruction_suffix: str = "",
    use_real_instruction_template: bool = False,
) -> str:
    if use_real_instruction_template:
        try:
            from prompt_utils import instruction_template, base_prompt_template
            if use_instruction:
                return instruction_template(prompt, dataset_name="BBQ")
            return base_prompt_template(prompt)
        except ImportError:
            print(
                "  [WARN] Could not import prompt_utils' instruction_template — "
                "falling back to an inlined copy. Verify this hasn't drifted "
                "from the real composition logic."
            )

    if not use_instruction:
        return prompt + ANSWER_SUFFIX

    parts = [p for p in (instruction_prefix, prompt, instruction_suffix) if p]
    return "\n\n".join(parts) + ANSWER_SUFFIX


def debug_print_sample_pairs(
    categories: List[str],
    n: int = 5,
    seed: Optional[int] = None,
    use_instruction: bool = True,
    use_real_instruction_template: bool = True,
) -> None:
    if seed is not None:
        random.seed(seed)

    all_records = load_bbq_as_dpo(categories=categories, n_samples=None)

    if not all_records:
        print("  [DEBUG] No BBQ-DPO records loaded — nothing to show.")
        return

    sample = random.sample(all_records, min(n, len(all_records)))

    print("\n" + "=" * 78)
    print(f"  BBQ-DPO DEBUG: {len(sample)} sample pair(s), AS TOKENIZED BY THE MODEL")
    print(f"  (use_instruction={use_instruction}, "
          f"use_real_instruction_template={use_real_instruction_template})")
    print("=" * 78)

    for i, rec in enumerate(sample, 1):
        composed_prompt = _debug_compose_prompt(
            rec["prompt"],
            use_instruction=use_instruction,
            use_real_instruction_template=use_real_instruction_template,
        )

        chosen_sequence   = composed_prompt + " " + rec["chosen"]
        rejected_sequence = composed_prompt + " " + rec["rejected"]

        print(f"\n  ── Sample {i}/{len(sample)}  [category={rec['category']}] ──")
        print(f"\n  [CHOSEN full sequence — this is chosen_input_ids before tokenization]")
        print(f"  {'-' * 74}")
        print("  " + chosen_sequence.replace("\n", "\n  "))
        print(f"\n  [REJECTED full sequence — this is rejected_input_ids before tokenization]")
        print(f"  {'-' * 74}")
        print("  " + rejected_sequence.replace("\n", "\n  "))
        print(f"\n  {'.' * 74}")

    print("\n" + "=" * 78)
    print(f"  END DEBUG OUTPUT — {len(sample)} sample(s) shown")
    print("=" * 78 + "\n")