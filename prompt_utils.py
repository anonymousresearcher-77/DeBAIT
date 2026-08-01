from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from prompts import INSTRUCTION_REGISTRY, EVAL_PROMPT_REGISTRY, GENERIC_INSTRUCTION

ANSWER_SUFFIX = "\nAnswer:"

def _compose_from_registry(registry: Dict[str, Tuple[str, str]], names: List[str], position_filter: str, category: Optional[str] = None) -> str:
    parts: List[str] = []
    for name in names:
        if name not in registry: continue
        text, pos = registry[name]
        if isinstance(text, (tuple, list)): text = "\n".join(text)
        if pos in (position_filter,) or (position_filter == "prefix" and pos == "system"):
            rendered = text.format(category=category) if "{category}" in text else text
            parts.append(rendered)
    return "\n".join(parts)

def compose_instruction_prefix(names: List[str], category: Optional[str] = None) -> str:
    return _compose_from_registry(INSTRUCTION_REGISTRY, names, "prefix", category)

def compose_instruction_suffix(names: List[str]) -> str:
    return _compose_from_registry(INSTRUCTION_REGISTRY, names, "suffix")

def compose_eval_prefix(names: List[str], category: Optional[str] = None) -> str:
    return _compose_from_registry(EVAL_PROMPT_REGISTRY, names, "prefix", category)

def compose_eval_suffix(names: List[str], category: Optional[str] = None) -> str:
    return _compose_from_registry(EVAL_PROMPT_REGISTRY, names, "suffix", category)

def instruction_template(text: str, context_condition: Optional[str] = None, question_polarity: Optional[str] = None, dataset_name: str = "BiasDPO", instruction_names: Optional[List[str]] = None, category: Optional[str] = None) -> str:
    instr = GENERIC_INSTRUCTION
    prefix = compose_instruction_prefix(instruction_names or [], category=category)
    suffix = compose_instruction_suffix(instruction_names or [])
    parts = [p for p in (prefix, instr, text, suffix) if p]
    return "\n".join(parts) + ANSWER_SUFFIX

def base_prompt_template(text: str) -> str:
    return text + ANSWER_SUFFIX

def eval_prompt_template(text: str, eval_prompt_names: Optional[List[str]] = None, category: Optional[str] = None) -> str:
    prefix = compose_eval_prefix(eval_prompt_names or [], category=category)
    suffix = compose_eval_suffix(eval_prompt_names or [], category=category)
    parts = [p for p in (prefix, text, suffix) if p]
    return "\n".join(parts) + ANSWER_SUFFIX