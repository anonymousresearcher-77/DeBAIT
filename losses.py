from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from config import Config

def _sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    lp = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    safe = shift_labels.clone()
    safe[~mask] = 0
    token_lp = lp.gather(2, safe.unsqueeze(2)).squeeze(2) * mask.float()
    return token_lp.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)

def _ce_loss(shift_logits, shift_labels, vocab, label_smoothing=0.0):
    return F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1), ignore_index=-100, label_smoothing=label_smoothing, reduction="mean")

def _get_label_logits(shift_logits, shift_labels):
    B = shift_logits.shape[0]
    label_mask = (shift_labels != -100)
    has_label = label_mask.any(dim=1)
    flipped_pos = label_mask.flip(dims=[1]).long().argmax(dim=1)
    label_pos = (shift_labels.shape[1] - 1) - flipped_pos
    label_pos = label_pos * has_label.long()
    return shift_logits[torch.arange(B, device=shift_logits.device), label_pos], has_label

def _bias_dpo(label_logits, correct_token_ids, stereo_token_ids, has_label, vocab, beta, ref_label_logits=None, use_ref=True):
    is_biased_mask = (stereo_token_ids != -1) & has_label
    if not is_biased_mask.any(): return label_logits.new_zeros(())
    idx, n = is_biased_mask.nonzero(as_tuple=True)[0], is_biased_mask.sum().item()
    arange = torch.arange(n, device=label_logits.device)
    lp = F.log_softmax(label_logits[idx], dim=-1)
    correct_lp = lp[arange, correct_token_ids[idx].clamp(0, vocab - 1)]
    stereo_lp = lp[arange, stereo_token_ids[idx].clamp(0, vocab - 1)]
    policy_margin = correct_lp - stereo_lp
    ref_margin = label_logits.new_zeros(n)
    if use_ref and ref_label_logits is not None:
        ref_lp = F.log_softmax(ref_label_logits[idx], dim=-1)
        ref_margin = ref_lp[arange, correct_token_ids[idx].clamp(0, vocab - 1)] - ref_lp[arange, stereo_token_ids[idx].clamp(0, vocab - 1)]
    return torch.nan_to_num(-F.logsigmoid(beta * (policy_margin - ref_margin)), nan=0.0).sum() / n

def _bias_ipo(chosen_logits, rejected_logits, chosen_labels, rejected_labels, tau, frozen_chosen_logits=None, frozen_rejected_logits=None):
    lp_chosen, lp_rejected = _sequence_logprob(chosen_logits, chosen_labels), _sequence_logprob(rejected_logits, rejected_labels)
    if frozen_chosen_logits is not None and frozen_rejected_logits is not None:
        ref_lp_chosen, ref_lp_rejected = _sequence_logprob(frozen_chosen_logits, chosen_labels), _sequence_logprob(frozen_rejected_logits, rejected_labels)
    else:
        ref_lp_chosen, ref_lp_rejected = torch.zeros_like(lp_chosen), torch.zeros_like(lp_rejected)
    margin = (lp_chosen - ref_lp_chosen) - (lp_rejected - ref_lp_rejected)
    return torch.nan_to_num((margin - torch.full_like(margin, 1.0 / (2.0 * tau))).pow(2).mean(), nan=0.0)

def _kl_regularisation(logits, frozen_logits, labels, attention_mask=None, mode="full_sequence"):
    shift_s, shift_t = logits[:, :-1, :].contiguous(), frozen_logits[:, :-1, :].contiguous()
    valid = (attention_mask[:, 1:] == 1) if mode == "full_sequence" and attention_mask is not None else (labels[:, 1:] != -100)
    if not valid.any(): return logits.new_zeros(())
    per_pos = F.kl_div(F.log_softmax(shift_s, dim=-1), F.softmax(shift_t, dim=-1), reduction="none").sum(dim=-1)
    return (per_pos * valid.float()).sum() / valid.sum().clamp(min=1)

def compute_biasdpo_loss(chosen_logits, rejected_logits, chosen_labels, rejected_labels, cfg: Config, frozen_chosen_logits=None, frozen_rejected_logits=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    _, _, vocab = chosen_logits.shape
    components: Dict[str, torch.Tensor] = {}
    shift_c_logits, shift_c_labels = chosen_logits[:, :-1, :].contiguous(), chosen_labels[:, 1:].contiguous()
    ce = _ce_loss(shift_c_logits, shift_c_labels, vocab, label_smoothing=cfg.label_smooth)
    components["ce"] = ce
    if cfg.loss_fn == "ce": return ce, {**components, "total": ce}

    if cfg.loss_fn in ("ce_ipo", "ce_ipo_kl"):
        ipo = _bias_ipo(chosen_logits, rejected_logits, chosen_labels, rejected_labels, tau=cfg.ipo_tau, frozen_chosen_logits=frozen_chosen_logits, frozen_rejected_logits=frozen_rejected_logits)
        components["ipo"] = ipo
        kl = chosen_logits.new_zeros(())
        if cfg.loss_fn == "ce_ipo_kl" and frozen_chosen_logits is not None:
            kl = _kl_regularisation(chosen_logits, frozen_chosen_logits, chosen_labels, attention_mask=None, mode="label_only")
            components["kl"] = kl
        total = ce + cfg.bias_lambda * ipo + cfg.kl_lambda * kl
        components["total"] = total
        return total, components

    dpo, hinge, kl = chosen_logits.new_zeros(()), chosen_logits.new_zeros(()), chosen_logits.new_zeros(())
    if cfg.loss_fn in ("ce_dpo", "ce_dpo_kl", "ce_dpo_hinge_kl"):
        lp_chosen, lp_rejected = _sequence_logprob(chosen_logits, chosen_labels), _sequence_logprob(rejected_logits, rejected_labels)
        dpo = -F.logsigmoid(cfg.dpo_beta * (lp_chosen - lp_rejected)).mean()
        components["dpo"] = dpo
    if cfg.loss_fn == "ce_dpo_hinge_kl":
        hinge = F.relu(_sequence_logprob(rejected_logits, rejected_labels) - _sequence_logprob(chosen_logits, chosen_labels) + cfg.bias_margin).mean()
        components["hinge"] = hinge
    if cfg.loss_fn in ("ce_kl", "ce_dpo_kl", "ce_dpo_hinge_kl"):
        if frozen_chosen_logits is None: raise ValueError("frozen_logits required for KL loss variants.")
        kl = _kl_regularisation(chosen_logits, frozen_chosen_logits, chosen_labels, attention_mask=None, mode="label_only")
        components["kl"] = kl
    
    total = ce + cfg.bias_lambda * dpo + cfg.hinge_lambda * hinge + cfg.kl_lambda * kl
    components["total"] = total
    return total, components

def compute_loss(logits, labels, correct_token_ids, stereo_token_ids, cfg, frozen_logits=None, attention_mask=None, context_condition_ids=None, question_polarity_ids=None, paired_logits=None, has_pair=None):
    B, seq_len, vocab = logits.shape
    shift_logits, shift_labels = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
    if (shift_labels != -100).sum() == 0:
        zero = logits.new_zeros(()); return zero, {"ce": zero, "total": zero}
    
    smooth = cfg.label_smooth if cfg.loss_fn == "label_smoothing" else 0.0
    ce = _ce_loss(shift_logits, shift_labels, vocab, label_smoothing=smooth)
    if cfg.loss_fn in ("ce", "focal", "label_smoothing"): return ce, {"ce": ce, "total": ce}

    label_logits, has_label = _get_label_logits(shift_logits, shift_labels)
    ref_label_logits = None
    if frozen_logits is not None:
        ref_label_logits, _ = _get_label_logits(frozen_logits[:, :-1, :].contiguous(), shift_labels)
    
    components: Dict[str, torch.Tensor] = {"ce": ce}
    bias_term = _bias_dpo(label_logits, correct_token_ids, stereo_token_ids, has_label, vocab, beta=cfg.dpo_beta, ref_label_logits=ref_label_logits, use_ref=cfg.dpo_ref)
    components["dpo"] = bias_term
    
    kl = logits.new_zeros(())
    if cfg.loss_fn in {"ce_dpo_kl"}:
        if frozen_logits is None: raise ValueError("frozen_logits required for KL loss variants.")
        kl = _kl_regularisation(logits, frozen_logits, labels, attention_mask=attention_mask, mode=cfg.kl_mode)
        components["kl"] = kl
        
    total = ce + cfg.bias_lambda * bias_term + cfg.kl_lambda * kl
    components["total"] = total
    return total, components