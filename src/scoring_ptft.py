from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


LABEL_ORDER = ["entailment", "neutral", "contradiction"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class SeqClsBundle:
    model: Any
    tokenizer: Any
    device: torch.device


def load_seqcls_model(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> SeqClsBundle:
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    base.config.use_cache = False
    if getattr(base.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        base.config.pad_token_id = tokenizer.pad_token_id

    model = base
    if adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError("peft is not installed")
        model = PeftModel.from_pretrained(base, adapter_path)

    model.eval()

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return SeqClsBundle(model=model, tokenizer=tokenizer, device=device)


def encode_pairs(
    tokenizer,
    premises: List[str],
    hypotheses: List[str],
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:

    texts = [f"Premise: {p}\nHypothesis: {h}" for p, h in zip(premises, hypotheses)]
    batch = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return batch


@torch.no_grad()
def forward_seqcls(
    model,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    
    out = model(**batch)
    logits = out.logits 
    probs = F.softmax(logits.float(), dim=-1)

    pred_ids = torch.argmax(probs, dim=-1).tolist()
    pred_labels = [ID2LABEL[i] for i in pred_ids]

    return {
        "logits": logits.detach().float().cpu(), 
        "probs": probs.detach().float().cpu(),
        "pred_labels": pred_labels,
    }
