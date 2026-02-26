import os
import math
import csv
import json
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
VALID_LABELS = set(LABEL2ID.keys())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_json", type=str, default=None)
    return ap.parse_args()


def is_missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def normalize_label(x):
    if is_missing(x):
        return None
    s = str(x).strip().lower()
    return s if s in VALID_LABELS else None


def normalize_text(x):
    if is_missing(x):
        return None
    s = str(x).replace("\xa0", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s if s else None


def strict_csv_row_length_fix(in_csv: str) -> str:
    fixed_csv = in_csv.replace(".csv", ".fixed_strict.csv")

    with open(in_csv, "r", encoding="utf-8", errors="replace", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        ncols = len(header)

    kept = dropped = 0
    with open(in_csv, "r", encoding="utf-8", errors="replace", newline="") as fin, \
         open(fixed_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if len(row) == ncols:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    print(f"[csv] wrote={fixed_csv} kept={kept} dropped={dropped}")
    return fixed_csv


def load_and_clean_dataset(valid_csv: str):
    valid_fixed = strict_csv_row_length_fix(valid_csv)
    ds = load_dataset("csv", data_files={"validation": valid_fixed})

    def clean_split(dset):
        dset = dset.map(lambda ex: {
            "premise": normalize_text(ex.get("premise")),
            "hypothesis": normalize_text(ex.get("hypothesis")),
            "label": normalize_label(ex.get("label")),
        })
        dset = dset.filter(lambda ex: ex["premise"] is not None and ex["hypothesis"] is not None and ex["label"] is not None)
        return dset

    ds["validation"] = clean_split(ds["validation"])
    print("[data] valid=", len(ds["validation"]))
    return ds["validation"]


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    valid_ds = load_and_clean_dataset(args.valid_csv)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        texts = [f"Premise: {p}\nHypothesis: {h}" for p, h in zip(examples["premise"], examples["hypothesis"])]
        out = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
        out["labels"] = [LABEL2ID[l] for l in examples["label"]]
        return out

    tokenized = valid_ds.map(preprocess, batched=True, remove_columns=valid_ds.column_names)

    collator = lambda batch: tokenizer.pad(batch, padding=True, return_tensors="pt")

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False

    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    # evaluation loop
    n = len(tokenized)
    correct = 0
    all_preds = []
    all_labels = []

    prob_sum = np.zeros(3, dtype=np.float64)

    for start in range(0, n, args.batch_size):
        batch = tokenized.select(range(start, min(start + args.batch_size, n)))
        batch = collator([batch[i] for i in range(len(batch))])
        labels = batch.pop("labels").numpy()

        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            out = model(**batch)
            logits = out.logits.detach().float().cpu().numpy()

        preds = np.argmax(logits, axis=-1)
        correct += int((preds == labels).sum())

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

        probs = softmax_np(logits)
        prob_sum += probs.sum(axis=0)

    acc = correct / n if n > 0 else float("nan")
    mean_probs = (prob_sum / n).tolist() if n > 0 else [float("nan")] * 3

    cm = np.zeros((3, 3), dtype=int)
    for y, yhat in zip(all_labels, all_preds):
        cm[y, yhat] += 1

    metrics = {
        "accuracy": acc,
        "n_examples": n,
        "label_order": ["entailment", "neutral", "contradiction"],
        "mean_probs": mean_probs,
        "confusion_matrix": cm.tolist(),
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }

    print(json.dumps(metrics, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("[save] metrics ->", args.out_json)


if __name__ == "__main__":
    main()
