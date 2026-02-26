from __future__ import annotations

import os
import argparse
import pandas as pd
from tqdm.auto import tqdm

import torch

from src.scoring_ptft import (
    load_seqcls_model,
    encode_pairs,
    forward_seqcls,
    LABEL_ORDER,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)

    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, default=None)

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_rows", type=int, default=None)

    return ap.parse_args()


def batch_predict_seqcls(
    input_tsv: str,
    output_csv: str,
    base_model: str,
    adapter_path: str | None,
    max_length: int = 256,
    batch_size: int = 16,
    max_rows: int | None = None,
):
    bundle = load_seqcls_model(base_model, adapter_path=adapter_path)

    df = pd.read_csv(input_tsv, sep="\t")
    if max_rows is not None:
        df = df.head(max_rows)

    if "premise" not in df.columns or "hypothesis" not in df.columns:
        raise ValueError("Input TSV must contain columns: premise, hypothesis")

    records = []
    n = len(df)

    for start in tqdm(range(0, n, batch_size), desc="Predict (seqcls)"):
        end = min(start + batch_size, n)
        chunk = df.iloc[start:end]

        premises = chunk["premise"].astype(str).tolist()
        hypotheses = chunk["hypothesis"].astype(str).tolist()

        batch = encode_pairs(bundle.tokenizer, premises, hypotheses, max_length=max_length)

        batch = {k: v.to(bundle.device) for k, v in batch.items()}

        out = forward_seqcls(bundle.model, batch)
        logits = out["logits"].numpy()
        probs = out["probs"].numpy()
        preds = out["pred_labels"]

        for i, (_, row) in enumerate(chunk.iterrows()):
            rec = row.to_dict()
            rec["pred_label"] = preds[i]

            for j, lab in enumerate(LABEL_ORDER):
                rec[f"logit_{lab}"] = float(logits[i, j])
                rec[f"p_{lab}"] = float(probs[i, j])

            records.append(rec)

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def main():
    args = parse_args()
    batch_predict_seqcls(
        input_tsv=args.input_tsv,
        output_csv=args.output_csv,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
