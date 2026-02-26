import argparse
from src.predict_ptft import batch_predict_seqcls

DEFAULT_DEBERTA = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=DEFAULT_DEBERTA)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    out = args.output_csv or f"output/{args.model_id.replace('/','_')}_projection_deberta.csv"

    batch_predict_seqcls(
        input_tsv="data/projection.tsv",
        output_csv=out,
        base_model=args.model_id,
        adapter_dir=None,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )

if __name__ == "__main__":
    main()
