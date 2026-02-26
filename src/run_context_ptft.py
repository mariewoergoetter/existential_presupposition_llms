import argparse
from src.predict_ptft import batch_predict_seqcls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True, help="Directory created by finetune --out_dir")
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    out = args.output_csv or f"output/{args.base_model.replace('/','_')}_context_ptft.csv"

    batch_predict_seqcls(
        input_tsv="data/context.tsv",
        output_csv=out,
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )

if __name__ == "__main__":
    main()
