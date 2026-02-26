import pandas as pd
from tqdm.auto import tqdm
from src.scoring import score_hypothesis, load_prompt

def batch_predict(input_tsv, model, tokenizer, device, output_csv, prompt_path, max_rows=None):
    df = pd.read_csv(input_tsv, sep="\t")
    if max_rows:
        df = df.head(max_rows)

    prompt_template = load_prompt(prompt_path)

    rec = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred, scores = score_hypothesis(
            model, tokenizer, row["premise"], row["hypothesis"], device, prompt_template
        )
        rec.append({
            **row.to_dict(),
            "pred_label": pred,
            **{f"p_{k}": round(v, 4) for k, v in scores.items()}
        })

    out = pd.DataFrame(rec)
    out.to_csv(output_csv, index=False)
    return out
