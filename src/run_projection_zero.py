import sys
from src.models import load_model
from src.predict import batch_predict

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.run_projection_zero <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    model, tokenizer, device = load_model(model_name)

    batch_predict(
        "data/projection.tsv",
        model,
        tokenizer,
        device,
        f"output/{model_name.replace('/','_')}_projection_zero.csv",
        prompt_path="prompts/zero_shot.json"
    )
