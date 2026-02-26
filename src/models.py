import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",         
        torch_dtype=torch.bfloat16, 
    )

    model.eval()
    return model, tokenizer, device
