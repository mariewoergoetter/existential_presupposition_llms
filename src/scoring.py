import torch
import torch.nn.functional as F
import json

CANDIDATES = ["entailment", "neutral", "contradiction"]

def load_prompt(path):
    with open(path) as f:
        return json.load(f)

def format_prompt(prompt_template, premise, hypothesis):
    text = json.dumps(prompt_template)
    text = text.replace("<PREMISE>", premise)
    text = text.replace("<HYPOTHESIS>", hypothesis)
    text = text.replace("<PREMISE_TO_TEST>", premise)
    text = text.replace("<HYPOTHESIS_TO_TEST>", hypothesis)
    return json.loads(text)

def score_hypothesis(model, tokenizer, premise, hypothesis, device, prompt_template):
    chat = format_prompt(prompt_template, premise, hypothesis)

    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    cand_logps = []
    for cand in CANDIDATES:
        full_text = prompt_text + cand
        inputs = tokenizer(full_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        logits = outputs.logits[0, -len(cand_ids)-1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = [log_probs[i, cand_ids[i]].item() for i in range(len(cand_ids))]
        cand_logps.append(sum(token_log_probs) / len(cand_ids))

    probs = torch.softmax(torch.tensor(cand_logps), dim=0).tolist()
    pred = CANDIDATES[int(torch.argmax(torch.tensor(probs)))]

    return pred, dict(zip(CANDIDATES, probs))
