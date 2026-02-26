# Existential Presupposition in Large Language Models

This repository contains the code for the experiments of our paper "There is No Spoon: Existential Presupposition in Large Language
Models" (link to be added) which investigates how large language models (LLMs) handle existential presupposition across different syntactic environments and contextual configurations.

The project compares three modelling paradigms:

- Instruction-tuned language models (in zero-shot and few-shot prompt configuration)
- Pretrained (pt) base models fine-tuned using LoRA on three NLI datasets
- An NLI-fine-tuned variant of DeBERTa as a baseline

All evaluations are conducted on constructed premise–hypothesis pairs across three experimental conditions:

- control  
- projection  
- context  

---

## Experimental Paradigms

### 1. Instruction-Tuned Models 

Instruction-tuned models are evaluated using structured prompts.

Two prompting regimes are used:

- **Zero-shot prompting**
- **Few-shot prompting**

For the few-shot setting:

- Five independent runs were conducted.
- For each run, six NLI examples were sampled from the fine-tuning dataset.
- Each sampled set consisted of:
  - 2 entailment examples  
  - 2 neutral examples  
  - 2 contradiction examples  
- These examples were manually inserted into the few-shot JSON prompt template before the evaluation runs.

Evaluation is performed via label log-probability scoring.

---

### 2. Pretrained Models Fine-Tuned with LoRA

Pretrained base models (LLaMA-3.1-8B, Gemma-3-12B-pt) were fine-tuned on NLI data using LoRA.

Fine-tuning settings:

- LoRA rank: 16  
- Scaling factor (α): 32  
- Dropout: 0.05  
- Epochs: 2  
- Maximum sequence length: 256  
- Learning rate:
  - LLaMA-3.1-8B: 2e-4  
  - Gemma-3-12B-pt: 1.5e-4  

Fine-tuning was performed as a three-way NLI classification task (entailment / neutral / contradiction).  
Supervision was applied only at the label level.

Adapters were saved locally during training but **are not included in this repository** due to size constraints.

---

### 3. DeBERTa Baseline

We used the publicly available checkpoint:

MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

This model serves as an NLI baseline.

---

## Data

### Evaluation Data (Included)

The repository includes the via template constructed evaluation datasets:

- control.tsv  
- projection.tsv  
- context.tsv  

---

### Fine-Tuning Dataset (Not Included)

The fine-tuning dataset is a concatenation of three publicly available NLI datasets:

- WANLI  
  https://huggingface.co/datasets/alisawuffles/WANLI  

- ANLI  
  https://huggingface.co/datasets/facebook/anli 
 
- MNLI
  https://huggingface.co/datasets/nyu-mll/multi_nli


---

## Reproducibility Notes

LoRA adapter weights are not included in this repository due to size constraints.
To reproduce the fine-tuned models, access the base models from Hugging Face for the scripts in the finetune/ folder (meta-llama/Llama-3.1-8B, google/gemma-3-12b-pt)

Use the configuration files and fine-tuning script provided in the finetune/ folder. After fine-tuning, run the evaluation scripts in src/.

The fine-tuning dataset is not included. See the section Fine-Tuning Dataset above for details on the required NLI datasets and how to reconstruct the concatenated training data.

For the instruction-tuned models (zero and few-shot prmpt configurations), use these models available on Hugging Face:

- meta-llama/Llama-3.1-8B-Instruct

- google/gemma-3-12b-it

Note for few-shot prompting:

1. Sample from the concatenated fine-tuning NLI dataset

2. Select 2 entailment, 2 neutral, and 2 contradiction examples

3. Insert these examples into the few-shot prompt template before evaluation

4. Run the evaluation scripts corresponding to each experimental condition.

---

## Output

Outputs include the predicted NLI label and per-label scores. For:

- **Instruction-tuned models (prompt scoring):**
  - `pred_label`
  - `p_entailment`, `p_neutral`, `p_contradiction` (softmax over candidate completion scores)

- **Sequence-classification models (fine-tuned models):**
  - `pred_label`
  - raw logits for each label
  - softmax probabilities for each label


---

## Dependencies

The repository relies on PyTorch, Hugging Face Transformers, PEFT for LoRA and your standard data-processing libraries.  
See `requirements.txt` for more details.

---

## Citation

If you use this repository, please cite our associated paper (to be added).

## Contact

If you have questions about the experiments or implementation details, please feel free to reach out via email:

**marie-leontine.woergoetter@univie.ac.at**