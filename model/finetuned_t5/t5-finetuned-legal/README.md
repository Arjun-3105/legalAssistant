# Model Card for `Nawal11/t5-finetuned-legal`

## Model Summary

This is a T5-base model fine-tuned on a legal dataset containing 500 cases. The model was trained to generate legal judgment summaries based on case petitions, aiding in automated legal reasoning and decision drafting. It is part of a broader Retrieval-Augmented Generation (RAG) system used for answering legal queries.

## Model Details

* **Developer:** Nawal Akhlaq
* **Model Type:** Sequence-to-sequence text generation
* **Language(s):** English (legal domain)
* **Base Model:** [`t5-base`](https://huggingface.co/t5-base) by Google AI
* **Fine-tuned for:** Legal case judgment generation
* **Training Framework:** Hugging Face Transformers
* **RAG Integration:** Yes, used in a local setup with retrieval

## Model Sources

* **Repository:** [Hugging Face Hub - Nawal11/t5-finetuned-legal](https://huggingface.co/Nawal11/t5-finetuned-legal)

## Intended Use

### Direct Use

* Input a legal petition (text describing the facts and legal request).
* Output is a plausible legal judgment or result generated in natural language.

### Downstream Use

* Integration in legal RAG pipelines
* Legal assistant chatbots
* Educational tools for law students

### Out-of-Scope Use

* Non-legal domains (e.g., finance, health)
* Making legally binding decisions without human supervision
* Use in jurisdictions outside the context of training data (e.g., outside Pakistani law if applicable)

## Bias, Risks, and Limitations

### Limitations

* May generalize incorrectly on unseen legal formats or rare case types
* Biased by patterns in small fine-tuning dataset (500 examples)

### Risks

* Outputs may appear authoritative but contain errors
* May hallucinate legal terms or rulings

### Recommendations

* Always validate model output with legal experts
* Use alongside retrieval-based evidence or citations

## How to Use

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("Nawal11/t5-finetuned-legal")
model = T5ForConditionalGeneration.from_pretrained("Nawal11/t5-finetuned-legal")

input_text = "legal_judgment: The petitioner was terminated without notice..."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

output_ids = model.generate(input_ids, max_length=512)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Training Details

### Training Data

* A CSV dataset of 500 labeled legal case examples (`petition`, `result` columns)
* Data split: 90% training / 10% testing

### Training Procedure

* Base model: `t5-base`
* Epochs: 35
* Max token length: 512
* Learning rate: 2e-5
* Batch size: 2 (with gradient accumulation for effective batch size 16)
* Truncation and padding applied

### Compute Environment

* **Platform:** Google Colab
* **Hardware:** 1x GPU (likely T4 or P100)
* **Estimated training time:** \~45–60 minutes
* **Precision:** fp32

## Evaluation

### Metrics

* Rouge (used in the trainer’s evaluation phase)
* Manual qualitative review of generated judgments

## Environmental Impact

* **Cloud Provider:** Google Colab
* **Carbon Emission Estimate:** Minimal (based on \~1 hour of training on a single GPU)
* [Estimate with MLCO2 calculator](https://mlco2.github.io/impact#compute)

## Citation

```bibtex
@misc{nawal2025t5legal,
  author = {Nawal Akhlaq},
  title = {T5 Fine-Tuned on Legal Case Dataset},
  year = {2025},
  howpublished = {\url{https://huggingface.co/Nawal11/t5-finetuned-legal}},
  note = {Fine-tuned on 500 legal cases for automated judgment generation}
}
```

## Contact

For questions or feedback, reach out via [Hugging Face profile](https://huggingface.co/Nawal11).

---