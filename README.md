# MPDD — Malicious Prompt Detection Dataset

Balanced benchmark dataset and comparison of 13 models, from logistic regression to DeBERTa.

![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)

![License](https://img.shields.io/badge/License-MIT-green.svg)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)

**Author:** Mohammed Amine Jebbar

---

## What this is

A dataset of 39,234 prompts, split evenly between malicious (jailbreaks, prompt injections, harmful requests) and benign, aggregated from four public sources, deduplicated and preprocessed. Alongside it, the benchmark code I used to compare 13 models on the same task.

I built it to answer a question I couldn't find a clean answer to: **which approach actually catches malicious prompts, and at what cost?**

---

## The dataset

|               |                                                      |
| ------------- | ---------------------------------------------------- |
| Samples       | 39,234                                               |
| Balance       | 50% malicious / 50% benign                           |
| Sources       | 4 public datasets (below)                            |
| Preprocessing | Deduplicated, cleaned, standardised                  |
| Format        | CSV + pickle (`processed_data/MPDD.csv`, `MPDD.pkl`) |

### Source datasets

- [Prompt Injection in the Wild — PredictionGuard](https://www.kaggle.com/datasets/arielzilber/prompt-injection-in-the-wild)
- [Prompt Injection Malignant](https://www.kaggle.com/datasets/marycamilainfo/prompt-injection-malignant)
- [LLM 7 Prompt Training Dataset](https://www.kaggle.com/datasets/carlmcbrideellis/llm-7-prompt-training-dataset)
- [Quora Question Pairs](https://www.kaggle.com/datasets/quora/question-pairs-dataset)

The creation process is documented in [`notebooks/malicious_prompt_detection_dataset_creation.ipynb`](notebooks/malicious_prompt_detection_dataset_creation.ipynb).

---

## Models benchmarked

| Family        | Models                                                                                 | Count |
| ------------- | -------------------------------------------------------------------------------------- | ----- |
| Classical ML  | Logistic Regression, Linear SVM, Random Forest, XGBoost, Naive Bayes, Dummy (baseline) | 6     |
| Deep learning | MLP, LSTM, CNN                                                                         | 3     |
| Transformers  | BERT, RoBERTa, DistilBERT, DeBERTa                                                     | 4     |

---

## Results

The full numbers live in `experiments/results_deep_models.csv` and `experiments/results_transformers.csv`. Comparison plots are in `experiments/`.

**Headline finding:** classical models perform better than their reputation suggests, and the transformer gain over them is smaller than the cost difference.

> Full table to be added here from the CSVs above.

---

## Repository layout

```
.
├── datasets/              # Raw sources (removed from git — file size limits)
├── experiments/
│   ├── baseline_models.ipynb        # Classical ML
│   ├── deep_models.py               # Deep learning
│   ├── transformer_models.ipynb     # Transformers
│   ├── results_deep_models.csv
│   ├── results_transformers.csv
│   └── *.png                        # Comparison plots
├── notebooks/
│   └── malicious_prompt_detection_dataset_creation.ipynb
├── processed_data/        # MPDD.csv, MPDD.pkl
├── requirements.txt
└── README.md
```

Note: `datasets/` is not in the repository — several source files exceed GitHub's size limit. They are all linked above.

---

## Reproduce

```bash
pip install -r requirements.txt
```

1. Dataset creation — `notebooks/malicious_prompt_detection_dataset_creation.ipynb`
2. Classical ML — `experiments/baseline_models.ipynb`
3. Deep learning — `experiments/deep_models.py`
4. Transformers — `experiments/transformer_models.ipynb`

---

## Used in research

If you use MPDD, I'd like to know — it's the only way I find out.

- **BAGEL** (Hassan et al., 2026) uses MPDD as one of nine core datasets. The authors report outperforming OpenAI's Moderation API and ShieldGemma (F1 = 0.92).
- **Information / MDPI** (Kushnerov et al., 2026) uses MPDD as its main dataset in a benchmarking study of eight neural architectures against adversarial attacks.
- **Procedia Computer Science** (Vyas et al., 2026) uses it in work on sanitising malicious prompts rather than only flagging them.
- **GuardNet** (Neves et al., 2026) references results obtained on MPDD when benchmarking lightweight guardrail models.

---

## Status

Done:

- [x] Dataset collection, cleaning, deduplication
- [x] Classical ML benchmarks
- [x] Deep learning benchmarks
- [x] Transformer benchmarks

Not done:

- [ ] Results table in this README
- [ ] Write-up of the analysis
- [ ] Code cleanup

---

## Contact

- **Kaggle**: [mohammedaminejebbar](https://www.kaggle.com/mohammedaminejebbar)
- **LinkedIn**: [Amine Jebbar](https://www.linkedin.com/in/codebyamine/)

---

## License

MIT — see [LICENSE](LICENSE).
