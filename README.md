# MPDD: A Comprehensive Benchmark for Malicious Prompt Detection
## From Classical ML to Transformers

[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)

> **⚠️ This repository is currently under active development**

**Author:** Mohammed Amine Jebbar - Independent Researcher

## 🎯 What's This About?

I've been working on something that I think is pretty important for AI safety - detecting malicious prompts before they can cause harm. This project introduces **MPDD**, a balanced dataset of 40k prompts (well, 39,234 to be exact) where half are malicious and half are benign.

The main goal here is simple: figure out which machine learning approaches work best for catching bad prompts. I've tested everything from basic models to state-of-the-art transformers to see what actually performs in the real world.

## 🔬 Research Objective

The study aims to answer a fundamental question: **What's the most effective approach for detecting malicious prompts?**

I've divided this into three main categories of models to get a complete picture:
- **Classical ML** (the reliable workhorses)
- **Deep Learning** (the complex middle ground)  
- **Transformers** (the current SOTA)

## 🤖 Models I've Tested

### Classical ML Models
- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost
- Naive Bayes
- Dummy Classifier (baseline)

### Deep Learning Models
- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)

### Transformer Models
- BERT (bert-base-uncased)
- RoBERTa (roberta-base)
- DistilBERT (distilbert-base-uncased)
- DeBERTa (microsoft/deberta-base)

## 📊 Dataset: MPDD

The **Malicious Prompt Detection Dataset (MPDD)** is a carefully curated collection of 39,234 prompts, perfectly balanced between malicious and benign examples. I've spent considerable time cleaning and deduplicating content from multiple sources to create something that's actually useful for research.

**Key Features:**
- 🎯 **Balanced**: 50% malicious, 50% benign
- 🧹 **Clean**: Deduplicated and preprocessed
- 📈 **Large**: 40k samples for robust training
- 🔄 **Diverse**: Multiple source datasets combined

**Available on Kaggle:** [Malicious Prompt Detection Dataset](https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd)

### Source Datasets
I've aggregated and cleaned several existing datasets:
- [Prompt Injection in the Wild - PredictionGuard](https://www.kaggle.com/datasets/arielzilber/prompt-injection-in-the-wild)
- [Prompt Injection Malignant](https://www.kaggle.com/datasets/marycamilainfo/prompt-injection-malignant)
- [LLM 7 Prompt Training Dataset](https://www.kaggle.com/datasets/carlmcbrideellis/llm-7-prompt-training-dataset)
- [Quora Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset)

## 📁 Project Structure

```
.
├── datasets/                    # Raw datasets from various sources
│   ├── forbidden_question_set_df.csv
│   ├── jailbreak_prompts.csv
│   ├── malicous_deepset.csv
│   ├── malignant.csv
│   ├── predictionguard_df.csv
│   └── question_pairs_dataset.csv
├── experiments/                 # All the ML experiments
│   ├── baseline_models.ipynb   # Classical ML models training
│   ├── deep_models.py          # Deep learning experiments
│   ├── deep_models_comparison.png
│   ├── results/                # Transformer model outputs (empty for now)
│   │   ├── bert-base-uncased/
│   │   ├── distilbert-base-uncased/
│   │   ├── microsoft_deberta-base/
│   │   └── roberta-base/
│   ├── results_deep_models.csv
│   ├── results_transformers.csv
│   ├── transformer_models.ipynb
│   └── transformers_models_comparison.png
├── notebooks/
│   └── malicious_prompt_detection_dataset_creation.ipynb  # Dataset creation process
├── processed_data/             # The final MPDD dataset
│   ├── MPDD.csv
│   └── MPDD.pkl
├── requirements.txt
└── README.md
```
The datasets/ folder has been removed from this Git repository due to file size limitations imposed by GitHub. Some dataset files exceeded the maximum allowed size, preventing successful pushes to the remote repository.
## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Quick Start
1. **Dataset Creation**: Check out `notebooks/malicious_prompt_detection_dataset_creation.ipynb` to see how I built MPDD
2. **Classical ML**: Run `experiments/baseline_models.ipynb` for traditional approaches
3. **Deep Learning**: Execute `experiments/deep_models.py` for neural network experiments  
4. **Transformers**: Use `experiments/transformer_models.ipynb` for SOTA models

## 📈 Initial Results (Spoiler Alert)

Without giving too much away, I've found some pretty interesting patterns:
- Classical models perform surprisingly well
- There are some unexpected results in the deep learning category
- Transformers deliver as expected, but the cost-benefit trade-off is worth discussing

Full results and analysis coming soon in the research paper!

## 🔄 Current Status

- [x] Dataset collection and preprocessing
- [x] Classical ML experiments completed
- [x] Deep learning experiments completed  
- [x] Transformer experiments completed
- [ ] Comprehensive analysis and paper writing
- [ ] Code cleanup and documentation
- [ ] Results visualization improvements

## 📝 Citation (Coming Soon)

This work is being prepared for academic publication. Citation format will be updated once submitted.

## 🤝 Contributing

This is primarily a research project, but I'm always open to discussions and suggestions! Feel free to:
- Open issues for questions or suggestions
- Reach out if you're working on similar problems
- Share your own experiments with the dataset

## 📧 Contact

Mohammed Amine Jebbar - Independent Researcher
- **Kaggle**: [mohammedaminejebbar](https://www.kaggle.com/mohammedaminejebbar)
- **LinkedIn**: [Connect with me](https://www.linkedin.com/in/codebyamine/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*"Building safer AI, one prompt at a time."*