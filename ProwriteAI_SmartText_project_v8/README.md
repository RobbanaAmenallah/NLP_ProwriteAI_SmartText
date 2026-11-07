# ProwriteAI — Smart Text Improvement (4 Seq2Seq Models from Scratch)

> **Notice:** The main project documentation, full benchmarks, architecture diagrams, and real-world dataset reports are located in the [root README.md](../README.md).

---

## 📌 Overview

This directory contains the production codebase, trained checkpoints, evaluation datasets, and interactive Streamlit web application for **ProwriteAI**: an end-to-end Sequence-to-Sequence NLP system trained completely from scratch in PyTorch to transform noisy, casual, or grammatically flawed English text into professional correspondence.

## 🚀 Quick Launch

### 1. Run the Interactive Streamlit Demo
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501).

The application automatically loads `D_Pointer` (the best model based on validation loss) and provides:
- Live text rewriting with real-time confidence indicators (🟢 / 🟡 / 🔴)
- Token-level and out-of-vocabulary (OOV) copy inspection
- Academic multi-model comparison view (evaluating across `A_LSTM`, `B_BiLSTM`, `C_Attention`, and `D_Pointer` simultaneously)
- Toggle between greedy decoding and beam search ($k=3, 5$)

### 2. Run / Inspect the Training Notebook
```bash
jupyter notebook ProwriteAI_SmartText_4Models_Training.ipynb
```
The notebook is **already fully executed** with all plots, tables, and metrics saved in `prowrite_outputs/`.

---

## 📂 Directory Contents

| Path | Description |
|---|---|
| `app.py` | Full-featured Streamlit web application. |
| `calibrate_confidence.py` | Calibration tool computing log-probability thresholds on validation data. |
| `ProwriteAI_SmartText_4Models_Training.ipynb` | End-to-end executed Jupyter notebook covering preprocessing, vocabulary, models, training, and evaluation. |
| `data/` | **Benchmark Dataset v3** (36,000 pairs: 30k train / 3k val / 3k test) with zero test frame leakage. |
| `data_v1/`, `data_v2/` | Intermediate datasets (12,000 and 30,000 pairs) used during development iterations. |
| `dataset/` | Raw corpora suite: **CoEdIT** (70,000+ real human editing pairs), **Enron corporate email clean samples**, and generator scripts. |
| `prowrite_outputs/checkpoints/` | Production weights for `A_LSTM.pt`, `B_BiLSTM.pt`, `C_Attention.pt`, and `D_Pointer.pt` (~7–14 MB each). |
| `prowrite_outputs/artifacts/` | Serialized `vocab.json` and `tokenizer.json`. |
| `prowrite_outputs/plots/` | Generated training loss curves, validation comparisons, and vocabulary distributions. |
| `prowrite_outputs/*.csv` | Complete quantitative logs, qualitative outputs, and metric summaries. |

---

## 📊 Summary of Results

| Model | Architecture | Params | Val Loss | Test Loss | BLEU | ROUGE-L | chrF | Fact Recall | OOV Copy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Copy Baseline* | Identity | — | — | — | 42.11 | 65.55 | 70.91 | 41.74 % | 100.0 % |
| **A_LSTM** | LSTM Enc-Dec | 1.96 M | 3.854 | 3.687 | 6.93 | 31.42 | 22.49 | 1.86 % | 0.0 % |
| **B_BiLSTM** | BiLSTM Enc + LSTM Dec | 2.61 M | 3.724 | 3.615 | 8.29 | 33.99 | 23.91 | 2.70 % | 0.0 % |
| **C_Attention** | BiLSTM + Bahdanau Attention | 3.57 M | 2.016 | 1.897 | 47.90 | 73.01 | 59.85 | 80.72 % | 0.0 % |
| **D_Pointer** | Pointer-Generator + Coverage | 3.57 M | **1.245** | **1.300** | **66.35** | **83.26** | **78.90** | **88.71 %** | **85.29 %** |

For complete theoretical derivations, ablation studies, and qualitative analyses, see the [main README.md](../README.md).
