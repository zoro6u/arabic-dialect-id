# Arabic Dialect Identification

Country-level Arabic dialect classification from short text (tweets), covering 18 Arab countries. Built as a portfolio project comparing a classical baseline against a fine-tuned Arabic transformer, with the goal of serving the best model as an API.

## Dataset

[QADI](https://huggingface.co/datasets/Abdelrahman-Rezk/Arabic_Dialect_Identification) (Abdelali et al., 2020): ~540k tweets automatically labeled by author country. Labels are noisy by construction (the paper reports ~91.5% label accuracy). Splits: 440,052 train / 9,164 validation / 8,981 test.

Countries: OM, SD, SA, KW, QA, LB, JO, SY, IQ, MA, EG, PL, YE, BH, DZ, AE, TN, LY.

## Preprocessing

- Remove user mentions and URLs
- Strip `#` from hashtags (keep the word)
- Remove Arabic diacritics (tashkeel)
- Collapse whitespace

## Results (test set, 8,981 tweets)

| Model | Train data | Epochs | Macro-F1 |
|---|---|---|---|
| QADI paper (Abdelali et al., 2020) | full | – | 0.606 |
| TF-IDF char n-grams (2–5) + LinearSVC | 50k | – | 0.465 |
| TF-IDF char n-grams (2–5) + LinearSVC | full (440k) | – | 0.582 |
| MARBERTv2 fine-tuned | 100k | 1 | 0.476 |
| MARBERTv2 fine-tuned | 100k | 2 | 0.494 |
| **MARBERTv2 fine-tuned** | **full (440k)** | **2** | **0.614** |

The fine-tuned MARBERTv2 improves on the classical baseline by 3.2 points and matches the original paper (the 0.8-point difference is within noise on a test set this size).

Model: [zoro6u/marbert-arabic-dialect-id](https://huggingface.co/zoro6u/marbert-arabic-dialect-id)
Training: lr 2e-5, batch 128 (2×T4), max_len 64, fp16, warmup 400 steps, weight decay 0.01. ~1h40 on Kaggle.

### Per-dialect F1: baseline vs MARBERT

| | OM | SD | SA | KW | QA | LB | JO | SY | IQ | MA | EG | PL | YE | BH | DZ | AE | TN | LY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF+SVC | .46 | .67 | .54 | .62 | .55 | .71 | .44 | .44 | .65 | .69 | .81 | .59 | .41 | .44 | .65 | .49 | .59 | .74 |
| MARBERTv2 | .52 | .72 | .54 | .65 | .55 | .73 | .49 | .51 | .68 | .71 | .85 | .63 | .41 | .46 | .66 | .51 | .66 | .78 |

### Observations

- **Data size matters more than model choice at small scale.** TF-IDF on 50k (0.465) ≈ MARBERT on 100k for 1 epoch (0.476). MARBERT only overtakes the baseline when trained on the full 440k.
- **Small/confusable classes collapse without enough data.** With 100k samples MARBERT scored JO 0.10, SY 0.14, TN 0.08; with the full set they recovered to 0.49 / 0.51 / 0.66.
- **Easiest dialects:** EG, LY, LB, MA, SD — distinctive vocabulary and/or large classes.
- **Hardest:** the Gulf cluster (SA, QA, BH, AE, OM: 0.46–0.55) and YE (0.41, smallest class, unchanged by the transformer).
- **Errors cluster by region.** In the baseline confusion matrix, Gulf dialects default to KW, Levantine to PL, and everything else to EG — each the largest class in its region. The task behaves like ~4 regional groups with weak within-group signal at tweet level.

## Repo structure

```
src/load_data.py   # dataset loading + label names
src/baseline.py    # TF-IDF + LinearSVC baseline, confusion analysis
notebooks/         # MARBERT fine-tuning (Kaggle)
api/               # FastAPI inference service (planned)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install datasets scikit-learn pandas joblib
python src/baseline.py          # full run (~15 min on CPU)
python src/baseline.py 50000    # quick run on a 50k sample
```

## Limitations

- Labels are user-level and automatic; a Sudanese user living in Egypt is labeled SD regardless of what they write.
- Test set is small per class (TN: 181 samples), so per-dialect scores have wide error bars.
- Dataset license is not specified by the uploader; this repo uses it for research/educational purposes only and does not redistribute it.

## Citation

Abdelali, A., Mubarak, H., Samih, Y., Hassan, S., & Darwish, K. (2020). *Arabic Dialect Identification in the Wild*.