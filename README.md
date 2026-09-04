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

## Results (test set, macro-F1)

| Model | Train data | Macro-F1 |
|---|---|---|
| QADI paper (Abdelali et al., 2020) | full | 0.606 |
| TF-IDF char n-grams (2–5) + LinearSVC | full (440k) | **0.582** |
| MARBERTv2 fine-tuned, 1 epoch | 100k subset | 0.476 |
| MARBERTv2 fine-tuned, 2 epochs | 100k subset | 0.494 |
| MARBERTv2 fine-tuned, 2 epochs | full (440k) | <!-- pending --> |

### Per-dialect observations (baseline)

- Easiest: EG (0.81), LY (0.74), LB (0.71), MA (0.69) — distinctive dialects and/or large classes.
- Hardest: Gulf countries (SA/KW/QA/BH/AE/OM, 0.46–0.62), JO/SY (0.44), YE (0.41).
- Sudanese (SD): F1 0.67, precision 0.76 — most often confused with EG.

### Confusion structure

Errors cluster by region, not randomly. Within each region the model defaults to the largest class:
- Gulf → KW
- Levant → PL
- Everything else → EG (largest class overall)

This suggests the task is effectively ~4 regional groups with weak within-group signal at the tweet level, plus class imbalance.

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
pip install datasets scikit-learn pandas
python src/baseline.py          # full run (~15 min on CPU)
python src/baseline.py 50000    # quick run on a 50k sample
```

## Limitations

- Labels are user-level and automatic; a Sudanese user living in Egypt is labeled SD regardless of what they write.
- Test set is small per class (TN: 181 samples), so per-dialect scores have wide error bars.
- Dataset license is not specified by the uploader; this repo uses it for research/educational purposes only and does not redistribute it.

## Citation

Abdelali, A., Mubarak, H., Samih, Y., Hassan, S., & Darwish, K. (2020). *Arabic Dialect Identification in the Wild*.