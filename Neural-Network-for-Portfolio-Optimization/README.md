# End-to-End Neural Network for Portfolio Optimization

Comparing MLP and LSTM neural networks against a Mean-Variance Optimization (MVO) benchmark for direct portfolio weight allocation across 10 diversified ETFs (2015–2025).

---

## Research Questions

| # | Question |
|---|---|
| 1 | Does ML improve Sharpe ratio vs MVO benchmark? |
| 2 | Which architecture performs better: MLP or LSTM? |
| 3 | Are allocations stable and practical (low turnover)? |
| 4 | How robust is performance under drawdowns? |

---

## Results (Test Period: May 2024 – Dec 2025)

| Metric | MVO | MLP | LSTM |
|---|---|---|---|
| Cumulative Return | 31.86% | 65.25% | 46.07% |
| Annual Return | 17.72% | 33.19% | 24.53% |
| Annual Volatility | 10.52% | 20.10% | 14.16% |
| Sharpe Ratio | 1.6842 | 1.6514 | **1.7324** |
| Max Drawdown | -9.17% | -16.12% | -11.52% |

---

## Dataset

- **Source:** Yahoo Finance via `yfinance`
- **Period:** January 2015 – December 2025
- **Assets:** SPY, QQQ, IWM, EFA, EEM, TLT, LQD, GLD, VNQ, XLE
- **Split:** 70% train / 15% validation / 15% test

---

## How to Run on Google Colab

**Step 1 — Upload the notebook**

Go to [colab.research.google.com](https://colab.research.google.com) → `File` → `Upload notebook` → select `Neural_Network_for_Portfolio_Optimization.ipynb`

**Step 2 — Set runtime to GPU**

`Runtime` → `Change runtime type` → Hardware accelerator → select `T4 GPU` → Save

**Step 3 — Run all cells**

`Runtime` → `Run all`  

Total runtime is approximately 20–30 minutes depending on GPU availability.

---

## Repository Structure

```
├── Neural_Network_for_Portfolio_Optimization.ipynb   # main notebook
└── README.md
```

---

## Requirements

All dependencies are standard and pre-installed in Google Colab:

```
yfinance, pandas, numpy, scipy, scikit-learn, torch, matplotlib
```

---

## Key Design Choices

- **Loss function:** Negative Sharpe ratio + turnover penalty (λ=0.5) + drawdown penalty (λ=0.5)
- **MLP input:** 110 features × 21-day rolling window (flattened)
- **LSTM input:** 60 time-sensitive features (returns, lags, momentum) × 21-day sequence
- **LSTM architecture:** LayerNorm + orthogonal weight init + intermediate FC layer
- **Gradient clipping:** Applied to LSTM only (`max_norm=1.0`)

---

## Authors

Muhammad Zaid Maqsood — MS Data Science, Fordham University  
Course: CISC 5800
