# Real-Time API-Driven Tracking of Pharmaceutical Supply Chains and Stockout Forecasting

Data Science Project – **Sanjana B (722824104212)**, B.E. Computer Science and Engineering,
Sri Eshwar College of Engineering, Coimbatore (Batch 2024–2028).
Guide: Dr. K. Suresh Kumar, M.Tech., Ph.D., Associate Professor, CSE.

A tracking system that collects real-time data from pharmaceutical supply-chain APIs
(manufacturer, drug-shortage, distributor order, shipment-tracking and pharmacy inventory feeds),
forecasts medicine demand 14 days ahead, predicts which medicines will **stock out in the next
14 days** at each facility, and recommends an action (reorder, expedite, alternate manufacturer
or inter-facility transfer) on a live dashboard.

## How it works

```
 Supply-chain APIs ──► ingest.py ──► preprocess.py ──► features.py ──► models.py ──► predict.py ──► api.py / dashboard.py
 (5 feeds, polled      (async,       (duplicates,      (days of supply,  (LightGBM 14-day      (risk + recommended
  concurrently)         validate,     imputation)       reorder point,    demand forecast and    action per
                        de-duplicate)                   shipment ETA)     stockout classifier)   facility-medicine)
```

The data covers **30 healthcare facilities in 15 cities of Tamil Nadu × 50 essential medicines**
(1,500 series, 1 Jan 2024 – 30 Jun 2025, 822,961 raw records). Because live access to hospital and
distributor systems requires institutional agreements, the API feeds are produced by a realistic
simulator (`gen_data.py`) with seasonality, outbreaks, upstream shortages, distributor disruptions,
a monsoon flood period, missing values and duplicate pushes.

## Project structure

```
pharma-stockout-forecasting/
├── run_pipeline.py        # runs every step end to end
├── requirements.txt
├── src/
│   ├── config.py          # paths and modelling assumptions
│   ├── gen_data.py        # supply-chain data simulator (API feeds)
│   ├── ingest.py          # async API polling, validation, de-duplication, Kafka/in-memory topics
│   ├── preprocess.py      # duplicate removal and missing-value imputation
│   ├── features.py        # days of supply, reorder point, lead-time, ETA and shortage features
│   ├── models.py          # demand forecasting + stockout classification and baselines
│   ├── simulate.py        # existing vs proposed replenishment policy simulation
│   ├── stats_analysis.py  # descriptive statistics, Shapiro-Wilk, Pearson correlation
│   ├── predict.py         # stockout prediction module (risk + recommended action)
│   ├── api.py             # FastAPI service
│   └── dashboard.py       # Plotly Dash monitoring dashboard
├── data/                  # facilities.csv, drugs.csv (+ generated panels, not committed)
├── outputs/               # metrics.json, simulation.json, statistics.json (+ trained models)
└── figures/               # charts produced by the pipeline
```

## Steps to run

```bash
git clone https://github.com/<your-username>/pharma-stockout-forecasting.git
cd pharma-stockout-forecasting
pip install -r requirements.txt

python run_pipeline.py                      # full pipeline (about 10 minutes on a laptop)

cd src
python predict.py --top 10                  # ten highest-risk medicines right now
python predict.py F003 D045                 # one facility / medicine
uvicorn api:app --reload                    # REST API  -> http://127.0.0.1:8000/docs
python dashboard.py                         # dashboard -> http://127.0.0.1:8050
```

## Results (held-out test period, Jan–Jun 2025)

**Stockout prediction (14-day horizon)**

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC |
|---|---|---|---|---|---|
| Rule-based ROP alert (existing) | 74.17 | 16.23 | 35.85 | 22.34 | – |
| Logistic Regression | 91.10 | 56.86 | 58.72 | 57.78 | 0.889 |
| Random Forest | 92.03 | 63.34 | 54.96 | 58.85 | 0.902 |
| XGBoost | 91.53 | 59.44 | 57.61 | 58.51 | 0.900 |
| **Proposed (LightGBM + forecast & tracking features)** | 94.13 | 72.89 | 69.08 | 70.93 | 0.943 |

**14-day demand forecast (all 1,500 series)**

| Model | MAE (units) | WMAPE |
|---|---|---|
| Naive (last 14 days) | 146.44 | 12.51% |
| Moving Average (28-day) | 138.21 | 11.81% |
| Random Forest | 133.69 | 11.42% |
| LightGBM (proposed) | 129.80 | 11.09% |

**Replenishment simulation (existing static policy vs proposed predictive policy)**

| KPI | Existing | Proposed |
|---|---|---|
| Stockout days | 9,063 (3.34%) | 3,302 (1.22%) |
| Fill rate | 96.82% | 98.95% |
| Median inventory (days of demand) | 13.0 | 12.4 |
| Alternate-source orders | – | 706 |

The proposed policy cuts stockout days by **63.6%** while holding slightly less stock.

> **Note on reproducibility.** Training is deterministic, so these numbers are identical on every run.
> The dataset, the baselines, the forecasting results and the simulation reproduce the project report exactly.
> The proposed classifier differs from the report by less than 0.3 points (report: accuracy 94.33%,
> F1 71.21%, ROC-AUC 0.944) because the report was produced with an earlier LightGBM version.

## Tools

Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, XGBoost, Matplotlib, Seaborn, Plotly Dash, FastAPI.

## License

MIT – see [LICENSE](LICENSE).
