"""stats_analysis.py - descriptive statistics, normality tests and correlation analysis (report Sections 4.4, 4.5)."""
import json
import numpy as np, pandas as pd
from scipy import stats
from config import DATA, OUTPUTS


def main():
    df = pd.read_pickle(DATA / "clean_truth.pkl").sort_values(["facility_id", "drug_id", "day"])
    d28 = df.groupby(["facility_id", "drug_id"]).demand.transform(lambda s: s.rolling(28, min_periods=7).mean())
    dos = (df.on_hand / (d28 + 1e-6)).replace([np.inf, -np.inf], np.nan).dropna().clip(upper=365)
    test = pd.read_pickle(OUTPUTS / "test_scored.pkl")
    variables = {"Daily demand (units)": df.demand, "On-hand stock (units)": df.on_hand, "Days of supply": dos,
                 "Replenishment lead time (days)": df.lead_time_days.dropna(), "Next shipment ETA (days)": df.eta_days[df.eta_days < 30],
                 "Predicted stockout risk": test.risk}
    desc = {}
    for name, v in variables.items():
        v = v.astype(float); w, p = stats.shapiro(v.sample(5000, random_state=42))     # Shapiro-Wilk on a 5,000 sample
        desc[name] = dict(mean=v.mean(), median=v.median(), std=v.std(), min=v.min(), max=v.max(),
                          skewness=stats.skew(v), kurtosis=stats.kurtosis(v), shapiro_W=w, shapiro_p=p)
    feats = {"days_of_supply": "On-hand days of supply", "pos_days_of_supply": "Inventory-position days of supply", "rop_gap": "Reorder-point gap",
             "eta_days": "Next shipment ETA (days)", "intransit_14d": "In-transit quantity (14 days)", "lt_mean": "Mean lead time (90 days)",
             "short_days30": "Shortage days (30 days)", "stockout_days30": "Past stockout days (30 days)", "d_std28": "Demand std. dev. (28 days)"}
    corr = {}
    for col, name in feats.items():
        v = test[[col, "future_stockout14"]].replace([np.inf, -np.inf], np.nan).dropna()
        r, p = stats.pearsonr(v[col], v.future_stockout14); corr[name] = dict(pearson_r=r, p_value=p)
    json.dump({"descriptive": desc, "correlation_with_stockout_target": corr}, open(OUTPUTS / "statistics.json", "w"), indent=1, default=float)
    print("[statistics] descriptive statistics:")
    for k, v in desc.items(): print(f"   {k:<32} mean {v['mean']:9.2f}  median {v['median']:8.2f}  skew {v['skewness']:5.2f}  Shapiro p {v['shapiro_p']:.1e}")
    print("[statistics] Pearson correlation with the 14-day stockout target:")
    for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]["pearson_r"])): print(f"   {k:<36} r = {v['pearson_r']:+.3f}")


if __name__ == "__main__":
    main()
