"""features.py - rolling supply-chain features and weekly prediction snapshots (report Sections 3.3 and 4.2)."""
import numpy as np
import pandas as pd
from config import DATA, HORIZON, N_DAYS, TRAIN_END_DAY

H = HORIZON


def series_features(gr: pd.DataFrame) -> pd.DataFrame:
    d = gr["demand"]
    out = pd.DataFrame(index=gr.index)
    out["d_mean7"] = d.rolling(7).mean(); out["d_mean14"] = d.rolling(14).mean(); out["d_mean28"] = d.rolling(28).mean()
    out["d_std28"] = d.rolling(28).std(); out["d_trend"] = out["d_mean7"] / (out["d_mean28"] + 1e-6)
    out["d_lag7"] = d.shift(7); out["d_sum_last14"] = d.rolling(14).sum()
    out["future_demand14"] = d[::-1].rolling(H).sum()[::-1].shift(-1)                 # forecast target
    out["future_stockout14"] = gr["stockout"][::-1].rolling(H).max()[::-1].shift(-1)  # classification target (Eq. 4.1)
    ltv = gr["lead_time_days"]
    out["lt_mean"] = ltv.rolling(90, min_periods=1).mean().ffill(); out["lt_std"] = ltv.rolling(90, min_periods=2).std().ffill()
    out["short_days30"] = gr["upstream_shortage"].rolling(30, min_periods=1).sum()
    out["disrupt_days14"] = gr["distributor_disruption"].rolling(14, min_periods=1).sum()
    out["stockout_days30"] = gr["stockout"].rolling(30, min_periods=1).sum()
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    F = df.groupby(["facility_id", "drug_id"], group_keys=False).apply(series_features)
    df = pd.concat([df, F], axis=1)
    df["lt_mean"] = df["lt_mean"].fillna(df.groupby("supplier_id")["lead_time_days"].transform("mean"))
    df["lt_std"] = df["lt_std"].fillna(df["lt_mean"] * 0.3)
    df["days_of_supply"] = df["on_hand"] / (df["d_mean28"] + 1e-6)                                  # Eq. 3.1
    df["pos_days_of_supply"] = (df["on_hand"] + df["on_order"]) / (df["d_mean28"] + 1e-6)
    df["safety_stock"] = 1.65 * np.sqrt(df["lt_mean"] * df["d_std28"] ** 2 + df["d_mean28"] ** 2 * df["lt_std"] ** 2)  # Eq. 3.2
    df["rop"] = df["d_mean28"] * df["lt_mean"] + df["safety_stock"]                                  # Eq. 3.3
    df["rop_gap"] = (df["on_hand"] + df["on_order"] - df["rop"]) / (df["d_mean28"] + 1e-6)
    df["woy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)
    df["woy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)
    df["cat_code"] = df["category"].astype("category").cat.codes
    df["ftype_code"] = df["facility_type"].astype("category").cat.codes
    return df


def make_snapshots(df: pd.DataFrame):
    """One row per series every 7 days (rows already out of stock are skipped); time-based split."""
    snap = df[(df.day >= 35) & (df.day % 7 == 0) & (df.day <= N_DAYS - H - 1)].dropna(subset=["future_stockout14", "d_mean28"]).copy()
    snap = snap[snap.on_hand > 0]
    return snap[snap.day < TRAIN_END_DAY].copy(), snap[snap.day >= TRAIN_END_DAY].copy()


def add_forecast_features(D: pd.DataFrame) -> pd.DataFrame:
    """Features using the 14-day forecast fc14 and the shipment-tracking API (Eq. 3.4 and 3.5)."""
    D["cover_until_eta"] = D.on_hand / (D.fc14 / H + 1e-6) - D.eta_days
    D["fc_net14"] = (D.on_hand + D.intransit_14d - D.fc14) / (D.fc14 / H + 1e-6)
    D["fc_dos"] = (D.on_hand + D.on_order) / (D.fc14 / H + 1e-6)
    D["fc_gap"] = (D.on_hand + D.on_order - D.fc14 * (D.lt_mean * (1 + 1.6 * D.upstream_shortage)) / H) / (D.fc14 / H + 1e-6)
    return D


def main():
    df = add_features(pd.read_pickle(DATA / "clean_panel.pkl"))
    train, test = make_snapshots(df)
    train.to_pickle(DATA / "train_snap.pkl"); test.to_pickle(DATA / "test_snap.pkl")
    print(f"[features] {len(train):,} training snapshots (2024), {len(test):,} test snapshots (Jan-Jun 2025)")


if __name__ == "__main__":
    main()
