"""preprocess.py - de-duplication, master-data lookup and missing-value imputation (report Section 4.1)."""
import json
import pandas as pd
from config import DATA, OUTPUTS, START_DATE

FIELDS = ["on_hand", "demand", "on_order", "supplier_id", "category"]


def preprocess(raw: pd.DataFrame, drugs: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates(["facility_id", "drug_id", "day"]).sort_values(["facility_id", "drug_id", "day"]).reset_index(drop=True)
    master = drugs.set_index("drug_id")
    for col in ["category", "supplier_id"]:                       # exact recovery from the drug master
        df[col] = df[col].fillna(df["drug_id"].map(master[col]))
    g = df.groupby(["facility_id", "drug_id"])
    m = df["demand"].isna() & (df["stockout"] == 0)               # dispensed == demand when not out of stock
    df.loc[m, "demand"] = df.loc[m, "dispensed"]
    df["demand"] = g["demand"].transform(lambda s: s.fillna(s.rolling(7, min_periods=1).median()).ffill().bfill())
    df["on_hand"] = g["on_hand"].transform(lambda s: s.interpolate(limit_direction="both")).round(0)
    df["on_order"] = g["on_order"].transform(lambda s: s.ffill().fillna(0))
    df["date"] = pd.Timestamp(START_DATE) + pd.to_timedelta(df["day"], unit="D")
    return df


def main():
    raw = pd.read_pickle(DATA / "raw_panel.pkl"); drugs = pd.read_csv(DATA / "drugs.csv")
    before = raw[FIELDS].isna().sum(); n_dup = int(raw.duplicated(["facility_id", "drug_id", "day"]).sum())
    df = preprocess(raw, drugs); after = df[FIELDS].isna().sum()
    df.to_pickle(DATA / "clean_panel.pkl")
    json.dump({"raw_rows": len(raw), "duplicates_removed": n_dup, "clean_rows": len(df),
               "missing_before": before.astype(int).to_dict(), "missing_after": after.astype(int).to_dict()},
              open(OUTPUTS / "preprocess_summary.json", "w"), indent=1)
    print(f"[preprocess] removed {n_dup:,} duplicates, imputed {int(before.sum()):,} missing values -> {len(df):,} clean rows")


if __name__ == "__main__":
    main()
