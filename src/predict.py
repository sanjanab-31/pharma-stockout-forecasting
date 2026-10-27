"""predict.py - stockout prediction module (report Section 5.4).

Scores the latest snapshot of a facility-medicine pair with the saved classifier and
returns the 14-day forecast, the stockout risk and a recommended action.

    python predict.py F003 D045          # facility F003, Hepatitis B Vaccine
    python predict.py --top 10           # ten highest-risk pairs on the latest snapshot
"""
import argparse, json
import lightgbm as lgb
import pandas as pd
from config import OUTPUTS

_model = None


def _load():
    global _model
    if _model is None:
        _model = lgb.Booster(model_file=str(OUTPUTS / "stockout_classifier.txt"))
    meta = json.load(open(OUTPUTS / "metrics.json"))
    snaps = pd.read_pickle(OUTPUTS / "test_scored.pkl")
    return _model, meta["feature_list"], meta["proposed_threshold"], snaps[snaps.day == snaps.day.max()]


def action(row, threshold):
    if row["risk"] < threshold: return "No action"
    if row["upstream_shortage"] and row["n_manufacturers"] <= 1: return "Expedite open order / inter-facility transfer"
    if row["upstream_shortage"]: return "Switch to alternate manufacturer"
    return "Reorder now"


def predict(facility_id: str, drug_id: str) -> dict:
    model, feats, thr, latest = _load()
    row = latest[(latest.facility_id == facility_id) & (latest.drug_id == drug_id)]
    if row.empty:
        raise KeyError(f"no snapshot for {facility_id}/{drug_id} (it may currently be out of stock)")
    r = row.iloc[0].copy(); r["risk"] = float(model.predict(row[feats])[0])
    return dict(facility_id=facility_id, city=r.city, drug_id=drug_id, drug_name=r.drug_name, date=str(r.date.date()),
                days_of_supply=round(float(r.days_of_supply), 1), forecast_14d=round(float(r.fc14), 0),
                next_shipment_eta=None if r.eta_days >= 29.9 else round(float(r.eta_days), 1),
                upstream_shortage=bool(r.upstream_shortage), risk=round(r["risk"], 3), alert=bool(r["risk"] >= thr),
                action=action(r, thr))


def top_risks(n=10):
    _, _, thr, latest = _load()
    top = latest.sort_values("risk", ascending=False).head(n)
    return [dict(facility_id=r.facility_id, city=r.city, drug_name=r.drug_name, days_of_supply=round(float(r.days_of_supply), 1),
                 risk=round(float(r.risk), 3), action=action(r, thr)) for _, r in top.iterrows()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("facility_id", nargs="?"); ap.add_argument("drug_id", nargs="?")
    ap.add_argument("--top", type=int, default=0); a = ap.parse_args()
    if a.top or not a.facility_id:
        for r in top_risks(a.top or 10): print(r)
    else:
        print(json.dumps(predict(a.facility_id, a.drug_id), indent=1))


if __name__ == "__main__":
    main()
