"""models.py - 14-day demand forecasting and stockout-risk classification (report Sections 5.3 and 6.1)."""
import json, os, warnings
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve, mean_absolute_error, mean_squared_error)
from config import DATA, FIGURES, OUTPUTS, HORIZON, VALID_START_DAY
from features import add_forecast_features

warnings.filterwarnings("ignore")
H = HORIZON
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})
# deterministic LightGBM training so results are identical on every run
LGB_FIXED = dict(deterministic=True, **{"force_col_wise": True})

FC_FEATS = ["d_mean7", "d_mean14", "d_mean28", "d_std28", "d_trend", "d_lag7", "woy_sin", "woy_cos", "cat_code", "ftype_code",
            "upstream_shortage", "weather_event", "stockout_days30", "cold_chain"]
BASE_FEATS = ["on_hand", "on_order", "days_of_supply", "pos_days_of_supply", "d_mean7", "d_mean28", "d_std28", "d_trend",
              "lt_mean", "lt_std", "upstream_shortage", "short_days30", "distributor_disruption", "disrupt_days14",
              "weather_event", "stockout_days30", "rop_gap", "n_manufacturers", "cold_chain", "cat_code", "ftype_code", "woy_sin", "woy_cos"]
TRACK_FEATS = ["eta_days", "intransit_14d", "cover_until_eta"]
PROP_FEATS = BASE_FEATS + TRACK_FEATS + ["fc14", "fc_dos", "fc_gap", "fc_net14"]
PROPOSED = "Proposed (LightGBM + forecast features)"


def lgb_reg(n=600):
    return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, random_state=1, **LGB_FIXED)


def lgb_clf(spw):
    return lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, scale_pos_weight=spw, verbose=-1, random_state=1, **LGB_FIXED)


def forecast_metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                WMAPE=float(np.abs(y - p).sum() / y.sum() * 100), Bias=float((p - y).sum() / y.sum() * 100))


def best_threshold(y, p):
    """Threshold that maximises F1 on the validation window (Oct-Dec 2024)."""
    pr, rc, th = precision_recall_curve(y, p); f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return float(th[np.argmax(f1[:-1])])


def train_forecaster(train, test):
    ytr = train.future_demand14
    fc = {"Naive (last 14 days)": test.d_sum_last14.values, "Moving Average (28-day)": test.d_mean28.values * H}
    rf = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=1)
    rf.fit(train[FC_FEATS], np.log1p(ytr)); fc["Random Forest"] = np.expm1(rf.predict(test[FC_FEATS]))
    model = lgb_reg(); model.fit(train[FC_FEATS], np.log1p(ytr)); fc["LightGBM (proposed)"] = np.expm1(model.predict(test[FC_FEATS]))
    metrics = {k: forecast_metrics(test.future_demand14.values, v) for k, v in fc.items()}
    # out-of-fold forecasts for the training year, so the classifier never sees leaked forecasts
    oof = np.zeros(len(train)); qs = np.quantile(train.day, [0, .5, 1])
    for lo, hi in [(qs[0], qs[1]), (qs[1], qs[2] + 1)]:
        mask = ~((train.day >= lo) & (train.day < hi))
        m = lgb_reg(400); m.fit(train.loc[mask, FC_FEATS], np.log1p(ytr[mask]))
        oof[~mask.values] = np.expm1(m.predict(train.loc[~mask, FC_FEATS]))
    train["fc14"] = oof; test["fc14"] = fc["LightGBM (proposed)"]
    return model, metrics


def train_classifiers(train, test):
    for D in (train, test): add_forecast_features(D)
    ytr, yte = train.future_stockout14.astype(int), test.future_stockout14.astype(int)
    spw = (1 - ytr.mean()) / ytr.mean()
    val = train[train.day >= VALID_START_DAY]; yval = val.future_stockout14.astype(int)
    P, thr = {}, {}
    k = "Rule-based ROP alert (existing)"; P[k] = (test.on_hand + test.on_order < test.static_rop).astype(float).values; thr[k] = 0.5
    sc = StandardScaler().fit(train[BASE_FEATS])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(train[BASE_FEATS]), ytr)
    P["Logistic Regression"] = lr.predict_proba(sc.transform(test[BASE_FEATS]))[:, 1]
    thr["Logistic Regression"] = best_threshold(yval, lr.predict_proba(sc.transform(val[BASE_FEATS]))[:, 1])
    rf = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=4, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=1).fit(train[BASE_FEATS], ytr)
    P["Random Forest"] = rf.predict_proba(test[BASE_FEATS])[:, 1]; thr["Random Forest"] = best_threshold(yval, rf.predict_proba(val[BASE_FEATS])[:, 1])
    xg = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=spw, eval_metric="logloss", random_state=1, n_jobs=-1).fit(train[BASE_FEATS], ytr)
    P["XGBoost"] = xg.predict_proba(test[BASE_FEATS])[:, 1]; thr["XGBoost"] = best_threshold(yval, xg.predict_proba(val[BASE_FEATS])[:, 1])
    clf = lgb_clf(spw).fit(train[PROP_FEATS], ytr)
    P[PROPOSED] = clf.predict_proba(test[PROP_FEATS])[:, 1]; thr[PROPOSED] = best_threshold(yval, clf.predict_proba(val[PROP_FEATS])[:, 1])
    metrics = {}
    for k, p in P.items():
        yh = (p >= thr[k]).astype(int)
        metrics[k] = dict(Accuracy=accuracy_score(yte, yh) * 100, Precision=precision_score(yte, yh) * 100, Recall=recall_score(yte, yh) * 100,
                          F1=f1_score(yte, yh) * 100, ROC_AUC=None if k.startswith("Rule") else float(roc_auc_score(yte, p)),
                          PR_AUC=float(average_precision_score(yte, p)), threshold=thr[k])
    return clf, P, metrics, yte


def plots(P, yte, clf, threshold):
    cm = confusion_matrix(yte, (P[PROPOSED] >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax, xticklabels=["No stockout", "Stockout"], yticklabels=["No stockout", "Stockout"])
    ax.set_xlabel("Predicted (14-day horizon)"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix - Proposed Model")
    plt.savefig(FIGURES / "confusion_matrix.png"); plt.close()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for k, p in P.items():
        if k.startswith("Rule"): continue
        fpr, tpr, _ = roc_curve(yte, p); ax.plot(fpr, tpr, label=f"{k.split(' (')[0]} (AUC={roc_auc_score(yte, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8); ax.set_title("ROC Curves of Stockout Prediction Models"); plt.savefig(FIGURES / "roc_curves.png"); plt.close()
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=PROP_FEATS).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 4.4)); ax.barh(imp.index, imp.values / imp.sum() * 100, color="#4f81bd")
    ax.set_xlabel("Relative importance (% of total gain)"); ax.set_title("Top 15 Feature Importances - Proposed Model")
    plt.savefig(FIGURES / "feature_importance.png"); plt.close()
    return cm


def main():
    train, test = pd.read_pickle(DATA / "train_snap.pkl"), pd.read_pickle(DATA / "test_snap.pkl")
    forecaster, fmetrics = train_forecaster(train, test)
    clf, P, cmetrics, yte = train_classifiers(train, test)
    cm = plots(P, yte, clf, cmetrics[PROPOSED]["threshold"])
    forecaster.booster_.save_model(str(OUTPUTS / "forecaster.txt"))
    clf.booster_.save_model(str(OUTPUTS / "stockout_classifier.txt"))
    test["risk"] = P[PROPOSED]
    test.to_pickle(OUTPUTS / "test_scored.pkl")
    json.dump({"forecast_full_test": fmetrics, "classification": cmetrics, "confusion_matrix": cm.tolist(),
               "proposed_threshold": cmetrics[PROPOSED]["threshold"], "feature_list": PROP_FEATS},
              open(OUTPUTS / "metrics.json", "w"), indent=1)
    print("[models] 14-day demand forecast WMAPE (all 1,500 series, Jan-Jun 2025):")
    for k, v in fmetrics.items(): print(f"   {k:<26} {v['WMAPE']:6.2f}%")
    print("[models] 14-day stockout prediction (Jan-Jun 2025 test period):")
    for k, v in cmetrics.items():
        auc = "  -  " if v["ROC_AUC"] is None else f"{v['ROC_AUC']:.3f}"
        print(f"   {k:<40} Acc {v['Accuracy']:5.2f}  P {v['Precision']:5.2f}  R {v['Recall']:5.2f}  F1 {v['F1']:5.2f}  ROC-AUC {auc}")


if __name__ == "__main__":
    main()
