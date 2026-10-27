"""gen_data.py - realistic synthetic pharmaceutical supply-chain data.

Simulates what the API ingestion layer collects: facility inventory and dispensing
feeds, distributor order / shipment-tracking feeds and an upstream drug-shortage
feed, as a daily panel for 30 facilities x 50 medicines (1 Jan 2024 - 30 Jun 2025).
The existing (static, weekly) reorder policy is simulated to create realistic stockouts.
"""
import json
import numpy as np
import pandas as pd
from config import DATA, SEED, START_DATE, N_DAYS

rng = np.random.default_rng(SEED)
START, DAYS = pd.Timestamp(START_DATE), N_DAYS
dates = pd.date_range(START, periods=DAYS, freq="D")

cities = ["Coimbatore", "Chennai", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode", "Vellore",
          "Thoothukudi", "Thanjavur", "Dindigul", "Tiruppur", "Karur", "Namakkal", "Hosur"]
FTYPES = ["Tertiary Hospital", "District Hospital", "Retail Pharmacy Chain", "Primary Health Centre"]
FSIZE = dict(zip(FTYPES, [2.2, 1.4, 1.0, 0.6]))
facilities = []
for i in range(30):
    ftype = FTYPES[i % 4]
    facilities.append(dict(facility_id=f"F{i+1:03d}", city=cities[i % 15], facility_type=ftype,
                           size=FSIZE[ftype] * rng.uniform(0.8, 1.2)))

drug_spec = [  # name, category, base daily demand, season type, cold chain
    ("Amoxicillin 500mg", "Antibiotic", 120, "winter", 0), ("Azithromycin 500mg", "Antibiotic", 60, "winter", 0),
    ("Ceftriaxone 1g Inj", "Antibiotic", 35, "monsoon", 0), ("Ciprofloxacin 500mg", "Antibiotic", 45, "monsoon", 0),
    ("Doxycycline 100mg", "Antibiotic", 30, "monsoon", 0), ("Meropenem 1g Inj", "Antibiotic", 8, "flat", 0),
    ("Piperacillin-Tazobactam", "Antibiotic", 10, "flat", 0), ("Paracetamol 650mg", "Analgesic", 300, "monsoon", 0),
    ("Ibuprofen 400mg", "Analgesic", 110, "flat", 0), ("Diclofenac 50mg", "Analgesic", 80, "flat", 0),
    ("Tramadol 50mg", "Analgesic", 25, "flat", 0), ("Morphine 10mg Inj", "Analgesic", 5, "flat", 0),
    ("Metformin 500mg", "Antidiabetic", 220, "trend", 0), ("Glimepiride 2mg", "Antidiabetic", 90, "trend", 0),
    ("Insulin Glargine", "Antidiabetic", 20, "trend", 1), ("Insulin Regular", "Antidiabetic", 25, "trend", 1),
    ("Sitagliptin 100mg", "Antidiabetic", 40, "trend", 0), ("Amlodipine 5mg", "Cardiovascular", 180, "trend", 0),
    ("Atorvastatin 20mg", "Cardiovascular", 150, "trend", 0), ("Telmisartan 40mg", "Cardiovascular", 130, "trend", 0),
    ("Metoprolol 50mg", "Cardiovascular", 70, "flat", 0), ("Clopidogrel 75mg", "Cardiovascular", 60, "flat", 0),
    ("Enoxaparin 40mg Inj", "Cardiovascular", 12, "flat", 1), ("Heparin 5000IU Inj", "Cardiovascular", 15, "flat", 0),
    ("Salbutamol Inhaler", "Respiratory", 40, "winter", 0), ("Budesonide Inhaler", "Respiratory", 22, "winter", 0),
    ("Montelukast 10mg", "Respiratory", 55, "winter", 0), ("Levocetirizine 5mg", "Respiratory", 95, "winter", 0),
    ("Prednisolone 10mg", "Respiratory", 35, "winter", 0), ("Oseltamivir 75mg", "Antiviral", 6, "winter", 0),
    ("Acyclovir 400mg", "Antiviral", 12, "flat", 0), ("Tenofovir 300mg", "Antiviral", 18, "flat", 0),
    ("ORS Sachet", "Gastrointestinal", 160, "summer", 0), ("Pantoprazole 40mg", "Gastrointestinal", 170, "flat", 0),
    ("Ondansetron 4mg", "Gastrointestinal", 50, "monsoon", 0), ("Metronidazole 400mg", "Gastrointestinal", 65, "monsoon", 0),
    ("Loperamide 2mg", "Gastrointestinal", 30, "summer", 0), ("Zinc Sulphate 20mg", "Gastrointestinal", 45, "summer", 0),
    ("Cisplatin 50mg Inj", "Oncology", 2, "flat", 0), ("Methotrexate 2.5mg", "Oncology", 4, "flat", 0),
    ("Tamoxifen 20mg", "Oncology", 5, "flat", 0), ("Carboplatin 450mg Inj", "Oncology", 2, "flat", 0),
    ("Rabies Vaccine", "Vaccine", 8, "flat", 1), ("Tetanus Toxoid", "Vaccine", 14, "flat", 1),
    ("Hepatitis B Vaccine", "Vaccine", 6, "flat", 1), ("Anti-Snake Venom", "Emergency", 3, "monsoon", 1),
    ("Adrenaline 1mg Inj", "Emergency", 4, "flat", 0), ("Atropine 0.6mg Inj", "Emergency", 4, "flat", 0),
    ("Normal Saline 500ml", "Emergency", 70, "summer", 0), ("Dexamethasone 4mg Inj", "Emergency", 18, "flat", 0)]
drugs = [dict(drug_id=f"D{j+1:03d}", drug_name=n, category=c, base=b, season=s, cold_chain=cc)
         for j, (n, c, b, s, cc) in enumerate(drug_spec)]

n_sup = 8
supplier_lt = rng.uniform(4, 11, n_sup)          # mean lead time (days) per distributor
drug_sup = rng.integers(0, n_sup, len(drugs))     # primary distributor of each medicine
n_mfr = rng.choice([1, 1, 2, 3, 4, 5], len(drugs))  # approved manufacturers per medicine

# upstream shortage episodes (openFDA / CDSCO-style shortage feed)
shortage = np.zeros((len(drugs), DAYS), dtype=np.int8)
for j in range(len(drugs)):
    for _ in range(rng.poisson(1.6 / n_mfr[j] * 1.5)):
        s = rng.integers(0, DAYS - 20); L = rng.integers(14, 75)
        shortage[j, s:s + L] = 1
# distributor disruptions (transport strikes, floods)
disrupt = np.zeros((n_sup, DAYS), dtype=np.int8)
for sidx in range(n_sup):
    for _ in range(rng.poisson(2)):
        s = rng.integers(0, DAYS - 10); disrupt[sidx, s:s + rng.integers(3, 12)] = 1
# regional monsoon flood (Nov-Dec 2024) affecting coastal cities
flood = ((dates >= "2024-11-25") & (dates <= "2024-12-12")).astype(int)

doy = dates.dayofyear.values; dow = dates.dayofweek.values; t = np.arange(DAYS)


def season_curve(kind):
    if kind == "winter":  return 1 + 0.35 * np.cos(2 * np.pi * (doy - 15) / 365)
    if kind == "monsoon": return 1 + 0.40 * np.exp(-((doy - 290) / 40.0) ** 2) + 0.15 * np.exp(-((doy - 200) / 35.0) ** 2)
    if kind == "summer":  return 1 + 0.45 * np.exp(-((doy - 125) / 35.0) ** 2)
    if kind == "trend":   return 1 + 0.00045 * t
    return np.ones(DAYS)


weekly = np.array([1.05, 1.08, 1.04, 1.02, 1.03, 0.95, 0.83])[dow]

rows = []
for f in facilities:
    coastal = f["city"] in ("Chennai", "Thoothukudi")
    ob = np.ones(DAYS)                       # local outbreak clusters (dengue / fever)
    for _ in range(rng.poisson(1.2)):
        s = rng.integers(0, DAYS - 30); L = rng.integers(10, 30)
        ob[s:s + L] *= rng.uniform(1.3, 1.9)
    for j, d in enumerate(drugs):
        lam = d["base"] * f["size"] * season_curve(d["season"]) * weekly
        if d["category"] in ("Antibiotic", "Analgesic", "Gastrointestinal", "Respiratory", "Emergency"):
            lam = lam * ob
        demand = rng.negative_binomial(8, 8 / (8 + lam))
        sidx = drug_sup[j]
        mu_d = demand[:120].mean(); sd = demand[:120].std()
        # existing static policy: weekly review, s = mu*(LT+R) + 1*sd*sqrt(LT+R), order up to S = s + 14*mu
        s_pt = mu_d * (supplier_lt[sidx] + 7) + 1.0 * sd * np.sqrt(supplier_lt[sidx] + 7)
        S_pt = s_pt + mu_d * 14
        on_hand = S_pt * rng.uniform(0.6, 1.0); pipeline = []   # (arrival_day, qty)
        review = rng.integers(0, 7)
        for k in range(DAYS):
            arrived = sum(q for a, q in pipeline if a == k); pipeline = [(a, q) for a, q in pipeline if a != k]
            on_hand += arrived
            dem = demand[k]; sold = min(on_hand, dem); on_hand -= sold
            on_order = sum(q for a, q in pipeline)
            lt_mult = 1 + 1.6 * shortage[j, k] + 1.2 * disrupt[sidx, k] + (0.8 if (coastal and flood[k]) else 0)
            ordered = 0; lt_real = np.nan
            if (k % 7 == review) and (on_hand + on_order < s_pt):
                q = S_pt - on_hand - on_order
                fill = rng.uniform(0.35, 0.75) if shortage[j, k] else rng.uniform(0.93, 1.0)
                lt_real = max(1, rng.lognormal(np.log(supplier_lt[sidx] * lt_mult), 0.3))
                pipeline.append((k + int(round(lt_real)), q * fill)); ordered = q
            if pipeline:                     # shipment-tracking API: ETA of next arrival (noisy) and qty due in 14 days
                a0 = min(a for a, q in pipeline); eta = max(0, a0 - k + rng.normal(0, 1.2))
                in14 = sum(q for a, q in pipeline if a - k <= 14)
            else:
                eta = 30.0; in14 = 0.0
            rows.append((f["facility_id"], d["drug_id"], k, dem, sold, on_hand, on_order, ordered, lt_real,
                         shortage[j, k], disrupt[sidx, k], int(coastal and flood[k]), int(on_hand <= 0 and dem > sold),
                         round(s_pt, 1), round(eta, 1), round(in14, 0)))

cols = ["facility_id", "drug_id", "day", "demand", "dispensed", "on_hand", "on_order", "order_qty", "lead_time_days",
        "upstream_shortage", "distributor_disruption", "weather_event", "stockout", "static_rop", "eta_days", "intransit_14d"]
df = pd.DataFrame(rows, columns=cols)
df["date"] = START + pd.to_timedelta(df["day"], unit="D")
fac = pd.DataFrame(facilities); drg = pd.DataFrame(drugs)
drg["supplier_id"] = [f"S{s+1:02d}" for s in drug_sup]; drg["n_manufacturers"] = n_mfr
df = df.merge(fac[["facility_id", "city", "facility_type"]], on="facility_id").merge(
    drg[["drug_id", "drug_name", "category", "cold_chain", "supplier_id", "n_manufacturers"]], on="drug_id")
df["on_hand"] = df["on_hand"].round(0); df["on_order"] = df["on_order"].round(0); df["order_qty"] = df["order_qty"].round(0)

# API ingestion problems: missing fields and duplicate pushes
raw = df.copy(); n = len(raw)
for col, frac in [("on_hand", 0.021), ("demand", 0.012), ("on_order", 0.009)]:
    idx = rng.choice(n, int(n * frac), replace=False); raw.loc[idx, col] = np.nan
idx = rng.choice(n, int(n * 0.004), replace=False); raw.loc[idx, "supplier_id"] = np.nan
idx = rng.choice(n, int(n * 0.006), replace=False); raw.loc[idx, "category"] = np.nan
dup = raw.sample(int(n * 0.003), random_state=1)
raw = pd.concat([raw, dup]).sort_values(["facility_id", "drug_id", "day"]).reset_index(drop=True)

raw.to_pickle(DATA / "raw_panel.pkl"); df.to_pickle(DATA / "clean_truth.pkl")
fac.to_csv(DATA / "facilities.csv", index=False); drg.to_csv(DATA / "drugs.csv", index=False)
json.dump(dict(supplier_lt=supplier_lt.tolist()), open(DATA / "meta.json", "w"))
print(f"[gen_data] {len(raw):,} raw records written to data/  |  stockout-day rate {df.stockout.mean():.2%}")
