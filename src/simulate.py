"""simulate.py - replay Jan-Jun 2025 under the existing and the proposed replenishment policy (report Sections 5.4, 6.1.5).

Both policies see identical demand and identical random draws for lead times and fill rates
(common random numbers), so the difference in outcomes is caused only by the policy.
  * existing policy : weekly review, static reorder point, order up to a static level
  * proposed policy : daily review, forecast demand + live lead time -> dynamic reorder point
                      (99 % service level), alternate manufacturer during an upstream shortage
"""
import json, zlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA, FIGURES, OUTPUTS, SERVICE_Z, TRAIN_END_DAY, N_DAYS

T0, T1 = TRAIN_END_DAY, N_DAYS


def simulate():
    df = pd.read_pickle(DATA / "clean_truth.pkl"); drg = pd.read_csv(DATA / "drugs.csv").set_index("drug_id")
    slt = np.array(json.load(open(DATA / "meta.json"))["supplier_lt"])
    res = []; daily = {"base": np.zeros(T1 - T0), "prop": np.zeros(T1 - T0)}
    for (fid, did), g in df.groupby(["facility_id", "drug_id"]):
        g = g.sort_values("day"); dem = g.demand.values; sh = g.upstream_shortage.values
        dis = g.distributor_disruption.values; wev = g.weather_event.values
        drow = drg.loc[did]; sidx = int(drow.supplier_id[1:]) - 1; nm = drow.n_manufacturers
        hist = dem[:T0]; mu = hist[:120].mean(); sd = hist[:120].std()
        s_pt = g.static_rop.iloc[0]; S_pt = s_pt + mu * 14
        season = pd.Series(hist).rolling(28, center=True, min_periods=7).mean().values / hist.mean()   # 2024 seasonal index
        out = {}
        for pol in ("base", "prop"):
            rng = np.random.default_rng(zlib.crc32((fid + did).encode()))            # same random numbers for both policies
            oh = g.on_hand.values[T0 - 1]
            pipe = [(T0 + 3, g.on_order.values[T0 - 1])] if g.on_order.values[T0 - 1] > 0 else []
            ew = hist[-28:].mean(); so = served = tot = inv = norders = alt_n = 0
            for k in range(T0, T1):
                oh += sum(q for a, q in pipe if a == k); pipe = [(a, q) for a, q in pipe if a != k]
                d = dem[k]; s_ = min(oh, d); oh -= s_; served += s_; tot += d
                if d > s_: so += 1; daily[pol][k - T0] += 1
                inv += oh; onord = sum(q for a, q in pipe)
                mult = 1 + 1.6 * sh[k] + 1.2 * dis[k] + 0.8 * wev[k]
                ew = 0.9 * ew + 0.1 * d
                u1, u2, u3 = rng.random(), rng.random(), rng.random()
                lognorm = lambda L: max(1, np.exp(np.log(L) + 0.3 * np.sqrt(-2 * np.log(u2 + 1e-12)) * np.cos(2 * np.pi * u3)))
                if pol == "base":
                    if (k % 7 == 0) and (oh + onord < s_pt):
                        fill = 0.35 + 0.4 * u1 if sh[k] else 0.93 + 0.07 * u1
                        pipe.append((k + int(round(lognorm(slt[sidx] * mult))), (S_pt - oh - onord) * fill)); norders += 1
                else:
                    si = season[(k - 365) % 365]; si = 1.0 if np.isnan(si) else si
                    dh = ew * si                                                          # forecast daily demand
                    L = slt[sidx] * mult                                                  # live lead-time estimate
                    altsrc = sh[k] and nm > 1
                    if altsrc: L = slt[sidx] * 1.3 + 2
                    rop = dh * L + SERVICE_Z * np.sqrt(L * sd ** 2 + dh ** 2 * (0.3 * L) ** 2)   # Eq. 3.2-3.3
                    if oh + onord < rop:
                        if sh[k] and not altsrc: fill = 0.35 + 0.4 * u1
                        elif altsrc: fill = 0.85 + 0.12 * u1; alt_n += 1
                        else: fill = 0.93 + 0.07 * u1
                        pipe.append((k + int(round(lognorm(L))), (rop + dh * 12 - oh - onord) * fill)); norders += 1
            out[pol] = dict(so=so, fill=served / tot, inv_days=inv / (T1 - T0) / max(mu, 1e-6), orders=norders, alt=alt_n)
        res.append(dict(facility_id=fid, drug_id=did, category=drow.category, **{f"{p}_{k}": v for p in out for k, v in out[p].items()}))
    return pd.DataFrame(res), daily


def main():
    r, daily = simulate(); N = T1 - T0
    S = {p: dict(stockout_pct=r[f"{p}_so"].sum() / (len(r) * N) * 100, stockout_days=int(r[f"{p}_so"].sum()),
                 fill_rate=r[f"{p}_fill"].mean() * 100, median_inventory_days=float(r[f"{p}_inv_days"].median()),
                 orders=int(r[f"{p}_orders"].sum())) for p in ("base", "prop")}
    S["prop"]["alternate_source_orders"] = int(r.prop_alt.sum())
    S["stockout_day_reduction_pct"] = (1 - S["prop"]["stockout_days"] / S["base"]["stockout_days"]) * 100
    json.dump(S, open(OUTPUTS / "simulation.json", "w"), indent=1)
    cat = r.groupby("category")[["base_so", "prop_so"]].sum() / (r.groupby("category").size().values[:, None] * N) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.6)); x = np.arange(len(cat))
    ax.bar(x - .2, cat.base_so, .4, label="Existing static policy", color="#c0504d"); ax.bar(x + .2, cat.prop_so, .4, label="Proposed predictive policy", color="#4f81bd")
    ax.set_xticks(x); ax.set_xticklabels(cat.index, rotation=30, ha="right"); ax.set_ylabel("Stockout days (%)"); ax.legend()
    ax.set_title("Simulated Stockout Rate by Category (Jan-Jun 2025)"); plt.savefig(FIGURES / "simulation_by_category.png", bbox_inches="tight"); plt.close()
    b, p = S["base"], S["prop"]
    print(f"[simulate] stockout days {b['stockout_days']:,} -> {p['stockout_days']:,} ({S['stockout_day_reduction_pct']:.1f}% fewer) | "
          f"fill rate {b['fill_rate']:.2f}% -> {p['fill_rate']:.2f}% | inventory {b['median_inventory_days']:.1f} -> {p['median_inventory_days']:.1f} days")


if __name__ == "__main__":
    main()
