from pathlib import Path

import itertools
import numpy as np
import pandas as pd

df = pd.read_parquet(Path(__file__).resolve().parents[2] / "datos/intermedios/work.parquet")

# --- transport with lambda
LAM = 0.30
den = df.km_highway + LAM * (df.km_tertiary_named + df.km_highway_ring25)
g_T = np.where(den > 0, np.clip(1 - df.km_highway / den.replace(0, np.nan), 0, 1), 0.0)
g_T = np.where(df.km_highway == 0, 1.0, g_T)
g_B = df.g_B.values
g_P = df.g_P.values

# --- p_T tiers
mt = df.km_motorway + df.km_trunk
hw = df.km_highway
pT = np.select(
    [mt > 0, hw >= 2.0, hw > 0],
    [0.85, 0.55, 0.40],
    default=np.where(df.km_highway_ring25 > 0, 0.30, 0.25),
)
pB = np.where((df.bldg > 0) & (df.hu > 0), 0.97,
              np.where((df.hu > 0) | (df.bldg > 0), 0.90, 0.10))
pP = np.where((df.poi_total > 0) & (df["pop"] > 0), 0.95,
              np.where((df["pop"] > 0) | (df.poi_total > 0), 0.85, 0.20))

G = np.column_stack([g_T, g_B, g_P])
P = np.column_stack([pT, pB, pP])
u = np.zeros(len(df))
probs = {}
for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
    pr = np.ones(len(df))
    for i in range(3):
        pr = pr * (P[:, i] if i in S else (1 - P[:, i]))
    probs[S] = pr
    u += pr * G[:, list(S)].mean(axis=1)

print("config prob means:")
for k, v in probs.items():
    print("  ", k, round(float(v.mean()), 4), "sum", round(sum(probs.values()).mean(), 6))
print("p means T/B/P:", [round(float(m), 3) for m in P.mean(axis=0)])
print("\nu: mean=%.4f med=%.4f p90=%.4f p99=%.4f max=%.4f mean(u2)=%.5f share>0.5=%.4f share>0=%.4f"
      % (u.mean(), np.median(u), np.quantile(u, .9), np.quantile(u, .99), u.max(), (u ** 2).mean(), (u > .5).mean(), (u > 0).mean()))
print("share u>0 by region\n", pd.Series(u > 0).groupby(df.region.values).mean().round(4))
print("\nu by region\n", pd.Series(u).groupby(df.region.values).agg(["mean", "median", "max"]).round(3))

k2 = np.sqrt(0.0115 / (u ** 2).mean())
km = 0.100 / u.mean()
print("\nk by second moment=%.3f  k by mean cap=%.3f" % (k2, km))
k = min(k2, km)
pred = k * u
print("chosen k=%.3f -> mean=%.4f mean2=%.5f share>0.5=%.4f share>0.107=%.3f"
      % (k, pred.mean(), (pred ** 2).mean(), (pred > .5).mean(), (pred > .107).mean()))
print("\npred by region\n", pd.Series(pred).groupby(df.region.values).agg(["mean", "median", "max", lambda s: (s > .5).mean()]).round(4))
print("\npred quantiles", dict(zip([.1, .25, .5, .75, .9, .95, .99, 1.0], np.quantile(pred, [.1, .25, .5, .75, .9, .95, .99, 1.0]).round(4))))
print("\ncorr g_T,g_P", round(float(np.corrcoef(g_T, g_P)[0, 1]), 3),
      "g_B,g_P", round(float(np.corrcoef(g_B, g_P)[0, 1]), 3))
print("\ntop-20 pred tracts:")
top = pd.DataFrame({"GEOID": df.index, "region": df.region, "pred": pred, "gT": g_T, "gB": g_B, "gP": g_P, "pT": pT})
print(top.sort_values("pred", ascending=False).head(12).to_string(index=False))
