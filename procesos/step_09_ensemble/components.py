import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
base = ROOT / "datos" / "features_base"
b = pd.read_parquet(base / "features_buildings.parquet").set_index("GEOID")
r = pd.read_parquet(base / "features_roads.parquet").set_index("GEOID")
p = pd.read_parquet(base / "features_pois.parquet").set_index("GEOID")
df = pd.DataFrame(index=b.index)
df["region"] = b.region
df["pop"] = b.pop_total
df["hu"] = b.housing_units
df["pct_urban"] = b.pct_urban
df["ur_class"] = b.ur_class
df["ruca"] = b.ruca_primary
df["bldg"] = b.bldg_count
df["bldg_large"] = b.bldg_large_count
df["bldg_missing"] = b.bldg_missing
for c in ["km_motorway", "km_trunk", "km_primary", "km_secondary", "km_tertiary_named",
          "km_highway", "km_highway_ring25", "km_tertiary_named_ring25", "km_total_unfiltered"]:
    df[c] = r[c]
for c in ["poi_total", "poi_fire", "poi_ems", "poi_schools", "poi_education", "poi_health",
          "poi_retail", "poi_food", "poi_government", "infra_total"]:
    df[c] = p[c]
df["poi_commercial"] = df[["poi_retail", "poi_food", "poi_health", "poi_education", "poi_government"]].sum(axis=1)

# ---- building_gap
sph_u, sph_r, cut = 0.55, 1.20, 0.5
urban = df.pct_urban.fillna(0.0)
sph = np.where(urban >= cut, sph_u, sph_r)
hu_eff = df.hu.where(df.hu > 0, df["pop"] / 2.6)
exp_b = hu_eff * sph
df["g_B"] = np.where(exp_b > 0, np.clip(1 - df.bldg / exp_b.replace(0, np.nan), 0, 1), 0.0)

# ---- transport_gap
den = df.km_highway + df.km_tertiary_named + df.km_highway_ring25
df["g_T_cont"] = np.where(den > 0, np.clip(1 - df.km_highway / den.replace(0, np.nan), 0, 1), 0.0)
df["g_T_bin"] = (df.km_highway == 0).astype(float)

# ---- poi_gap
rate = 18 + 14 * urban
exp_p = df["pop"] / 1000 * rate + 0.05 * df.bldg_large
df["g_P"] = np.where(exp_p > 0, np.clip(1 - df.poi_total / exp_p.replace(0, np.nan), 0, 1), 0.0)

for c in ["g_B", "g_T_cont", "g_T_bin", "g_P"]:
    s = df[c]
    print(f"{c}: mean={s.mean():.4f} median={s.median():.4f} p75={s.quantile(.75):.4f} "
          f"p90={s.quantile(.9):.4f} max={s.max():.3f} share>0={(s > 0).mean():.4f} "
          f"share>0.5={(s > 0.5).mean():.4f}")

print("\ng_B by pct_urban bin")
print(df.assign(ub=pd.cut(df.pct_urban, [-.01, .3, .5, .7, .9, 1.01])).groupby("ub", observed=True)
      .g_B.agg(["mean", "median", lambda s: (s > 0).mean()]))
print("\ng_B by region")
print(df.groupby("region").g_B.agg(["mean", "median", "max", lambda s: (s > 0).mean()]))
print("\ng_T_cont by region")
print(df.groupby("region").g_T_cont.agg(["mean", "median", "max", lambda s: (s > 0).mean()]))
print("\ng_P by region")
print(df.groupby("region").g_P.agg(["mean", "median", "max", lambda s: (s > 0).mean()]))
print("\nratio km_highway/den by region")
print((df.km_highway / den.replace(0, np.nan)).groupby(df.region).describe(percentiles=[.1, .5, .9])[["mean", "10%", "50%", "90%"]])
print("\nshare tracts where bldg/hu < 0.55 in urban", float(((df.bldg / df.hu.replace(0, np.nan)) < 0.55)[urban >= cut].mean()))
print("bldg/hu quantiles:", (df.bldg / df.hu.replace(0, np.nan)).quantile([.05, .1, .25, .5]).round(3).to_dict())
df.to_parquet(ROOT / "datos/intermedios/work.parquet")
