from pathlib import Path

import numpy as np
import pandas as pd

# Exploracion previa a components.py: solo imprime descriptivos de los tres parquets base.
base = Path(__file__).resolve().parents[2] / "datos" / "features_base"
b = pd.read_parquet(base / "features_buildings.parquet")
r = pd.read_parquet(base / "features_roads.parquet")
p = pd.read_parquet(base / "features_pois.parquet")

print(b.shape, r.shape, p.shape)
print("geoideq", set(b.GEOID) == set(r.GEOID) == set(p.GEOID), b.GEOID.is_unique)
print("\nregion counts\n", b.region.value_counts())

pop = b.pop_total
print("\npop sum by region\n", b.groupby("region").pop_total.sum())
print("housing sum by region\n", b.groupby("region").housing_units.sum())
print("bldg sum by region\n", b.groupby("region").bldg_count.sum())

print("\n--- bldg per housing unit, by urban class ---")
b["r_bh"] = b.bldg_count / b.housing_units.replace(0, np.nan)
print(b.groupby("ur_class").r_bh.describe(percentiles=[.1, .25, .5, .75, .9]))
print("\nby pct_urban bins")
b["u_bin"] = pd.cut(b.pct_urban, [-.01, .1, .3, .5, .7, .9, 1.01])
print(b.groupby("u_bin", observed=True).r_bh.describe(percentiles=[.25, .5, .75]))

print("\n--- road class presence ---")
for c in ["km_motorway", "km_trunk", "km_primary", "km_secondary", "km_tertiary", "km_residential", "km_unclassified"]:
    print(c, "zero share", round(float((r[c] == 0).mean()), 4))
print("motorway+trunk==0", round(float(((r.km_motorway + r.km_trunk) == 0).mean()), 4))
print("km_highway==0", round(float((r.km_highway == 0).mean()), 4))
print("\nby region motorway+trunk==0 & km_highway==0")
g = r.assign(mt=(r.km_motorway + r.km_trunk) > 0, hw=r.km_highway > 0, ring=r.km_highway_ring25 > 0)
print(g.groupby("region")[["mt", "hw", "ring"]].mean())

print("\n--- POIs ---")
print(p.poi_total.describe(percentiles=[.1, .25, .5, .75, .9, .99]))
print("poi per 1k pop:", float((p.poi_total.sum() / b.pop_total.sum()) * 1000))
p2 = p.merge(b[["GEOID", "bldg_large_count", "housing_units"]], on="GEOID")
p2["poi_per_1k"] = p2.poi_total / p2.pop_total.replace(0, np.nan) * 1000
p2["u_bin"] = pd.cut(p2.pct_urban, [-.01, .1, .3, .5, .7, .9, 1.01])
print(p2.groupby("u_bin", observed=True).poi_per_1k.describe(percentiles=[.1, .5, .9]))
print("region poi_per_1k")
print(p2.groupby("region").apply(lambda d: d.poi_total.sum() / d.pop_total.sum() * 1000, include_groups=False))
print("poi_total==0 count", int((p.poi_total == 0).sum()), "pop==0", int((b.pop_total == 0).sum()))
print("corr poi_total~pop", float(np.corrcoef(p.poi_total, b.pop_total)[0, 1]))
