"""s09: interacciones baratas + suavizado por condado (leave-one-out).

Salidas:
  datos/features_sonda/s09_interactions.parquet
  datos/features_sonda/s09_interactions.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COMP = ROOT / "datos/features_base/components_cand00.parquet"
POIS = ROOT / "datos/features_base/features_pois.parquet"
NAT = ROOT / "CONCURSO/data/strata/national/national-strata-tract-table.parquet"
OUT = ROOT / "datos/features_sonda"

cols_comp = ["GEOID", "region", "pct_urban", "g_transport", "poi_total", "pop_total", "bldg_count", "km_highway"]
comp = pd.read_parquet(COMP, columns=cols_comp)
comp["GEOID"] = comp["GEOID"].astype(str).str.zfill(11)

comp = comp.merge(pd.read_parquet(POIS, columns=["GEOID", "poi_total"]).rename(
    columns={"poi_total": "poi_total_a04"}), on="GEOID", how="left")
if not comp["poi_total_a04"].fillna(-1).eq(comp["poi_total"]).all():
    raise SystemExit("poi_total discrepa entre A09 y A04")
comp = comp.drop(columns="poi_total_a04")

nat = pd.read_parquet(NAT, columns=["GEOID", "STATEFP", "COUNTYFP", "state_usps", "tribal_any", "ruca_primary"])
nat["GEOID"] = nat["GEOID"].astype(str).str.zfill(11)
df = comp.merge(nat, on="GEOID", how="left", validate="one_to_one")
if df[["STATEFP", "COUNTYFP", "tribal_any"]].isna().any().any():
    raise SystemExit("faltan tractos en la tabla nacional")
if len(df) != 9794 or df["GEOID"].duplicated().any():
    raise SystemExit(f"filas/duplicados inesperados: {len(df)}")

df["county"] = df["STATEFP"] + df["COUNTYFP"]
df["tribal"] = df["tribal_any"].astype(bool).astype(float)

# pct_urban falta en 27 tractos sin poblacion (no estan en la tabla nacional):
# sin poblacion no hay clasificacion rural posible, se trata como no rural.
df["rural"] = (df["pct_urban"] < 0.5).fillna(False).astype(float)

df["x_rural_gtransport"] = df["rural"] * df["g_transport"]
df["x_tribal_rural"] = df["tribal"] * df["rural"]

# Tasa de POIs por 1000 hab. dentro de cada region (denominador comun, no LOO).
df["pop_k"] = df["pop_total"] / 1000.0
rate = df.groupby("region", observed=True)["poi_total"].sum() / df.groupby("region", observed=True)["pop_k"].sum()
df["poi_rate_region"] = df["region"].map(rate)
df["poi_deficit"] = 1.0 - df["poi_total"] / (df["poi_rate_region"] * df["pop_k"])
df.loc[df["pop_k"] <= 0, "poi_deficit"] = 0.0  # sin poblacion no hay deficit medible
df["poi_deficit"] = df["poi_deficit"].clip(0.0, 1.0)


def loo_mean(values: pd.Series, groups: pd.Series) -> pd.Series:
    g = values.groupby(groups.mask(values.isna()))
    s = g.transform("sum") - values.fillna(0.0)
    n = g.transform("count") - values.notna().astype(int)
    out = pd.Series(0.0, index=values.index)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    return out


df["x_county_gtransport"] = loo_mean(df["g_transport"], df["county"])
df["x_county_poi_deficit"] = loo_mean(df["poi_deficit"], df["county"])

FEATS = ["x_rural_gtransport", "x_tribal_rural", "x_county_gtransport", "x_county_poi_deficit"]
out = df[["GEOID"] + FEATS].copy()

assert len(out) == 9794 and not out["GEOID"].duplicated().any()
assert out[FEATS].notna().all().all(), out[FEATS].isna().sum().to_dict()
assert np.isfinite(out[FEATS].to_numpy()).all()
assert (out[FEATS].std(ddof=0) > 0).all(), "columna constante"

OUT.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT / "s09_interactions.parquet", engine="pyarrow", compression="zstd", index=False)

rows = []
for f in FEATS:
    rows.append({
        "feature": f,
        "corr_rural": out[f].corr(df["rural"]),
        "corr_gtransport": out[f].corr(df["g_transport"]),
        "min": out[f].min(),
        "median": out[f].median(),
        "max": out[f].max(),
    })
rep = pd.DataFrame(rows)
fmt = lambda v: f"{v:+.3f}"
lines = [
    "# s09 interacciones + suavizado por condado (9,794 tractos, 0 missing)",
    "",
    "| feature | corr rural | corr g_transport | min | mediana | max |",
    "|---|---:|---:|---:|---:|---:|",
]
for _, r in rep.iterrows():
    lines.append(f"| {r.feature} | {fmt(r.corr_rural)} | {fmt(r.corr_gtransport)} | "
                 f"{r['min']:.3f} | {r['median']:.3f} | {r['max']:.3f} |")
lines.append("")
lines.append(f"325 condados; LOO = (suma - x)/(n-1), 0 si n==1. "
             f"rural=1[pct_urban<0.5] (27 tractos pop=0 sin pct_urban -> rural=0); "
             f"deficit POI vs tasa regional por 1000 hab., recortado [0,1], pop=0 -> 0.")
(OUT / "s09_interactions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(rep.to_string(index=False))
print("\nmissing:", out[FEATS].isna().sum().to_dict())
print("colinealidad max entre features:")
print(out[FEATS].corr().round(3).to_string())
print("\nOK", OUT / "s09_interactions.parquet")
