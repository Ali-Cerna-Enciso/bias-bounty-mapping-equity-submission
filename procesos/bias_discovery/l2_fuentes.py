"""L2: la cobertura de edificios de Overture cambia segun la fuente que la alimenta?

Por fuente dominante dentro de region x ruralidad: n, mediana Overture/USFS,
Overture/CarbonPlan, IC bootstrap 95% (semilla 42) y los tractos mas extremos.
USFS y CarbonPlan: comparacion diagnostica.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(42)
CSV = ROOT / "datos" / "bias_discovery" / "l2_fuentes.csv"
SHARES = ["src_microsoft_share", "src_osm_share", "src_google_share",
          "src_other_share", "src_missing_share"]
LABELS = np.array(["microsoft", "osm", "google", "esri_usgs_lidar", "missing"])


def boot_median(x, b=2000):
    if len(x) < 20:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(b, len(x)))
    med = np.median(x[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def load():
    b = pd.read_parquet(ROOT / "datos/features_base/features_buildings.parquet")
    s = pd.read_parquet(
        ROOT / "CONCURSO/data/strata/national/national-strata-tract-table.parquet",
        columns=["GEOID", "COUNTYFP", "state_usps", "aiannh_name", "aiannh_kind",
                 "tribal_pct", "usfs_BuildingCount_sum", "carbonplan_buildings",
                 "usfs_CRPS_mean", "svi_overall"])
    b["GEOID"] = b["GEOID"].astype(str)
    s["GEOID"] = s["GEOID"].astype(str)
    d = b.merge(s, on="GEOID", how="left", validate="one_to_one")
    d["dom"] = LABELS[d[SHARES].to_numpy().argmax(axis=1)]
    d["ruralidad"] = np.where(d["pct_urban"] < 0.5, "rural", "urbano")
    d["eu"] = d["bldg_count"] / d["usfs_BuildingCount_sum"].replace(0, np.nan)
    d["ec"] = d["bldg_count"] / d["carbonplan_buildings"].replace(0, np.nan)
    d["eh"] = d["bldg_count"] / d["housing_units"].replace(0, np.nan)
    d["lugar"] = np.where(
        d["tribal_pct"].fillna(0) > 0.5,
        d["aiannh_name"].fillna("tribal sin nombre"),
        d["state_usps"].fillna("??") + "-" + d["COUNTYFP"].fillna("???"))
    return d[d["bldg_count"] > 0].copy()


def celdas(d):
    ok = d[d["eu"].notna() & d["ec"].notna()]
    rows = []
    for (reg, ru, dom), g in ok.groupby(["region", "ruralidad", "dom"]):
        if len(g) < 5:
            continue
        lo_u, hi_u = boot_median(g["eu"].to_numpy())
        lo_c, hi_c = boot_median(g["ec"].to_numpy())
        rows.append(dict(kind="celda", region=reg, ruralidad=ru, fuente=dom,
                         n=len(g), med_eu=round(g["eu"].median(), 4),
                         eu_lo=round(lo_u, 4) if lo_u == lo_u else np.nan,
                         eu_hi=round(hi_u, 4) if hi_u == hi_u else np.nan,
                         med_ec=round(g["ec"].median(), 5),
                         ec_lo=round(lo_c, 5) if lo_c == lo_c else np.nan,
                         ec_hi=round(hi_c, 5) if hi_c == hi_c else np.nan,
                         med_eh=round(g["eh"].median(), 3)))
    return pd.DataFrame(rows)


def extremos(d, k=10):
    ok = d[d["eu"].notna()].copy()
    cols = ["kind", "region", "ruralidad", "fuente", "n", "GEOID", "lugar",
            "tribal_pct", "bldg_count", "usfs_BuildingCount_sum",
            "carbonplan_buildings", "med_eu", "med_ec", "med_eh"]
    out = []
    for etiqueta, sub in (("extremo_alto", ok.nlargest(k, "eu")),
                          ("extremo_bajo", ok.nsmallest(k, "eu"))):
        for _, r in sub.iterrows():
            out.append(dict(kind=etiqueta, region=r.region, ruralidad=r.ruralidad,
                            fuente=r.dom, n=1, GEOID=r.GEOID, lugar=r.lugar,
                            tribal_pct=round(float(r.tribal_pct or 0), 3),
                            bldg_count=int(r.bldg_count),
                            usfs_BuildingCount_sum=round(float(r.usfs_BuildingCount_sum), 1),
                            carbonplan_buildings=round(float(r.carbonplan_buildings), 1),
                            med_eu=round(float(r.eu), 3),
                            med_ec=round(float(r.ec), 4),
                            med_eh=round(float(r.eh), 3) if r.eh == r.eh else np.nan))
    return pd.DataFrame(out)[cols]


def controles(d):
    print("\n== CONTROL 1: son independientes las referencias? ==")
    v = d[d["eu"].notna() & d["ec"].notna()]
    ec = v["carbonplan_buildings"] / v["bldg_count"]
    print(f"CarbonPlan/bldg_count: mediana {ec.median():.5f}, IQR {ec.quantile(.25):.5f}-{ec.quantile(.75):.5f}, "
          f"corr(log) {np.corrcoef(np.log1p(v.carbonplan_buildings), np.log1p(v.bldg_count))[0,1]:.5f}")
    eu = v["usfs_BuildingCount_sum"] / v["bldg_count"]
    print(f"USFS/bldg_count     : mediana {eu.median():.5f}, IQR {eu.quantile(.25):.5f}-{eu.quantile(.75):.5f}, "
          f"corr(log) {np.corrcoef(np.log1p(v.usfs_BuildingCount_sum), np.log1p(v.bldg_count))[0,1]:.5f}")

    print("\n== CONTROL 2: el efecto sobrevive dentro del condado? (>=5 tractos por fuente) ==")
    v = v.assign(g2=np.where(v.src_microsoft_share >= .5, "ms",
                             np.where(v.src_osm_share >= .5, "osm", "otra")))
    t = v.groupby(["state_usps", "COUNTYFP", "g2"]).eu.agg(["size", "median"]).unstack("g2")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    par = t[(t.get("size_ms", 0) >= 5) & (t.get("size_osm", 0) >= 5)].dropna(subset=["median_ms", "median_osm"])
    gap = par["median_osm"] - par["median_ms"]
    print(par.round(3).to_string())
    print(f"brecha osm-ms: mediana {gap.median():+.3f}, positiva en {int((gap>0).sum())}/{len(gap)} condados")

    print("\n== CONTROL 3: Yakama (la pista) frente a su propio condado ==")
    w = d[(d.region == "eastern-wa") & d["eu"].notna()].copy()
    yak = w["aiannh_name"].fillna("").str.contains("Yakama")
    resto = w[(w.COUNTYFP == "077") & ~yak]
    for nombre, sub in (("Yakama (13 tractos)", w[yak]), ("resto condado WA-077", resto),
                        ("eastern-wa entero", w)):
        print(f"{nombre:24s} n={len(sub):3d} eu={sub.eu.median():.3f} eh={sub.eh.median():.3f} "
              f"ms={sub.src_microsoft_share.median():.3f} osm={sub.src_osm_share.median():.3f}")


def main():
    d = load()
    print(f"tractos {len(d)}; con ambas referencias {int((d.eu.notna() & d.ec.notna()).sum())}")
    c = celdas(d)
    print("\n== Fuente dominante x region x ruralidad (USFS / CarbonPlan, IC95 bootstrap) ==")
    for reg in sorted(c.region.unique()):
        print(f"\n[{reg}]")
        print(c[c.region == reg][["ruralidad", "fuente", "n", "med_eu", "eu_lo", "eu_hi",
                                  "med_ec", "ec_lo", "ec_hi", "med_eh"]].to_string(index=False))
    print("\n== Direccion del efecto (OSM menos Microsoft), por region x ruralidad ==")
    p = c.pivot_table(index=["region", "ruralidad"], columns="fuente", values="med_eu")
    if {"microsoft", "osm"} <= set(p.columns):
        print((p["osm"] - p["microsoft"]).round(3).to_string())
    x = extremos(d)
    print("\n== 10 tractos mas extremos por lado ==")
    print(x[["kind", "region", "GEOID", "lugar", "fuente", "bldg_count",
             "usfs_BuildingCount_sum", "med_eu", "med_eh"]].to_string(index=False))
    controles(d)
    out = pd.concat([c, x], ignore_index=True)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV, index=False)
    print(f"\nescrito {CSV} ({len(out)} filas)")


if __name__ == "__main__":
    main()
